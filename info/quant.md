## K-quant 简述

K-quant（如 Q2_K/Q3_K/Q4_K/Q5_K/Q6_K）在每个 `QK_K` 大小的 block 上分组量化，先为子块计算尺度和偏移，再把实值映射到整数 codebook：先对 32/16 元分组做 `make_qkx*`/`make_qx_quants` 得到 scale/min，随后根据最优尺度把每个元素压缩到低比特并按高位掩码拆分存入 `qs` 与 `hmask` 等字段。伪代码（忽略具体 codebook 查找）：

```
for each block in rows:
  stats = collect(x_subblock)
  (scale, min) = choose_scales(stats)
  codes = quantize_with_scale(x_subblock, scale, min)
  pack codes into qs/hmask, store scale/min (d, dmin)
```

量化时如果提供了 `quant_weights`（重要性权重），会在选择尺度前按权重重排误差以降低重要位置的失真，再执行同样的打包流程。`quantize_q3_K`/`quantize_q4_K` 等实现中对于每个 16/32 长度分组使用 `quant_weights` 调整 `weight` 向量后再调用内部求尺度例程。见 `ggml/src/ggml-quants.c#1178-1445`、`ggml/src/ggml-quants.c#1396-1444`。

## I-quant 与 imatrix

I-quant 系列（IQ1/IQ2/IQ3/IQ4_XS/IQ4_NL 等）是「importance-aware」量化，可选传入每列的重要性矩阵 `imatrix` 作为 `quant_weights`。在量化流程中：

1) `llama_model_quantize_impl` 在处理每个张量时，从用户参数 `params->imatrix` 查找同名权重；若找到且大小匹配 `ne[0] * ne[2]`（列数 × expert 轴），则传给后端量化函数；尺寸不符会报错避免静默退化。见 `src/llama-quant.cpp#914-934`。  
2) 极低比特类型（IQ2_XXS/IQ2_XS/IQ2_S/IQ1_S/IQ1_M、以及 Q2_K_S）若未提供 imatrix 会直接报错，因为缺少重要性信息会导致质量严重下降。见 `src/llama-quant.cpp#937-948`。  
3) `ggml_quantize_chunk` 将 `imatrix` 作为 `quant_weights` 传入各量化核，I-quant 类型也在此分支中被断言必须带 `imatrix`。见 `ggml/src/ggml.c#7485-7524`。

在具体量化核里，`quant_weights` 按列（长度 = `n_per_row`，对于多 expert 追加到 `n_per_row * ne[2]`）与输入逐元素结合，例如 IQ/K 系列会对每 16/32 元小组以 `weight[j] = qw[j] * f(x_j)` 的形式重新加权后再搜索最佳尺度和码字，从而让重要列（激活较大）量化误差更小。见 `ggml/src/ggml-quants.c#1396-1444`、`ggml/src/ggml-quants.c#2060-2070`。

## I-quant imatrix 的 shape 与作用

* **Shape**：`imatrix` 为浮点数组，长度必须等于 `n_per_row * ne[2]`，即每列（`ne[0]`）乘以可能的 expert 轴（`ne[2]`）；矩阵按列主序展开，与待量化张量的列对齐，专家权重则按第三维逐块拼接。见 `src/llama-quant.cpp#914-934`。  
* **作用**：为每个元素提供重要性权重，量化核把该权重乘到输入或其方差上，再求尺度与码字，使误差在高重要性位置更小；部分低比特类型强制要求提供，否则拒绝量化。见 `ggml/src/ggml.c#7485-7524`、`ggml/src/ggml-quants.c#1396-1444`、`ggml/src/ggml-quants.c#2060-2070`。

伪代码（I-quant 有 imatrix 情况）：

```
imatrix = load(weights, shape = n_per_row * ne2)
for each expert e:
  w = imatrix[e * n_per_row : (e+1)*n_per_row]
  for each block in rows:
    weighted = apply_weights(x_block, w_block)
    (scale, code) = solve_best_quant(weighted)
    pack(code, scale)
```

当 `imatrix` 缺失时，I-quant 会退化或报错（取决于类型），无法保证精度。见 `src/llama-quant.cpp#937-948`。
