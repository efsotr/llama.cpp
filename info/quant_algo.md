# llama.cpp 量化算法概览

本文概述 llama.cpp 中的核心量化实现，并用伪代码概括流程，方便理解各类块量化与重要性矩阵量化策略。引用格式为 `文件#行号范围`。

## 4-bit 对称量化（Q4_0）

算法对每个固定长度块（`QK4_0`）寻找绝对值最大元素，生成尺度 `d = max / -8`，再将值按 `x * (1/d) + 8.5` 取整裁剪到 4 bit，低高 4 位分别写入同一字节。伪代码：

```
for block in chunks(x, QK4_0):
    amax, max_val = max_abs(block)
    d = max_val / -8
    id = 1 / d
    for j in 0..QK4_0/2-1:
        xi0 = clamp(int(block[j]   * id + 8.5), 0, 15)
        xi1 = clamp(int(block[j+QK4_0/2] * id + 8.5), 0, 15)
        pack_4bit(xi0, xi1)
    store_scale(d)
```

引用：`ggml/src/ggml-quants.c#36-69`

## 4-bit 带偏移量化（Q4_1）

对块内求 `min` 与 `max`，步长 `d = (max - min) / 15`，以 `(x - min) / d` 量化并存储偏移 `min`。伪代码：

```
for block in chunks(x, QK4_1):
    mn, mx = min(block), max(block)
    d = (mx - mn) / 15
    id = 1 / d
    for j in 0..QK4_1/2-1:
        xi0 = clamp(int((block[j]   - mn) * id + 0.5), 0, 15)
        xi1 = clamp(int((block[j+QK4_1/2] - mn) * id + 0.5), 0, 15)
        pack_4bit(xi0, xi1)
    store_scale(d), store_min(mn)
```

引用：`ggml/src/ggml-quants.c#73-108`

## 5-bit 量化（Q5_0 与 Q5_1）

Q5_0 与 Q5_1 与 Q4 系列类似，但使用 5 bit。低 4 bit 与额外的第 5 bit 拆分存储到 `qs` 与 `qh`。Q5_0 为对称量化，Q5_1 记录偏移 `min`。伪代码（对称版）：

```
for block in chunks(x, QK5):
    amax, max_val = max_abs(block)
    d = max_val / -16
    id = 1 / d
    for j in 0..QK5/2-1:
        xi0 = clamp(int(block[j]   * id + 16.5), 0, 31)
        xi1 = clamp(int(block[j+QK5/2] * id + 16.5), 0, 31)
        qs[j]  = low4(xi0) | (low4(xi1) << 4)
        qh    |= bit5(xi0) at pos j | bit5(xi1) at pos j+QK5/2
    store_scale(d), store_qh(qh) [, store_min for Q5_1]
```

引用：`ggml/src/ggml-quants.c#110-195`

## 8-bit 量化（Q8_0 与 Q8_1）

Q8_0 以块内绝对值最大值求尺度 `d = amax / 127`，直接将 `x * (1/d)` 四舍五入为 8 bit；Q8_1 额外累加量化后和 `sum*d` 作为补偿 `s`。伪代码：

```
for block in chunks(x, QK8):
    amax = max_abs(block)
    d = amax / 127
    id = 1 / d
    sum = 0
    for j in 0..QK8-1:
        q = round(block[j] * id)
        qs[j] = q
        sum += q              // 仅 Q8_1
    store_scale(d)
    store_sum(sum * d)        // 仅 Q8_1
```

引用：`ggml/src/ggml-quants.c#198-258`

## K-quant 超块量化（以 Q4_K 为例）

K-quant 在超块（`QK_K`）内按 32 子块分别求带权尺度与最小值（使用均值与绝对值混合权重），然后将所有子块尺度/最小值按全局最大尺度、全局最大最小值重新归一化并压缩存入 `scales`。最终用各子块的归一化尺度与最小值反推实际量化值并写入比特流。伪代码：

```
for superblock in chunks(x, QK_K):
    for each 32-lane subblock:
        weights = mean_abs(subblock) + abs(subblock)
        scale, min = weighted_quantize(subblock, weights)
        record scale, min, quantized L
        track max_scale, max_min
    inv_scale = 63 / max_scale
    inv_min   = 63 / max_min
    encode scales,mins into 6-bit packed array
    d    = max_scale / 63
    dmin = max_min   / 63
    reconstruct per-subblock scale/min and write quant bits
```

引用：`ggml/src/ggml-quants.c#1280-1334`

## 激活感知 i-Quant（以 IQ2_XXS 为例）

对每行数据按 `QK_K` 大小划分超块，调用 `quantize_row_iq2_xxs_impl` 使用重要性矩阵（activation-aware 权重）求尺度并量化，结果大小为 `nblock * sizeof(block_iq2_xxs)`。伪代码：

```
for row in rows:
    for superblock in chunks(row, QK_K):
        quantize_row_iq2_xxs_impl(superblock, imatrix_weights)
    advance pointers
```

引用：`ggml/src/ggml-quants.c#3383-3393`
