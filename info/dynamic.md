# dynamic quantization 与非动态路径的差异

- Hexagon 后端的流水线包含独立的 Dynamic Quantizer 阶段，可通过 `GGML_HEXAGON_OPMASK` 的 `0x2` 位开启，仅量化或组合其它阶段；不使用动态量化时可跳过该阶段直接复用已有量化张量或只排队/计算。(docs/backend/hexagon/README.md#228-239, ggml/src/ggml-hexagon/htp/htp-msg.h#12,#18)
- 在 matmul 入口，如果未设置 `HTP_OPFLAGS_SKIP_QUANTIZE`，会先将 fp32 输入按块量化为 q8x4x2，并把量化任务分发给 worker 池；跳过动态量化则复用已经量化的 src1，不执行这一步骤。(ggml/src/ggml-hexagon/htp/matmul-ops.c#1886,#2021-2026)
- 为支撑动态量化，src0 的 VTCM scratchpad 需要按量化后行大小扩张，以临时存放填充后的 src1 行，因此有额外的 VTCM 需求；跳过量化时不需要这一额外缓冲。(ggml/src/ggml-hexagon/htp/matmul-ops.c#1907-1911,#1936-1940)
