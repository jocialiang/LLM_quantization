# LLM_quantization
> 在沒有 NPU 的情況下，以 NVIDIA T4 作為替代平台，分別從 **Prefill（compute-bound）** 與 **Decode（memory-bound）** 兩個階段，驗證量化對延遲與記憶體的影響，並討論其在 NPU 上的意義。同時使用**activation outlier統計**與 **time profiling方法**分析 LLM 量化結果，做為改良方案依據。
