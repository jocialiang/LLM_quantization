# Lab — Qwen2.5-3B 量化實驗 (GPTQ W4A16 / SmoothQuant W8A8-O3)

Qwen2.5-3B 在單張 T4 上的量化與推論評估。流程：量化 → perplexity → prefill/decode 速度 → vLLM profiler。

## 實驗設定（所有腳本共用，定義在 `scripts/common.py`）

| 項目 | 設定 |
|---|---|
| 模型 | `Qwen/Qwen2.5-3B`，FP16 |
| Calibration | WikiText-2 train，128 筆 × 512 tokens，seed 42 |
| GPTQ | W4A16，`ignore = ["lm_head"]` |
| SmoothQuant O3 | SmoothQuant (strength 0.8) + GPTQ，W8A8；weight per-tensor INT8、activation per-tensor **static** INT8；`ignore = ["lm_head", "re:.*down_proj"]`（down_proj 不量化） |
| 推論引擎 | vLLM，`dtype=half`、`max_num_seqs=1`、`attention backend = TRITON_ATTN` |

> O3 不量化 down_proj，所以原本的 `down_proj ← up_proj` SmoothQuant mapping 也一併移除（down_proj 維持 FP16，對它做 smoothing 沒有好處）。

## 專案結構

```
.
├── USAGE.md
├── requirements-quant.txt      # 環境 A
├── requirements-vllm.txt       # 環境 B
├── scripts/
│   ├── 01_install_env.sh
│   ├── 02_quantize.py
│   ├── 03_perplexity.py
│   ├── 04_prefill_decode.py
│   ├── 05_vllm_profiler.py
│   └── common.py               # 共用設定
├── .venv-quant/                # 01_install_env.sh 建立
├── .venv-vllm/
└── lab_results/                # 執行後產生（checkpoint、結果 CSV/JSON、profiling json）
```

所有指令都在專案根目錄執行。

## 檔案一覽

| 檔案 | 用途 | 環境 |
|---|---|---|
| `requirements-quant.txt` | 量化環境套件（版本固定） | A |
| `requirements-vllm.txt` | vLLM 評估/profiling 環境套件（版本固定） | B |
| `scripts/01_install_env.sh` | 建立兩個 venv 並安裝 | — |
| `scripts/common.py` | 共用設定：模型、路徑、prompt、calibration、GPU 記憶體監控 | A、B |
| `scripts/02_quantize.py` | 產生 GPTQ 與 SmoothQuant-O3 checkpoint | A |
| `scripts/03_perplexity.py` | WikiText-2 perplexity 比較 | B |
| `scripts/04_prefill_decode.py` | prefill / decode 速度與 GPU 記憶體 | B |
| `scripts/05_vllm_profiler.py` | 錄製 vLLM profiler，只輸出 `profiling_*.json` | B |

**為什麼要兩個環境？** `llmcompressor 0.15.0` 固定 `compressed-tensors==0.20.1`，`vllm 0.34.0` 固定 `compressed-tensors==0.17.0`，兩者不能裝在同一個環境。量化用環境 A，其餘全部用環境 B。

## 1. 安裝

```bash
bash scripts/01_install_env.sh          # 兩個環境都裝（.venv-quant 與 .venv-vllm）
bash scripts/01_install_env.sh quant    # 只裝量化環境
bash scripts/01_install_env.sh vllm     # 只裝 vLLM 環境
```

安裝完會印出實際版本。需要 NVIDIA GPU 與對應的 CUDA driver。

輸出位置預設為專案根目錄下的 `lab_results/`（與從哪個目錄執行無關），可用環境變數改：

```bash
export LAB_ROOT=/content/lab_results
```

## 2. 量化 — `02_quantize.py`（環境 A）

```bash
.venv-quant/bin/python scripts/02_quantize.py                      # GPTQ + SmoothQuant-O3
.venv-quant/bin/python scripts/02_quantize.py --method gptq
.venv-quant/bin/python scripts/02_quantize.py --method smoothquant_o3
```

輸出：

- `$LAB_ROOT/Qwen2.5-3B-GPTQ-W4A16/`
- `$LAB_ROOT/Qwen2.5-3B-SmoothQuant-W8A8-O3-ignore-down_proj/`

量化時 GPU 需要額外工作記憶體；若 OOM，兩個方法分開跑（`--method`）。

## 3. Perplexity — `03_perplexity.py`（環境 B）

```bash
.venv-vllm/bin/python scripts/03_perplexity.py                       # 全部模型，20 chunks（與 notebook 相同）
.venv-vllm/bin/python scripts/03_perplexity.py --max-chunks 0        # 完整 WikiText-2 test set
.venv-vllm/bin/python scripts/03_perplexity.py --models FP16 GPTQ    # 只跑指定模型
```

每個模型獨立一個 process 執行（避免 vLLM 沒釋放顯存）。用 `prompt_logprobs` 對每個 512-token chunk 做一次 prefill 計算 PPL。預設 20 chunks 是 notebook 的除錯設定，要正式比較請用 `--max-chunks 0`。

輸出：`ppl_results.csv` / `ppl_results.json`（含相對 FP16 的 ΔPPL；沒跑 FP16 則無 Δ 欄位）。

## 4. Prefill / Decode — `04_prefill_decode.py`（環境 B）

```bash
.venv-vllm/bin/python scripts/04_prefill_decode.py                                   # notebook 的 400 字 prompt
.venv-vllm/bin/python scripts/04_prefill_decode.py --prefill-tokens 128 512 1024     # 指定 prefill 長度（可多個）
.venv-vllm/bin/python scripts/04_prefill_decode.py --models FP16 SmoothQuant_O3
.venv-vllm/bin/python scripts/04_prefill_decode.py --decode-tokens 256
```

| 參數 | 說明 |
|---|---|
| `--prefill-tokens` | prefill 的 prompt 長度（token 數），可一次給多個。長度精確等於指定值，上限 2047。不給就用 notebook 的 400 字 prompt |
| `--decode-tokens` | decode 測試的 `max_tokens` N，預設 128 |
| `--models` | 只跑指定模型 |

量測方式（batch size 1，CUDA graph 開啟，2 次 warmup + 3 次計時）：

- **Prefill**：`max_tokens=1`，整段 prompt 在一次 forward 內處理完並產生第 1 個 token，所以 latency ≈ time to first token。`prefill tok/s = prompt tokens / latency`。每個長度各一列結果。
- **Decode**：`max_tokens=N` 的 N 是**輸出 token 總數**，其中第 1 個 token 是 prefill 那一次 forward 產生的，剩下 N−1 個才是 decode step。所以：
  - 跑一次 `max_tokens=N`，耗時 `T_N` = prefill + (N−1) 次 decode
  - 再跑一次 `max_tokens=1`（同一個短 prompt），耗時 `T_1` = 只有 prefill
  - 每 token decode 時間 = `(T_N − T_1) / (N − 1)`

  例：N=128，`T_128 = 3.20 s`、`T_1 = 0.05 s` → decode = (3.20 − 0.05) / 127 ≈ 24.8 ms/token ≈ 40 tok/s。
  多跑一次 `T_1` 只是為了把 prefill 時間扣掉，讓 decode 速度不被 prompt 長度影響；`ignore_eos=True` 確保一定生成滿 N 個 token。
- 同時記錄 GPU 記憶體 baseline / peak（nvidia-smi，整段執行期間，含 KV cache 與 CUDA graph）。

輸出：`prefill_results.csv`（每個模型 × 每個 prefill 長度一列）、`decode_results.csv`（每個模型一列，含 GPU 記憶體）、`prefill_decode_results.json`（含每次 run 的原始數據）。

## 5. vLLM profiler — `05_vllm_profiler.py`（環境 B）

只錄製並存下 trace JSON，不做任何分析。

```bash
.venv-vllm/bin/python scripts/05_vllm_profiler.py --model FP16                 # eager（預設，每個 op 都看得到）
.venv-vllm/bin/python scripts/05_vllm_profiler.py --model FP16 --cuda-graph    # CUDA graph 開啟
.venv-vllm/bin/python scripts/05_vllm_profiler.py --model SmoothQuant_O3
.venv-vllm/bin/python scripts/05_vllm_profiler.py --model GPTQ --output profiling.json   # 自訂檔名
```

`--model` 可選 `FP16`、`GPTQ`、`SmoothQuant_O3`。預設輸出 `$LAB_ROOT/profiling_<model>_<eager|cudagraph>.json`，一次只 profile 一個 generation（warmup 1 次，生成 32 tokens）。

### 用 Perfetto 開啟

1. 瀏覽器開 <https://ui.perfetto.dev/>。
2. 左側 **Open trace file**，選 `profiling_*.json`（或直接把檔案拖進頁面）。
3. 常用操作：`W`/`S` 放大縮小、`A`/`D` 左右移動；點 slice 看 duration 與 args；上方搜尋列可搜尋 slice 名稱（例如 `Qwen2MLP`、`Qwen2Attention`、`RMSNorm`、`gemv`、`cutlass`、`scaled_int8_quant`），名稱依 vLLM 版本可能不同。
4. 要比較 FP16 / GPTQ / SmoothQuant，各自存一個 json，分開開兩個分頁對照。

JSON 約數十 MB，Perfetto 可直接開。檔案只在瀏覽器本機解析，不會上傳。

## 建議執行順序

```bash
bash scripts/01_install_env.sh
.venv-quant/bin/python scripts/02_quantize.py
.venv-vllm/bin/python  scripts/03_perplexity.py --max-chunks 0
.venv-vllm/bin/python  scripts/04_prefill_decode.py
.venv-vllm/bin/python  scripts/05_vllm_profiler.py --model FP16
.venv-vllm/bin/python  scripts/05_vllm_profiler.py --model SmoothQuant_O3
```

## 注意事項

- `requirements-*.txt` 的版本是依 PyPI 相依性解析（dry-run）選出的一組不衝突版本，**尚未在 GPU 上實際跑過**。若你在 Colab 跑通的版本不同，請以該環境的 `pip freeze` 結果覆蓋對應版本再發布。
- T4（sm75）沒有 FlashAttention-2，因此固定使用 `TRITON_ATTN`；換 GPU 可在 `scripts/common.py` 修改 `ATTENTION_BACKEND`。
