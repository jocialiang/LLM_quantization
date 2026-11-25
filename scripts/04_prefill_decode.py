"""Lab - prefill / decode speed of FP16 / GPTQ / SmoothQuant-O3 with vLLM.

Run inside the vLLM env (.venv-vllm):

    python scripts/04_prefill_decode.py                              # all models, notebook prompt
    python scripts/04_prefill_decode.py --models FP16 GPTQ           # subset
    python scripts/04_prefill_decode.py --prefill-tokens 128 512 1024 --decode-tokens 128

Settings: batch size 1 (max_num_seqs=1), CUDA graph ON, 2 warmup + 3 timed runs.

Prefill
    Prompt length is adjustable with --prefill-tokens (one or more lengths, exact token counts).
    Without it, the 400-word prompt of the Lab notebook is used.
    Generate max_tokens=1 -> latency ~= time to first token (all prompt tokens are processed
    in one forward pass).  prefill tok/s = prompt_tokens / latency.

Decode
    Short prompt, max_tokens=N (--decode-tokens, default 128; N counts ALL output tokens,
    the first one comes out of the prefill pass, so only N-1 are real decode steps).
        T_N = prefill + (N-1) decode steps        (max_tokens=N)
        T_1 = prefill only                        (max_tokens=1, same prompt)
        decode time per token = (T_N - T_1) / (N - 1)

GPU memory: baseline / peak from nvidia-smi over the whole run (weights + KV cache + CUDA graphs).

Outputs (in $LAB_ROOT): prefill_decode_<model>.json per model, then
prefill_results.csv, decode_results.csv and prefill_decode_results.json.
"""
import argparse
import gc
import json
import os
import statistics
import time

import common as C

MAX_MODEL_LEN = 2048
GPU_MEMORY_UTILIZATION = 0.35
WARMUP_RUNS = 2
BENCHMARK_RUNS = 3


def timed_generate(llm, prompt, sampling_params):
    start = time.perf_counter()
    outputs = llm.generate([prompt], sampling_params, use_tqdm=False)
    elapsed = time.perf_counter() - start
    out = outputs[0]
    return elapsed, len(out.prompt_token_ids), len(out.outputs[0].token_ids)


def build_prefill_prompts(prefill_tokens):
    """Return [(label, prompt)].  prompt is text (notebook default) or {'prompt_token_ids': [...]}."""
    if not prefill_tokens:
        return [("notebook-400-words", C.build_long_prompt())]

    import transformers

    too_long = [n for n in prefill_tokens if n < 1 or n + 1 > MAX_MODEL_LEN]
    if too_long:
        raise SystemExit(f"--prefill-tokens must be between 1 and {MAX_MODEL_LEN - 1}, got {too_long}")

    tokenizer = transformers.AutoTokenizer.from_pretrained(C.MODEL_ID)
    return [
        (f"{n}-tokens", {"prompt_token_ids": C.build_prompt_token_ids(tokenizer, n)})
        for n in prefill_tokens
    ]


def benchmark(model_name, model_path, decode_tokens, prefill_tokens):
    import torch
    from vllm import LLM, SamplingParams

    print("\n" + "=" * 70)
    print("MODEL:", model_name)
    print("PATH :", model_path)
    print("=" * 70)
    C.check_model_path(model_name, model_path)

    if decode_tokens < 2:
        raise SystemExit("--decode-tokens must be >= 2")
    prefill_prompts = build_prefill_prompts(prefill_tokens)
    short_prompt = C.SHORT_PROMPT

    monitor = C.GPUMemoryMonitor(interval=0.1)
    monitor.start()

    llm = LLM(
        model=model_path,
        dtype="half",
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        max_num_seqs=1,
        enforce_eager=False,
        enable_prefix_caching=False,
        attention_config={"backend": C.ATTENTION_BACKEND},
    )

    sp_1 = SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True)
    sp_n = SamplingParams(temperature=0.0, max_tokens=decode_tokens, ignore_eos=True)

    # warmup (graph capture / kernel autotune)
    for _ in range(WARMUP_RUNS):
        llm.generate([prefill_prompts[0][1]], sp_1, use_tqdm=False)
        llm.generate([short_prompt], sp_n, use_tqdm=False)

    # ---- prefill (one entry per prompt length) ----
    prefill_results = []
    for label, prompt in prefill_prompts:
        runs = []
        for i in range(BENCHMARK_RUNS):
            t, n_prompt, _ = timed_generate(llm, prompt, sp_1)
            runs.append({"run": i + 1, "latency_s": t, "prompt_tokens": n_prompt,
                         "prefill_tokens_per_s": n_prompt / t})
            print(f"[prefill {label}] run {i+1}: {t*1000:.1f} ms, {n_prompt} tokens, "
                  f"{n_prompt / t:.1f} tok/s")
        prefill_results.append({
            "label": label,
            "prompt_tokens": runs[0]["prompt_tokens"],
            "latency_s": statistics.mean(r["latency_s"] for r in runs),
            "tokens_per_s": statistics.mean(r["prefill_tokens_per_s"] for r in runs),
            "runs": runs,
        })

    # ---- decode ----
    decode_runs = []
    for i in range(BENCHMARK_RUNS):
        t_1, _, _ = timed_generate(llm, short_prompt, sp_1)          # prefill only
        t_n, n_prompt, n_gen = timed_generate(llm, short_prompt, sp_n)  # prefill + (N-1) decode steps
        per_token = (t_n - t_1) / (n_gen - 1)
        decode_runs.append({"run": i + 1, "latency_1tok_s": t_1, "latency_Ntok_s": t_n,
                            "generated_tokens": n_gen,
                            "decode_ms_per_token": per_token * 1000,
                            "decode_tokens_per_s": 1.0 / per_token})
        print(f"[decode ] run {i+1}: T_1={t_1*1000:.1f} ms, T_{n_gen}={t_n*1000:.1f} ms, "
              f"{per_token*1000:.2f} ms/token, {1.0 / per_token:.1f} tok/s")

    mem = monitor.stop()
    print("\nGPU memory:", mem)

    result = {
        "model": model_name,
        "model_path": model_path,
        "prefill": prefill_results,
        "decode": {
            "prompt_tokens": n_prompt,
            "generated_tokens": decode_tokens,
            "decode_ms_per_token": statistics.mean(r["decode_ms_per_token"] for r in decode_runs),
            "decode_tokens_per_s": statistics.mean(r["decode_tokens_per_s"] for r in decode_runs),
            "runs": decode_runs,
        },
        **mem,
    }

    del llm
    gc.collect()
    torch.cuda.empty_cache()
    return result


def summarize(names):
    import pandas as pd

    results = []
    for n in names:
        path = os.path.join(C.OUTPUT_ROOT, f"prefill_decode_{n}.json")
        if os.path.exists(path):
            with open(path) as f:
                results.append(json.load(f))
    if not results:
        return

    prefill_df = pd.DataFrame([
        {
            "Model": r["model"],
            "Prefill tokens": p["prompt_tokens"],
            "Prefill latency (s)": round(p["latency_s"], 4),
            "Prefill tok/s": round(p["tokens_per_s"], 1),
        }
        for r in results for p in r["prefill"]
    ])
    decode_df = pd.DataFrame([
        {
            "Model": r["model"],
            "Output tokens (N)": r["decode"]["generated_tokens"],
            "Decode ms/token": round(r["decode"]["decode_ms_per_token"], 2),
            "Decode tok/s": round(r["decode"]["decode_tokens_per_s"], 1),
            "GPU baseline (GB)": round(r["gpu_memory_baseline_GB"], 2),
            "GPU peak (GB)": round(r["gpu_memory_peak_GB"], 2),
        }
        for r in results
    ])
    print("\n[Prefill]\n" + prefill_df.to_string(index=False))
    print("\n[Decode]\n" + decode_df.to_string(index=False))

    with open(os.path.join(C.OUTPUT_ROOT, "prefill_decode_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    prefill_df.to_csv(os.path.join(C.OUTPUT_ROOT, "prefill_results.csv"), index=False)
    decode_df.to_csv(os.path.join(C.OUTPUT_ROOT, "decode_results.csv"), index=False)
    print("\nSaved: prefill_results.csv, decode_results.csv, prefill_decode_results.json in", C.OUTPUT_ROOT)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="*", help=f"subset of {list(C.MODEL_PATHS)}")
    parser.add_argument("--prefill-tokens", type=int, nargs="*", default=None,
                        help="prompt length(s) in tokens for the prefill test, e.g. 128 512 1024 "
                             f"(max {MAX_MODEL_LEN - 1}). Default: the notebook's 400-word prompt")
    parser.add_argument("--decode-tokens", type=int, default=128,
                        help="max_tokens N for the decode test, counting all output tokens (default 128)")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    models = C.select_models(args.models)

    if not args.worker:
        extra = ["--decode-tokens", str(args.decode_tokens)]
        if args.prefill_tokens:
            extra += ["--prefill-tokens", *map(str, args.prefill_tokens)]
        C.run_workers(os.path.abspath(__file__), list(models), extra_args=extra)
        summarize(list(models))
        return

    (name, path), = models.items()
    result = benchmark(name, path, args.decode_tokens, args.prefill_tokens)
    with open(os.path.join(C.OUTPUT_ROOT, f"prefill_decode_{name}.json"), "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
