"""Lab - WikiText-2 perplexity of FP16 / GPTQ / SmoothQuant-O3 with vLLM.

Run inside the vLLM env (.venv-vllm):

    python scripts/03_perplexity.py                         # all models, 20 chunks (notebook default)
    python scripts/03_perplexity.py --max-chunks 0          # full WikiText-2 test set
    python scripts/03_perplexity.py --models FP16 GPTQ      # subset

Method: one prefill pass per 512-token chunk with prompt_logprobs=0,
PPL = exp(mean NLL over tokens 1..511 of every chunk).

Outputs (in $LAB_ROOT): ppl_<model>.json per model, ppl_results.json / .csv summary.
"""
import argparse
import gc
import json
import math
import os
import time

import common as C

PPL_SEQ_LEN = 512
PPL_GPU_MEMORY_UTILIZATION = 0.85


def prepare_chunks(max_chunks):
    import transformers
    from datasets import load_dataset

    tokenizer = transformers.AutoTokenizer.from_pretrained(C.MODEL_ID)
    test_raw = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    test_ids = tokenizer("\n\n".join(test_raw["text"]), add_special_tokens=False)["input_ids"]

    num_chunks = len(test_ids) // PPL_SEQ_LEN          # drop the last partial chunk
    if max_chunks:
        num_chunks = min(num_chunks, max_chunks)
    chunks = [test_ids[i * PPL_SEQ_LEN:(i + 1) * PPL_SEQ_LEN] for i in range(num_chunks)]
    print("Test tokens:", len(test_ids))
    print("Chunks:", len(chunks), "x", PPL_SEQ_LEN, "tokens")
    return chunks


def evaluate_perplexity(model_name, model_path, chunks):
    import torch
    from vllm import LLM, SamplingParams

    print("\n" + "=" * 70)
    print("PPL MODEL:", model_name)
    print("PATH     :", model_path)
    print("=" * 70)
    C.check_model_path(model_name, model_path)

    llm = LLM(
        model=model_path,
        dtype="half",
        max_model_len=PPL_SEQ_LEN + 8,
        gpu_memory_utilization=PPL_GPU_MEMORY_UTILIZATION,
        max_num_seqs=1,
        enforce_eager=False,
        enable_prefix_caching=False,
        attention_config={"backend": C.ATTENTION_BACKEND},
    )

    # prompt_logprobs=0 -> log-prob of the actual prompt tokens only
    # max_tokens=1      -> a single prefill, no real generation
    sampling_params = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=0)
    prompts = [{"prompt_token_ids": ids} for ids in chunks]

    start = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params, use_tqdm=True)
    elapsed = time.perf_counter() - start

    total_nll, total_tokens = 0.0, 0
    for ids, out in zip(chunks, outputs):
        plp = out.prompt_logprobs
        assert len(plp) == len(ids), "prompt_logprobs length mismatch"
        # token 0 has no context (logprob None) -> start from token 1
        for tok, lp in zip(ids[1:], plp[1:]):
            total_nll -= lp[tok].logprob
            total_tokens += 1

    avg_nll = total_nll / total_tokens
    ppl = math.exp(avg_nll)
    print(f"Evaluated tokens: {total_tokens}")
    print(f"Avg NLL: {avg_nll:.4f}")
    print(f"Perplexity: {ppl:.4f}")
    print(f"Eval time: {elapsed:.1f}s")

    del llm
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "model": model_name,
        "model_path": model_path,
        "num_chunks": len(chunks),
        "seq_len": PPL_SEQ_LEN,
        "eval_tokens": total_tokens,
        "avg_nll": avg_nll,
        "perplexity": ppl,
        "eval_time_s": elapsed,
    }


def summarize(names):
    import pandas as pd

    results = []
    for n in names:
        path = os.path.join(C.OUTPUT_ROOT, f"ppl_{n}.json")
        if os.path.exists(path):
            with open(path) as f:
                results.append(json.load(f))
    if not results:
        return

    baseline = next((r["perplexity"] for r in results if r["model"] == "FP16"), None)
    df = pd.DataFrame([
        {
            "Model": r["model"],
            "Perplexity": round(r["perplexity"], 4),
            "Avg NLL": round(r["avg_nll"], 4),
            "Δ PPL vs FP16": round(r["perplexity"] - baseline, 4) if baseline else None,
            "Δ PPL vs FP16 (%)": round((r["perplexity"] / baseline - 1) * 100, 2) if baseline else None,
            "Eval tokens": r["eval_tokens"],
        }
        for r in results
    ])
    print("\n" + df.to_string(index=False))

    with open(os.path.join(C.OUTPUT_ROOT, "ppl_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    df.to_csv(os.path.join(C.OUTPUT_ROOT, "ppl_results.csv"), index=False)
    print("Saved:", os.path.join(C.OUTPUT_ROOT, "ppl_results.csv"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="*", help=f"subset of {list(C.MODEL_PATHS)}")
    parser.add_argument("--max-chunks", type=int, default=20,
                        help="number of 512-token chunks (0 = full test set). Default 20 = notebook setting")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    models = C.select_models(args.models)

    if not args.worker:
        # one process per model, then aggregate
        C.run_workers(os.path.abspath(__file__), list(models),
                      extra_args=["--max-chunks", str(args.max_chunks)])
        summarize(list(models))
        return

    chunks = prepare_chunks(args.max_chunks)
    (name, path), = models.items()
    result = evaluate_perplexity(name, path, chunks)
    with open(os.path.join(C.OUTPUT_ROOT, f"ppl_{name}.json"), "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
