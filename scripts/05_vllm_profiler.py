"""Lab - record a vLLM torch-profiler trace and save it as profiling.json.

Run inside the vLLM env (.venv-vllm):

    python scripts/05_vllm_profiler.py --model FP16
    python scripts/05_vllm_profiler.py --model GPTQ --cuda-graph        # eager OFF, CUDA graph ON
    python scripts/05_vllm_profiler.py --model SmoothQuant_O3 --output profiling.json

Only the trace is saved - no analysis.  Open the resulting .json at
https://ui.perfetto.dev/ (see USAGE.md).

Output: $LAB_ROOT/profiling_<model>_<eager|cudagraph>.json  (or --output)
Settings follow the Lab notebook: max_num_seqs=1, max_model_len=2048,
gpu_memory_utilization=0.45, 32 generated tokens, 1 warmup run,
enforce_eager=True by default (so every op shows up in the trace).
"""
import argparse
import glob
import gzip
import os
import shutil
import tempfile
import time

import common as C

MAX_MODEL_LEN = 2048
GPU_MEMORY_UTILIZATION = 0.45
MAX_NEW_TOKENS = 32
WARMUP_RUNS = 1


def wait_for_trace(trace_dir, timeout_s=180):
    """The engine process writes the trace after stop_profile(); wait until it is complete."""
    deadline = time.time() + timeout_s
    last_size, stable = -1, 0
    while time.time() < deadline:
        files = glob.glob(os.path.join(trace_dir, "**", "*.json*"), recursive=True)
        files = [f for f in files if "trace" in os.path.basename(f)]
        if files:
            newest = max(files, key=os.path.getmtime)
            size = os.path.getsize(newest)
            stable = stable + 1 if size == last_size and size > 0 else 0
            last_size = size
            if stable >= 3:                      # size unchanged for ~3 s
                return newest
        time.sleep(1)
    raise TimeoutError(f"No profiler trace appeared in {trace_dir}")


def save_as_json(trace_path, output_path):
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    if trace_path.endswith(".gz"):
        with gzip.open(trace_path, "rb") as src, open(output_path, "wb") as dst:
            shutil.copyfileobj(src, dst)
    else:
        shutil.copyfile(trace_path, output_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(C.MODEL_PATHS), default="FP16")
    parser.add_argument("--cuda-graph", action="store_true",
                        help="enforce_eager=False (CUDA graph on). Default: eager")
    parser.add_argument("--output", default=None, help="output .json path")
    args = parser.parse_args()

    from vllm import LLM, SamplingParams

    name, path = args.model, C.MODEL_PATHS[args.model]
    C.check_model_path(name, path)
    mode = "cudagraph" if args.cuda_graph else "eager"
    output = args.output or os.path.join(C.OUTPUT_ROOT, f"profiling_{name}_{mode}.json")

    trace_dir = tempfile.mkdtemp(prefix="vllm_trace_", dir=C.OUTPUT_ROOT)
    print(f"Model: {name} ({path}) | mode: {mode}")

    llm = LLM(
        model=path,
        dtype="half",
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        max_num_seqs=1,
        enforce_eager=not args.cuda_graph,
        attention_config={"backend": C.ATTENTION_BACKEND},
        profiler_config={"profiler": "torch", "torch_profiler_dir": trace_dir},
    )
    sampling_params = SamplingParams(temperature=0.0, max_tokens=MAX_NEW_TOKENS)

    print("Warming up...")
    for _ in range(WARMUP_RUNS):
        llm.generate([C.SHORT_PROMPT], sampling_params, use_tqdm=False)

    print("Profiling one generation...")
    llm.start_profile()
    outputs = llm.generate([C.SHORT_PROMPT], sampling_params, use_tqdm=False)
    llm.stop_profile()
    print("Generated:", outputs[0].outputs[0].text[:100].replace("\n", " "), "...")

    trace_path = wait_for_trace(trace_dir)
    save_as_json(trace_path, output)
    shutil.rmtree(trace_dir, ignore_errors=True)

    print(f"Saved: {output}  ({os.path.getsize(output) / 1024**2:.1f} MB)")
    print("Open it at https://ui.perfetto.dev/")


if __name__ == "__main__":
    main()
