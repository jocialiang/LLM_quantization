"""Shared settings for Lab (Qwen2.5-3B quantization on a single T4).

Every script imports from here so all experiments use the same model,
calibration data, paths and prompts.  This file must NOT import llmcompressor
or vllm at module level (the two live in different virtualenvs).
"""
import os
import random
import subprocess
import threading
import time

# ----------------------------------------------------------------------------
# Experiment settings (same as the Lab notebook)
# ----------------------------------------------------------------------------
MODEL_ID = "Qwen/Qwen2.5-3B"
CALIBRATION_SAMPLES = 128
CALIBRATION_SEQ_LEN = 512
SEED = 42

# Override with:  export LAB_ROOT=/some/path
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # parent of scripts/
OUTPUT_ROOT = os.environ.get("LAB_ROOT", os.path.join(_REPO_ROOT, "lab_results"))
os.makedirs(OUTPUT_ROOT, exist_ok=True)

# Quantized checkpoint folders produced by 02_quantize.py
GPTQ_DIR = os.path.join(OUTPUT_ROOT, "Qwen2.5-3B-GPTQ-W4A16")
SQ_O3_DIR = os.path.join(OUTPUT_ROOT, "Qwen2.5-3B-SmoothQuant-W8A8-O3-ignore-down_proj")

# name -> model path (FP16 baseline is the HF model id)
MODEL_PATHS = {
    "FP16": MODEL_ID,
    "GPTQ": GPTQ_DIR,
    "SmoothQuant_O3": SQ_O3_DIR,
}

# Modules that are NOT quantized
GPTQ_IGNORE = ["lm_head"]
SQ_O3_IGNORE = ["lm_head", "re:.*down_proj"]

# ----------------------------------------------------------------------------
# vLLM engine settings shared by 03 / 04 / 05
# ----------------------------------------------------------------------------
ATTENTION_BACKEND = "TRITON_ATTN"   # T4 (sm75) has no FlashAttention-2


def set_seed(seed: int = SEED):
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def check_model_path(name: str, path: str):
    """FP16 uses a HF model id; quantized models must exist locally."""
    if name != "FP16" and not os.path.isdir(path):
        raise FileNotFoundError(
            f"Quantized checkpoint not found: {path}\n"
            f"Run 02_quantize.py first (or set LAB_ROOT)."
        )


def select_models(names):
    """Return {name: path} for the requested model names (None = all)."""
    if not names:
        return dict(MODEL_PATHS)
    unknown = [n for n in names if n not in MODEL_PATHS]
    if unknown:
        raise SystemExit(f"Unknown model(s): {unknown}. Choose from {list(MODEL_PATHS)}")
    return {n: MODEL_PATHS[n] for n in names}


# ----------------------------------------------------------------------------
# Prompts used for prefill / decode / profiling
# ----------------------------------------------------------------------------
# Long prompt: first 400 whitespace-separated words of the notebook's "text-expansion"
# prompt (identical to the Lab notebook).  Used for the prefill benchmark.
LONG_PROMPT = 'You are a Distinguished AI Research Scientist and Principal Technical Writer at a top-tier research institution. Your task is to act as an advanced text expansion engine to complete a comprehensive, graduate-level textbook chapter on the architectural mechanics, training paradigms, and socio-technical implications of modern Artificial Intelligence. ### CRITICAL INSTRUCTIONS FOR GENERATION: 1. **Academic Rigor:** Write with extreme technical density. Use precise mathematical, algorithmic, and hardware-level nomenclature. Avoid high-level generalities, fluff, or hand-waving explanations. 2. **Structural Continuity:** Seamlessly transition from the provided text in Section 1 into Section 2, and complete the rest of the sections maintaining an authoritative, objective, and analytical tone. 3. **Exhaustive Detail:** Every section must be expanded to its maximum technical depth, utilizing subsections, numbered structural steps, and bulleted technical breakdowns. 4. **Target Output Length:** The completed document should be an exhaustive, long-form reference paper (aim for 3,000+ words of generated output). --- ### [START OF TEXTBOOK CHAPTER] #### Section 1: Introduction to Artificial Intelligence Artificial intelligence (AI) has transitioned from a theoretical concept in science fiction to an omnipresent force driving modern technology. At its core, AI refers to the simulation of human intelligence processes by machines, especially computer systems. These processes include learning, reasoning, problem-solving, perception, and language understanding. The history of AI dates back to the mid-20th century, when pioneers like Alan Turing questioned whether machines could possess the capacity to think. Over the decades, the field has experienced cycles of intense optimism followed by periods of disappointment and reduced funding, commonly referred to as "AI winters." Today, we are living in an era of unprecedented AI expansion, fueled by three main catalysts: the exponential increase in computational power, the availability of massive datasets, and breakthroughs in algorithmic design, particularly deep learning. Deep learning, a subset of machine learning, utilizes artificial neural networks with multiple layers to model complex patterns in data. These networks mimic the structural connectivity of the human brain, allowing software to recognize speech, identify images, and make predictions with astonishing accuracy. As a result, AI is no longer confined to academic laboratories; it is actively reshaping industries, transforming global economies, and redefining how humans interact with technology on a daily basis. #### Section 2: Technological Infrastructure and Deep Learning The infrastructure supporting modern artificial intelligence is highly sophisticated, relying heavily on specialized hardware like Graphics Processing Units (GPUs) and Tensor Processing Units (TPUs). Unlike traditional Central Processing Units (CPUs),'


def build_long_prompt() -> str:
    return LONG_PROMPT


def build_prompt_token_ids(tokenizer, num_tokens: int):
    """Return exactly `num_tokens` token ids built from LONG_PROMPT (repeated if needed)."""
    base = tokenizer(LONG_PROMPT, add_special_tokens=False)["input_ids"]
    ids = []
    while len(ids) < num_tokens:
        ids.extend(base)
    return ids[:num_tokens]


# Short prompt for decode benchmark and profiler
SHORT_PROMPT = (
    "Explain the difference between weight-only quantization "
    "and weight-activation quantization."
)


# ----------------------------------------------------------------------------
# Calibration data (WikiText-2 train, same concatenation as the PPL eval)
# ----------------------------------------------------------------------------
def build_calibration_blocks(tokenizer):
    from datasets import load_dataset

    raw = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    calib_text = "\n\n".join(raw["text"])
    all_ids = tokenizer(calib_text, add_special_tokens=False)["input_ids"]

    L = CALIBRATION_SEQ_LEN
    blocks = [all_ids[i:i + L] for i in range(0, len(all_ids) - L + 1, L)]
    random.Random(SEED).shuffle(blocks)
    return blocks[:CALIBRATION_SAMPLES]


# ----------------------------------------------------------------------------
# GPU memory monitor (polls nvidia-smi)
# ----------------------------------------------------------------------------
class GPUMemoryMonitor:
    def __init__(self, interval=0.1, gpu_id=0):
        self.interval = interval
        self.gpu_id = gpu_id
        self.peak_mb = 0
        self.baseline_mb = 0
        self.running = False
        self.thread = None

    def _read_memory_mb(self):
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "-i", str(self.gpu_id),
                 "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                encoding="utf-8", stderr=subprocess.DEVNULL,
            )
            return int(out.strip().splitlines()[0])
        except Exception:
            return None

    def _monitor(self):
        while self.running:
            used = self._read_memory_mb()
            if used is not None:
                self.peak_mb = max(self.peak_mb, used)
            time.sleep(self.interval)

    def start(self):
        initial = self._read_memory_mb()
        self.baseline_mb = initial if initial is not None else 0
        self.peak_mb = self.baseline_mb
        self.running = True
        self.thread = threading.Thread(target=self._monitor, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread is not None:
            self.thread.join()
        return {
            "gpu_memory_baseline_MB": self.baseline_mb,
            "gpu_memory_baseline_GB": self.baseline_mb / 1024,
            "gpu_memory_peak_MB": self.peak_mb,
            "gpu_memory_peak_GB": self.peak_mb / 1024,
            "gpu_memory_increase_MB": max(0, self.peak_mb - self.baseline_mb),
        }


# ----------------------------------------------------------------------------
# Run each model in its own process (vLLM does not reliably release GPU memory
# after `del llm`, so one process per model is the safe way on a 16 GB T4).
# ----------------------------------------------------------------------------
def run_workers(script_path, names, extra_args=()):
    import sys

    for name in names:
        print("\n" + "#" * 70)
        print(f"# Launching worker for {name}")
        print("#" * 70, flush=True)
        subprocess.run(
            [sys.executable, script_path, "--models", name, "--worker", *extra_args],
            check=True,
        )
