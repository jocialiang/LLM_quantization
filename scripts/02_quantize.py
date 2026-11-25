"""Lab - quantize Qwen2.5-3B with GPTQ W4A16 and SmoothQuant+GPTQ W8A8 (O3).

Run inside the quantization env (.venv-quant):

    python scripts/02_quantize.py                      # both methods
    python scripts/02_quantize.py --method gptq
    python scripts/02_quantize.py --method smoothquant_o3

Methods
-------
gptq             : GPTQ W4A16, ignore = ["lm_head"]
smoothquant_o3   : SmoothQuant (strength 0.8) + GPTQ W8A8
                   weight  : INT8 per-tensor, symmetric
                   act     : INT8 per-tensor, symmetric, STATIC (O3)
                   ignore  : ["lm_head", "re:.*down_proj"]   (down_proj stays FP16)

Because down_proj is not quantized in O3, the SmoothQuant mapping
down_proj <- up_proj is removed (smoothing a layer that stays FP16 only adds
risk to up_proj without any benefit).
"""
import argparse
import gc
import os
import time

import torch
import transformers

import common as C


def build_calibration_dataset(blocks):
    from datasets import Dataset

    return Dataset.from_dict({
        "input_ids": blocks,
        "attention_mask": [[1] * len(b) for b in blocks],
    })


def load_model_for_ptq(model_id):
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.float16,
        device_map={"": "cuda:0"},
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    model.eval()
    print("Model:", model_id)
    print("CUDA allocated (GiB):", round(torch.cuda.memory_allocated() / 1024**3, 3))
    return model


def run_oneshot_and_save(recipe, calibration_ds, tokenizer, output_dir, label):
    from llmcompressor import oneshot

    model = load_model_for_ptq(C.MODEL_ID)

    start = time.perf_counter()
    oneshot(
        model=model,
        dataset=calibration_ds,
        recipe=recipe,
        max_seq_length=C.CALIBRATION_SEQ_LEN,
        num_calibration_samples=C.CALIBRATION_SAMPLES,
    )
    elapsed = time.perf_counter() - start

    model.save_pretrained(output_dir, save_compressed=True)
    tokenizer.save_pretrained(output_dir)

    print(f"{label} quantization seconds: {elapsed:.2f}")
    print("Saved:", output_dir)

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return output_dir


# ----------------------------------------------------------------------------
# GPTQ W4A16
# ----------------------------------------------------------------------------
def run_gptq(calibration_ds, tokenizer):
    from llmcompressor.modifiers.gptq import GPTQModifier

    recipe = GPTQModifier(
        targets="Linear",
        scheme="W4A16",
        ignore=C.GPTQ_IGNORE,
    )
    return run_oneshot_and_save(recipe, calibration_ds, tokenizer, C.GPTQ_DIR, "GPTQ W4A16")


# ----------------------------------------------------------------------------
# SmoothQuant + GPTQ  W8A8  (O3: per-tensor static activation)
# ----------------------------------------------------------------------------
def build_smoothquant_o3_recipe():
    from compressed_tensors.quantization import QuantizationScheme
    from compressed_tensors.quantization.quant_args import (
        QuantizationArgs,
        QuantizationStrategy,
        QuantizationType,
    )
    from llmcompressor.modifiers.gptq import GPTQModifier
    from llmcompressor.modifiers.transform.smoothquant import SmoothQuantModifier

    weights_args = QuantizationArgs(
        num_bits=8,
        type=QuantizationType.INT,
        strategy=QuantizationStrategy.TENSOR,
        symmetric=True,
        dynamic=False,
    )
    activations_args = QuantizationArgs(
        num_bits=8,
        type=QuantizationType.INT,
        strategy=QuantizationStrategy.TENSOR,   # O3: per-tensor
        symmetric=True,
        dynamic=False,                          # O3: static
    )
    scheme = QuantizationScheme(
        targets=["Linear"],
        weights=weights_args,
        input_activations=activations_args,
    )

    # down_proj is NOT quantized -> no down_proj <- up_proj smoothing mapping.
    smooth_mappings = [
        [["re:.*q_proj", "re:.*k_proj", "re:.*v_proj"], "re:.*input_layernorm"],
        [["re:.*gate_proj", "re:.*up_proj"], "re:.*post_attention_layernorm"],
    ]

    return [
        SmoothQuantModifier(smoothing_strength=0.8, mappings=smooth_mappings),
        GPTQModifier(
            config_groups={"group_0": scheme},
            ignore=C.SQ_O3_IGNORE,              # lm_head + all down_proj stay FP16
        ),
    ]


def run_smoothquant_o3(calibration_ds, tokenizer):
    recipe = build_smoothquant_o3_recipe()
    return run_oneshot_and_save(
        recipe, calibration_ds, tokenizer, C.SQ_O3_DIR, "SmoothQuant W8A8 O3"
    )


# ----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        choices=["gptq", "smoothquant_o3", "all"],
        default="all",
    )
    args = parser.parse_args()

    C.set_seed()

    tokenizer = transformers.AutoTokenizer.from_pretrained(C.MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    blocks = C.build_calibration_blocks(tokenizer)
    print("Calibration blocks:", len(blocks), "x", C.CALIBRATION_SEQ_LEN)
    calibration_ds = build_calibration_dataset(blocks)

    if args.method in ("gptq", "all"):
        run_gptq(calibration_ds, tokenizer)
        gc.collect()
        torch.cuda.empty_cache()

    if args.method in ("smoothquant_o3", "all"):
        run_smoothquant_o3(calibration_ds, tokenizer)


if __name__ == "__main__":
    main()
