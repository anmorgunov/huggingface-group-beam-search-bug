"""
Usage:

    uv run bfloat-gen.py
"""

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- configuration ---
MODEL_ID = "sshleifer/tiny-gpt2"
DTYPE = torch.bfloat16
DEVICE = "cpu"

def run_repro():
    print("--- bug reproduction script ---")
    print(f"torch version: {torch.__version__}")
    print(f"transformers version: {transformers.__version__}")
    print(f"loading model '{MODEL_ID}' with dtype={DTYPE} on device='{DEVICE}'...")

    # 1. load model and tokenizer, forcing bfloat16 and cpu execution
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            dtype=DTYPE,
            device_map=DEVICE,
        )
        model.eval()
    except Exception:
        return

    print("model loaded successfully.")

    # 2. prepare inputs
    prompt = "this code will crash because"
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    print(f"input prompt: '{prompt}'")

    # 3. set generation parameters to trigger the buggy code path
    #    - num_beam_groups > 1   --> invokes _group_beam_search
    #    - output_scores = True  --> enters the `if output_scores:` block with the bug
    gen_kwargs = {
        "max_new_tokens": 10,
        "num_beams": 4,
        "num_beam_groups": 2,
        "output_scores": True,
        "diversity_penalty": 0.1,
        "return_dict_in_generate": True,
    }

    # 4. run generation and expect a crash
    with torch.no_grad():
        _ = model.generate(**inputs, **gen_kwargs)

if __name__ == "__main__":
    run_repro()
