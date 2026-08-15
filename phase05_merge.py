from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = Path(r"models\hf_cache\Qwen_Qwen3-4B-Instruct-2507")
ADAPTER = Path(r"experiments\phase04\exploratory_currentdata\dpo_qwen3_4b_beta_0_3")
OUT = Path(r"models\merged\Qwen3-4B-DPO-b03")

print("BASE =", BASE.resolve())
print("ADAPTER =", ADAPTER.resolve())
print("OUT =", OUT.resolve())

print("Loading base...")
base = AutoModelForCausalLM.from_pretrained(
    BASE,
    dtype=torch.float16,
    device_map="cpu",
    trust_remote_code=True,
    low_cpu_mem_usage=True,
)

print("Loading LoRA...")
model = PeftModel.from_pretrained(
    base,
    ADAPTER,
)

print("Merging...")
merged = model.merge_and_unload()

print("Saving...")
OUT.mkdir(parents=True, exist_ok=True)

merged.save_pretrained(
    OUT,
    safe_serialization=True,
    max_shard_size="4GB",
)

tokenizer = AutoTokenizer.from_pretrained(
    BASE,
    trust_remote_code=True,
)
tokenizer.save_pretrained(OUT)

print("PHASE05_LORA_MERGE = PASS")
