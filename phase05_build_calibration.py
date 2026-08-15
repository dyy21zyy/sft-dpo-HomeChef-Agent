import json
from pathlib import Path

src = Path(r"data\processed\sft\v0.3\train.jsonl")
dst = Path(r"data\calibration\phase05_qwen3_4b.txt")

def collect_strings(obj):
    if isinstance(obj, str):
        text = obj.strip()
        if text:
            yield text
    elif isinstance(obj, list):
        for item in obj:
            yield from collect_strings(item)
    elif isinstance(obj, dict):
        for value in obj.values():
            yield from collect_strings(value)

samples = []

for line in src.read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue

    row = json.loads(line)
    text = "\n".join(collect_strings(row))

    if text:
        samples.append(text)

dst.parent.mkdir(parents=True, exist_ok=True)

dst.write_text(
    "\n\n".join(samples),
    encoding="utf-8"
)

print("CALIBRATION_SAMPLES =", len(samples))
print("CALIBRATION_BYTES   =", dst.stat().st_size)
print("CALIBRATION_PATH    =", dst.resolve())
