"""Phase 03 SFT rendering from validated raw samples."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from homechef_booking.data.dataset_schema import SftMessage, SftSample
from homechef_booking.data.raw_sample import RawBookingSample
from homechef_booking.prompts import PromptBuilder


def canonical_decision_json(decision) -> str:
    return json.dumps(decision.model_dump(mode="json", exclude_none=False), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def render_sft_sample(raw: RawBookingSample) -> SftSample:
    messages = PromptBuilder().build_messages(raw.input)
    completion = canonical_decision_json(raw.expected)
    sft_messages = [SftMessage(role=str(msg["role"]), content=str(msg["content"]) if msg.get("content") is not None else "") for msg in messages]
    sft_messages.append(SftMessage(role="assistant", content=completion))
    prompt_text = "\n".join(json.dumps(msg.model_dump(), ensure_ascii=False, sort_keys=True) for msg in sft_messages[:-1])
    prompt_sha = hashlib.sha256(prompt_text.encode()).hexdigest()
    completion_sha = hashlib.sha256(completion.encode()).hexdigest()
    return SftSample(
        id=f"sft-{raw.id}",
        raw_id=raw.id,
        dataset_version=raw.dataset_version,
        messages=sft_messages,
        prompt_sha256=prompt_sha,
        completion_sha256=completion_sha,
        tags=list(raw.tags),
    )


def build_sft_dataset(raw_path: Path, train_path: Path, val_path: Path, val_ratio: float = 0.10, seed: int = 3001) -> list[SftSample]:
    from homechef_booking.data.raw_validator import load_valid_raw_samples
    raw_samples = load_valid_raw_samples(raw_path)
    sft_samples = [render_sft_sample(raw) for raw in raw_samples]
    import random
    rng = random.Random(seed)
    indices = list(range(len(sft_samples)))
    rng.shuffle(indices)
    split = max(1, int(len(sft_samples) * val_ratio))
    val_indices = set(indices[:split])
    train = [sft_samples[i] for i in indices if i not in val_indices]
    val = [sft_samples[i] for i in indices if i in val_indices]
    train_path.parent.mkdir(parents=True, exist_ok=True)
    val_path.parent.mkdir(parents=True, exist_ok=True)
    train_path.write_text("\n".join(json.dumps(s.model_dump(mode="json", exclude_none=False), ensure_ascii=False, sort_keys=True) for s in train), encoding="utf-8")
    val_path.write_text("\n".join(json.dumps(s.model_dump(mode="json", exclude_none=False), ensure_ascii=False, sort_keys=True) for s in val), encoding="utf-8")
    return sft_samples
