# sft-dpo-HomeChef-Agent

HomeChef Booking Agent local small-model post-training project.

## Phase 00 Scope

Phase 00 establishes the machine-readable booking contract, strict Python schemas, contract validator, and mock fixtures. It does not generate training data, download models, perform SFT/DPO, or integrate with HomeChef runtime.

## Source of Truth

- `BOOKING_MACHINE_CONTRACT_v1.md` — frozen machine contract
- `finetune-spec.md` — approved design spec
- `contracts/contract_manifest.yaml` — hash-manifest of source documents and schemas

## Validation Commands

```powershell
uv run ruff check .
uv run pytest -q
uv run homechef-contract-validate --root . --fixtures tests/fixtures/contracts
```

## Out of Scope

No training data generation, no SFT, no DPO, no model download, no quantization, no llama.cpp, no HomeChef database writes, no HomeChef integration code.
