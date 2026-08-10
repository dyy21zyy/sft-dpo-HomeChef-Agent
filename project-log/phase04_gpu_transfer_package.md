# Phase 04 SFT GPU Transfer Package

**Checkpoint C — Runtime Blocked (No Local GPU)**
**Prepared: 2026-08-10**
**Git Commit: `54c1374`**

---

## 1. Local GPU Block Reason

| Item | Detail |
|------|--------|
| Machine GPU | AMD Radeon 780M (integrated, no CUDA) |
| PyTorch build | `2.13.0+cpu` (CPU-only) |
| `torch.cuda.is_available()` | `False` |
| `nvidia-smi` | Not found |
| LLaMA-Factory | 0.9.5 (requires CUDA for GPU training) |
| Python | 3.12.13 (OK) |
| Formal SFT blocked | Cannot run 540-row, 3-epoch SFT on CPU |

---

## 2. Package Contents

Transfer the following **10 files** from `sft-dpo-HomeChef-Agent/` to the GPU host,
preserving the exact directory structure.

### Training Configs

| # | File | SHA256 |
|---|------|--------|
| 1 | `configs/training/phase04_sft_qwen3_0_6b.yaml` | `dfc7d54a028915e2948b3832dcdfc8d41f5ccab9751fdf70014249526ade1dcb` |
| 2 | `configs/training/phase04_sft_qwen3_1_7b.yaml` | `684efac91a733d5038a6165159479a5852f8a6382737b20b43706528c04cec8d` |
| 3 | `configs/training/phase04_dataset_info.json` | `6a2260ee7e56fc252172bc4a857eef19d195fe4f7d177964d52bdf3bbbd9c7fd` |

### Dataset Registry

| # | File | SHA256 |
|---|------|--------|
| 4 | `data/dataset_info.json` | `ce531275153c0439dc679f5c35a089bb5eaa3189984e3e18faada571349decf7` |

### SFT Training/Validation Data

| # | File | Rows | SHA256 |
|---|------|------|--------|
| 5 | `data/processed/phase03_sft_v0.1_train.jsonl` | 540 | `94d3748dcf1e68b51b09486f02f964273ed8e4ae9989797fd672d5484bf2639c` |
| 6 | `data/processed/phase03_sft_v0.1_val.jsonl` | 60 | `896da9374ea945c8f4675504599443d10501571e0cf83733340e67ad99249633` |
| 7 | `data/processed/phase03_dataset_manifest.v0.1.json` | — | `63b04de308bb4465d105fa3e3cc7c2dd9f77401a3ce056d9ebd729db16489318` |

### Dependencies & Approvals

| # | File | SHA256 |
|---|------|--------|
| 8 | `requirements-train.txt` | `773c8cb5752484d6ec2d98c85f94bad6f9652ae250f9cba7759d4d7996c0e9ba` |
| 9 | `project-log/phase04_dryrun_approval.json` | `4bbcab1c8c25fa0340f51a9a8f0bd3aa52d2c048a7741163ac4363566d87cde0` |
| 10 | `project-log/phase04_dryrun_manifest.json` | `8c862fd57eaf8c564fdec0ef70c749f0789da9b7025531d9d37cdc2ede48a33f` |

---

## 3. Explicitly Excluded Files

These MUST NOT be transferred or used as training data:

| Exclusion | Reason |
|-----------|--------|
| `data/eval/frozen_test.*` | Frozen Test — not for training |
| `data/dev/diagnostic_dev.*` | Diagnostic Dev — not for training |
| `data/phase02_*` | Phase 02 outputs — not for training |
| `data/phase04_dryrun_sft_1row.jsonl` | Dry-run temp dataset |
| `data/phase04_dryrun_dpo_1pair.jsonl` | Dry-run temp dataset |
| `experiments/phase04/dryrun/` | Dry-run outputs |
| `configs/training/phase04_dpo_dryrun_beta_0_1.yaml` | DPO config — not for SFT run |
| `data/processed/phase03_dpo_targeted_*` | DPO data — not for SFT run |
| `data/processed/phase03_dpo_*` | DPO data — not for SFT run |

---

## 4. Target GPU Host Setup Checklist

### Hardware
- [ ] NVIDIA GPU (RTX 4090 24GB or equivalent)
- [ ] `nvidia-smi` available and shows GPU

### Software
- [ ] Python 3.12.x installed
- [ ] Create and activate a Python 3.12 virtual environment
- [ ] Install dependencies: `pip install -r requirements-train.txt`
- [ ] Verify `llamafactory-cli` is on PATH

### Pre-flight Verification (run on GPU host)
```bash
git status --short
python --version
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
llamafactory-cli version
python -c "import llamafactory, datasets, peft, transformers, torch, accelerate; print('ok')"
```

Expected output:
- `git status --short` → empty (clean)
- `python --version` → `Python 3.12.x`
- CUDA available → `True`, shows GPU name
- `llamafactory-cli version` → `0.9.5` (or compatible)
- Import check → `ok`

### Verify dataset integrity (run on GPU host)
```bash
python -c "
import hashlib, json
expected = {
    'data/processed/phase03_sft_v0.1_train.jsonl': '94d3748dcf1e68b51b09486f02f964273ed8e4ae9989797fd672d5484bf2639c',
    'data/processed/phase03_sft_v0.1_val.jsonl': '896da9374ea945c8f4675504599443d10501571e0cf83733340e67ad99249633',
}
for f, expected_hash in expected.items():
    with open(f, 'rb') as fh:
        actual = hashlib.sha256(fh.read()).hexdigest()
    status = 'OK' if actual == expected_hash else 'MISMATCH'
    print(f'{status}: {f}')
"
```

---

## 5. Exact Training Commands

Run these commands in order from the `sft-dpo-HomeChef-Agent/` directory:

### 5a. Qwen3-0.6B-Base SFT
```bash
llamafactory-cli train configs/training/phase04_sft_qwen3_0_6b.yaml
```

### 5b. Qwen3-1.7B-Base SFT
```bash
llamafactory-cli train configs/training/phase04_sft_qwen3_1_7b.yaml
```

### After each run, collect:
1. Output directory path
2. Adapter checkpoint path
3. Trainer logs
4. `trainer_state.json` (if available)
5. Dependency snapshot (`pip freeze`)
6. `nvidia-smi` output
7. Git commit hash
8. Dataset SHA256 hashes
9. Config SHA256 hashes
10. `TrainingArtifactManifest`
11. Final train/eval loss (do not claim task effectiveness)

---

## 6. Training Config Summary

### Qwen3-0.6B-Base (`phase04_sft_qwen3_0_6b.yaml`)
| Parameter | Value |
|-----------|-------|
| Model | `Qwen/Qwen3-0.6B-Base` |
| Stage | `sft` |
| Finetuning | LoRA (rank=16, alpha=32, dropout=0.05) |
| Template | `qwen3` |
| Cutoff | 2048 |
| LR | 1e-4 (cosine, warmup 0.1) |
| Epochs | 3 |
| Batch size | 2 per device, grad accum 8 |
| Precision | bf16 |
| Eval/Save | per epoch, best by eval_loss |

### Qwen3-1.7B-Base (`phase04_sft_qwen3_1_7b.yaml`)
| Parameter | Value |
|-----------|-------|
| Model | `Qwen/Qwen3-1.7B-Base` |
| (all other params identical to 0.6B config) | |

### Dataset
| Split | Rows | Path |
|-------|------|------|
| Train | 540 | `data/processed/phase03_sft_v0.1_train.jsonl` |
| Val | 60 | `data/processed/phase03_sft_v0.1_val.jsonl` |
| Duplicates | 0 | All duplicate counts = 0 |
| Train/Val overlap | 0 | By SHA256 hash |

---

## 7. Constraints

- [ ] Run SFT only — no DPO, no GGUF, no quantization, no deployment
- [ ] No Frozen Test, Diagnostic Dev, or Phase 02 outputs as training data
- [ ] Do not claim training effectiveness from loss values alone
- [ ] Evaluation requires separate approval (Phase 05)
- [ ] After both SFT runs complete, STOP and report full results
