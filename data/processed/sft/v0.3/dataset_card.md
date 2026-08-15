# SFT v0.3 Dataset Card

**Version**: phase03_v0.3_sft
**Raw source**: data/raw/phase03_raw_strong_model_600_v0.3.jsonl
**Derivation**: deterministic render of validated Raw (no model call)
**Train**: 540 / **Val**: 60 (9:1 stratified, deterministic seed 3001)

## Split
- train 540 / val 60
- source_raw_id zero leakage
- difficulty stratified: train 162 easy / 216 medium / 162 hard; val 18/24/18

## Content
- messages from the real PromptBuilder
- assistant target directly from Raw.expected (canonical decision JSON)
- no model was called to generate SFT
