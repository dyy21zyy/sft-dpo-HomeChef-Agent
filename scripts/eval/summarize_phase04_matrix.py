"""Phase 04 16-cell comparison + deltas + case-level diff + exploratory selection.

Reads:
- 12 Phase04 scorecards (reports/generated/phase04/exploratory_currentdata/<run_id>/)
- 4 Phase02 Base cells (reports/generated/phase02/v3_final/...)

Produces (into the Phase04 exploratory output root):
- phase04_16cell_comparison.json / .csv / .md
- phase04_deltas.json / .md
- SFT->DPO case-level diffs
- exploratory_best_run selection (NOT a formal champion; formal_release_eligible=false)

Metrics reuse the Phase02 canonical field names (protocol_pass_rate,
effective_pass_rate, mean_task_correctness, critical_error_rate, plus
performance p95/ttft/tokens_per_second where available).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

EXPLORATORY_CLASS = "exploratory_currentdata"

# Fixed 12-cell run_ids (same order as the matrix).
RUN_IDS = [
    "phase04_1_7b_sft_u", "phase04_1_7b_sft_s",
    "phase04_1_7b_dpo_b01_u", "phase04_1_7b_dpo_b01_s",
    "phase04_1_7b_dpo_b03_u", "phase04_1_7b_dpo_b03_s",
    "phase04_4b_sft_u", "phase04_4b_sft_s",
    "phase04_4b_dpo_b01_u", "phase04_4b_dpo_b01_s",
    "phase04_4b_dpo_b03_u", "phase04_4b_dpo_b03_s",
]

# Phase02 Base cells: (label, model_key, variant, aggregate.json path pattern)
PHASE02_BASE = [
    ("1_7b_Base_U", "1_7b", "u", "qwen3_1_7b_unstructured"),
    ("1_7b_Base_S", "1_7b", "s", "qwen3_1_7b_structured"),
    ("4b_Base_U", "4b", "u", "qwen3_4b_unstructured"),
    ("4b_Base_S", "4b", "s", "qwen3_4b_structured"),
]


def _extract_metrics(data: dict) -> dict:
    """Normalize a scorecard/aggregate into canonical Phase02 metric fields.

    Handles both Phase04 scorecard.json and Phase02 aggregate.json shapes.
    """
    perf = data.get("performance") or {}
    return {
        "protocol_pass_rate": data.get("protocol_pass_rate") or data.get("metrics", {}).get("protocol_pass_rate"),
        "effective_pass_rate": data.get("effective_pass_rate") or data.get("metrics", {}).get("effective_pass_rate"),
        "mean_task_correctness": data.get("mean_task_correctness") or data.get("metrics", {}).get("mean_task_correctness"),
        "critical_error_rate": data.get("critical_error_rate")
                               or data.get("metrics", {}).get("critical_error_rate"),
        "p95_latency_s": perf.get("p95_latency_s"),
        "mean_ttft_s": perf.get("mean_ttft_s"),
        "mean_tokens_per_second": perf.get("mean_tokens_per_second"),
        "tool_timing": (data.get("metrics", {}).get("metrics", {}).get("tool_timing_accuracy")
                        if isinstance(data.get("metrics"), dict) else None),
        "state_inheritance": (data.get("metrics", {}).get("metrics", {}).get("state_inheritance_accuracy")
                              if isinstance(data.get("metrics"), dict) else None),
    }


def _load_phase04_scorecard(output_root: Path, run_id: str) -> dict | None:
    p = output_root / run_id / "scorecard.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def _load_phase02_base(phase02_root: Path, sub: str) -> dict | None:
    p = phase02_root / sub / "aggregate.json"
    if not p.exists():
        p = phase02_root / sub / "scorecard.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def build_comparison(output_root: Path, phase02_root: Path) -> list[dict]:
    """Build the 16-cell absolute-metrics table."""
    cells: list[dict] = []

    # 12 Phase04 cells.
    for run_id in RUN_IDS:
        sc = _load_phase04_scorecard(output_root, run_id)
        if sc is None:
            cells.append({"run_id": run_id, "model": None, "stage": None, "beta": None,
                          "variant": None, "available": False})
            continue
        # parse model/stage/beta/variant from run_id.
        parts = run_id.replace("phase04_", "").split("_")
        model = parts[0]          # 1_7b / 4b
        if parts[1] == "sft":
            stage, beta, variant = "sft", None, parts[2]
        else:
            stage, beta, variant = "dpo", parts[2], parts[3]
        m = _extract_metrics(sc)
        cells.append({
            "run_id": run_id, "model": model,
            "model_size": "1.7B" if model == "1_7b" else "4B",
            "stage": stage, "beta": beta, "variant": variant,
            "available": True, **m,
        })

    # 4 Phase02 Base cells.
    for label, model, variant, sub in PHASE02_BASE:
        agg = _load_phase02_base(phase02_root, sub)
        if agg is None:
            cells.append({"run_id": label, "model": model, "stage": "base", "beta": None,
                          "variant": variant, "available": False})
            continue
        m = _extract_metrics(agg)
        cells.append({
            "run_id": label, "model": model,
            "model_size": "1.7B" if model == "1_7b" else "4B",
            "stage": "base", "beta": None, "variant": variant, "available": True, **m,
        })
    return cells


def _delta(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    return round(a - b, 4)


def build_deltas(cells: list[dict]) -> dict:
    """Compute SFT/DPO/Structured deltas per model (critical error: lower is better)."""
    by_id = {c["run_id"]: c for c in cells if c.get("available")}

    def _g(model, stage, beta, variant):
        if stage == "sft":
            rid = f"phase04_{model}_sft_{variant}"
        else:
            rid = f"phase04_{model}_dpo_{beta}_{variant}"
        return by_id.get(rid)

    def _base(model, variant):
        return by_id.get(f"{model}_Base_{'U' if variant == 'u' else 'S'}")

    deltas: dict[str, Any] = {}
    metrics = ["protocol_pass_rate", "mean_task_correctness", "effective_pass_rate", "critical_error_rate"]

    for model in ("1_7b", "4b"):
        deltas[model] = {"sft_gain": {}, "dpo_gain": {}, "structured_gain": {}}
        # SFT gain (SFT - Base)
        for variant in ("u", "s"):
            sft = _g(model, "sft", None, variant)
            base = _base(model, variant)
            if sft and base:
                deltas[model]["sft_gain"][f"SFT-{variant.upper()}-Base-{variant.upper()}"] = {
                    mt: _delta(sft.get(mt), base.get(mt)) for mt in metrics}
        # DPO gain (DPO - SFT)
        for beta in ("b01", "b03"):
            for variant in ("u", "s"):
                dpo = _g(model, "dpo", beta, variant)
                sft = _g(model, "sft", None, variant)
                if dpo and sft:
                    deltas[model]["dpo_gain"][f"DPO-{beta}-{variant.upper()}-SFT-{variant.upper()}"] = {
                        mt: _delta(dpo.get(mt), sft.get(mt)) for mt in metrics}
        # Structured gain (S - U)
        for rid_s, rid_u in ((f"phase04_{model}_sft_s", f"phase04_{model}_sft_u"),
                             (f"phase04_{model}_dpo_b01_s", f"phase04_{model}_dpo_b01_u"),
                             (f"phase04_{model}_dpo_b03_s", f"phase04_{model}_dpo_b03_u")):
            s, u = by_id.get(rid_s), by_id.get(rid_u)
            if s and u:
                key = f"{rid_s.replace(f'phase04_{model}_', '')} - {rid_u.replace(f'phase04_{model}_', '')}"
                deltas[model]["structured_gain"][key] = {
                    mt: _delta(s.get(mt), u.get(mt)) for mt in metrics}
        # Base structured gain
        bs, bu = _base(model, "s"), _base(model, "u")
        if bs and bu:
            deltas[model]["structured_gain"]["Base-S - Base-U"] = {
                mt: _delta(bs.get(mt), bu.get(mt)) for mt in metrics}
    return deltas


def build_case_diff(output_root: Path) -> dict:
    """SFT(parent) vs DPO(child) per-case diff for each (model, variant) pair."""
    out: dict[str, Any] = {}
    pairs = [
        ("1_7b", "u", "sft", "b01"), ("1_7b", "u", "sft", "b03"),
        ("1_7b", "s", "sft", "b01"), ("1_7b", "s", "sft", "b03"),
        ("4b", "u", "sft", "b01"), ("4b", "u", "sft", "b03"),
        ("4b", "s", "sft", "b01"), ("4b", "s", "sft", "b03"),
    ]
    for model, variant, _parent_stage, beta in pairs:
        parent_rid = f"phase04_{model}_sft_{variant}"
        child_rid = f"phase04_{model}_dpo_{beta}_{variant}"
        pcr = output_root / parent_rid / "case_results.json"
        ccr = output_root / child_rid / "case_results.json"
        if not pcr.exists() or not ccr.exists():
            continue
        parent = {r["case_id"]: r for r in json.loads(pcr.read_text(encoding="utf-8"))}
        child = {r["case_id"]: r for r in json.loads(ccr.read_text(encoding="utf-8"))}
        improved, regressed, unchanged = [], [], []
        for cid, pc in parent.items():
            cc = child.get(cid)
            if cc is None:
                continue
            if cc.get("effective_pass") and not pc.get("effective_pass"):
                improved.append(cid)
            elif pc.get("effective_pass") and not cc.get("effective_pass"):
                regressed.append(cid)
            else:
                unchanged.append(cid)
        out[f"{model}_{beta}_{variant}"] = {
            "parent": parent_rid, "child": child_rid,
            "improved_cases": improved,
            "regressed_cases": regressed,
            "unchanged_cases": unchanged,
            "net_effective_pass": len(improved) - len(regressed),
            "critical_error_introduced": [cid for cid in regressed
                                          if parent.get(cid, {}).get("protocol_pass")
                                          and not child.get(cid, {}).get("protocol_pass")],
        }
    return out


def select_exploratory_best(cells: list[dict]) -> dict:
    """Exploratory selection (NOT a formal champion).

    Priority: protocol high > avoid critical error > maximize effective pass >
    task correctness tiebreak > smaller model. Always marks formal_release_eligible=false.
    """
    cands = [c for c in cells if c.get("available") and c.get("stage") != "base"]
    if not cands:
        return {"exploratory_best_run": None, "formal_release_eligible": False,
                "sft_gate_protocol_met": None, "sft_gate_task_met": None}

    def _key(c):
        crit = c.get("critical_error_rate") or 1.0
        return (
            -(c.get("protocol_pass_rate") or 0.0),   # protocol high first
            crit,                                      # critical error low (avoid)
            -(c.get("effective_pass_rate") or 0.0),    # effective pass max
            -(c.get("mean_task_correctness") or 0.0),  # task correctness tiebreak
            0 if c.get("model_size") == "1.7B" else 1,  # smaller model preferred
        )

    best = sorted(cands, key=_key)[0]
    # SFT gate (finetune-spec): protocol>=0.90, task>=0.85. Report whether met
    # by the best candidate.
    return {
        "exploratory_best_run": best.get("run_id"),
        "model_size": best.get("model_size"),
        "stage": best.get("stage"),
        "beta": best.get("beta"),
        "variant": best.get("variant"),
        "protocol_pass_rate": best.get("protocol_pass_rate"),
        "effective_pass_rate": best.get("effective_pass_rate"),
        "mean_task_correctness": best.get("mean_task_correctness"),
        "critical_error_rate": best.get("critical_error_rate"),
        "formal_release_eligible": False,
        "sft_gate_protocol_met": (best.get("protocol_pass_rate") or 0.0) >= 0.90,
        "sft_gate_task_met": (best.get("mean_task_correctness") or 0.0) >= 0.85,
    }


def _cells_available(cells: list[dict]) -> int:
    return sum(1 for c in cells if c.get("available"))


def write_reports(output_root: Path, cells: list[dict], deltas: dict, case_diff: dict, selection: dict) -> None:
    output_root.mkdir(parents=True, exist_ok=True)

    (output_root / "phase04_16cell_comparison.json").write_text(
        json.dumps(cells, indent=2), encoding="utf-8")
    (output_root / "phase04_deltas.json").write_text(
        json.dumps(deltas, indent=2), encoding="utf-8")

    # CSV
    cols = ["run_id", "model_size", "stage", "beta", "variant", "protocol_pass_rate",
            "mean_task_correctness", "effective_pass_rate", "critical_error_rate",
            "p95_latency_s", "mean_ttft_s", "mean_tokens_per_second", "tool_timing",
            "state_inheritance", "available"]
    with (output_root / "phase04_16cell_comparison.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for c in cells:
            w.writerow({k: c.get(k) for k in cols})

    # MD
    lines = ["# Phase04 16-Cell Frozen Test Comparison (exploratory_currentdata)", ""]
    lines.append("| run_id | model | stage | beta | variant | protocol | task | eff_pass | crit_err |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for c in cells:
        lines.append(
            f"| {c.get('run_id','')} | {c.get('model_size','')} | {c.get('stage','')} | "
            f"{c.get('beta','')} | {c.get('variant','')} | {c.get('protocol_pass_rate','-')} | "
            f"{c.get('mean_task_correctness','-')} | {c.get('effective_pass_rate','-')} | "
            f"{c.get('critical_error_rate','-')} |")
    lines.append("")
    lines.append("## Deltas")
    for model, d in deltas.items():
        lines.append(f"### {model}")
        for kind, d2 in d.items():
            for k, vals in d2.items():
                lines.append(f"- {kind}: {k} -> {vals}")
    lines.append("")
    lines.append("## Exploratory best")
    lines.append(json.dumps(selection, indent=2))
    (output_root / "phase04_16cell_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output_root / "phase04_deltas.md").write_text("\n".join(lines), encoding="utf-8")
    (output_root / "phase04_case_diff.json").write_text(
        json.dumps(case_diff, indent=2), encoding="utf-8")
    (output_root / "phase04_selection.json").write_text(
        json.dumps(selection, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Phase04 12-cell matrix + Phase02 Base")
    parser.add_argument("--output-root", type=str,
                        default="reports/generated/phase04/exploratory_currentdata")
    parser.add_argument("--phase02-root", type=str,
                        default="reports/generated/phase02/v3_final")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    phase02_root = Path(args.phase02_root)

    cells = build_comparison(output_root, phase02_root)
    deltas = build_deltas(cells)
    case_diff = build_case_diff(output_root)
    selection = select_exploratory_best(cells)
    write_reports(output_root, cells, deltas, case_diff, selection)

    n = _cells_available(cells)
    print(f"Cells with metrics: {n}/16")
    print(f"Exploratory best: {selection.get('exploratory_best_run')}")


if __name__ == "__main__":
    main()
