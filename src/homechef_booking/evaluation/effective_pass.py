from __future__ import annotations


def effective_pass(protocol_pass: bool, task_correctness: float, critical_error: bool) -> bool:
    return protocol_pass and task_correctness >= 0.95 and not critical_error
