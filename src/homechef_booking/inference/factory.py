from __future__ import annotations

import json
from pathlib import Path

import yaml

from homechef_booking.inference.backend import Backend
from homechef_booking.inference.mock_backend import MockBackend


def load_backend(config_path: Path) -> Backend:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config["backend"] != "mock":
        raise NotImplementedError(f"Phase 01 supports only mock backend, got {config['backend']}")
    if "predictions_path" in config:
        predictions_path = Path(config["predictions_path"])
        predictions = json.loads(predictions_path.read_text(encoding="utf-8"))
    else:
        predictions = config.get("predictions", {})
    return MockBackend(predictions=predictions)
