"""Task 7: Semantic slot tests."""

from homechef_booking.evaluation.scorers.semantic_slots import semantic_slot_f1


class FakeEmbedder:
    def similarity(self, left: str, right: str) -> float:
        if {left, right} == {"不吃花生", "花生过敏"}:
            return 0.91
        return 1.0 if left == right else 0.0


def test_semantic_slot_f1_uses_one_to_one_matching():
    result = semantic_slot_f1(["不吃花生", "水煮鱼"], ["花生过敏", "水煮鱼"], FakeEmbedder(), threshold=0.70)
    assert result == {"precision": 1.0, "recall": 1.0, "f1": 1.0}
