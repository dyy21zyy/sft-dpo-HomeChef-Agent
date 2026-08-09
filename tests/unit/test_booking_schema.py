import pytest
from pydantic import ValidationError

from homechef_booking.schemas.booking import BookingSlot, is_info_complete, missing_required_slots


def test_required_slots_use_contract_order() -> None:
    slot = BookingSlot()
    assert missing_required_slots(slot) == ["service_date", "start_time", "people", "address"]
    assert is_info_complete(slot) is False


def test_complete_required_slots_are_complete() -> None:
    slot = BookingSlot(service_date="2026-08-15", start_time="18:00", people=6, address="杨浦")
    assert missing_required_slots(slot) == []
    assert is_info_complete(slot) is True


@pytest.mark.parametrize("bad", ["24:00", "18:60"])
def test_start_time_rejects_invalid_clock_values(bad: str) -> None:
    with pytest.raises(ValidationError):
        BookingSlot(start_time=bad)


def test_service_date_rejects_non_calendar_date() -> None:
    with pytest.raises(ValidationError):
        BookingSlot(service_date="2026-02-31")


def test_strict_schema_rejects_type_coercion() -> None:
    with pytest.raises(ValidationError):
        BookingSlot(people="6")
    with pytest.raises(ValidationError):
        BookingSlot(ingredient_purchase=1)
