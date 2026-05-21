import math

import pytest

from recsys.data import coerce_item_ids


def test_coerce_item_ids_normalizes_sequence_values() -> None:
    assert coerce_item_ids([10, "11", 12.0]) == (10, 11, 12)


def test_coerce_item_ids_handles_empty_scalar_values() -> None:
    assert coerce_item_ids(None) == ()
    assert coerce_item_ids(math.nan) == ()
    assert coerce_item_ids(10) == (10,)


def test_coerce_item_ids_rejects_strings() -> None:
    with pytest.raises(ValueError, match="history_item_ids"):
        coerce_item_ids("10,11")
