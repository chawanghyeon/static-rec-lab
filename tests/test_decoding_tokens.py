import numpy as np
import pytest

from recsys.decoding.tokens import normalize_token, normalize_tokens, try_normalize_tokens


def test_normalize_tokens_accepts_python_and_numpy_integers() -> None:
    assert normalize_tokens([1, np.int64(2), 3]) == (1, 2, 3)


def test_normalize_tokens_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="문자열"):
        normalize_tokens("123")
    with pytest.raises(ValueError, match="비어"):
        normalize_tokens([])
    with pytest.raises(ValueError, match="bool"):
        normalize_token(True)
    with pytest.raises(ValueError, match="0 이상"):
        normalize_token(-1)


def test_try_normalize_tokens_returns_none_on_invalid_values() -> None:
    assert try_normalize_tokens("123", allow_empty=True) is None
    assert try_normalize_tokens([], allow_empty=True) == ()
