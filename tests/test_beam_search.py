import numpy as np
import pytest

from recsys.decoding import StaticTransitionMatrixDecoder, constrained_beam_search


def test_constrained_beam_search_only_returns_valid_semantic_ids() -> None:
    decoder = StaticTransitionMatrixDecoder.from_semantic_ids(
        [(1, 2), (1, 3), (4, 5)],
        vocab_size=10,
    )

    def logits_provider(
        prefixes: tuple[tuple[int, ...], ...],
        _states: np.ndarray,
        step: int,
    ) -> np.ndarray:
        logits = np.full((len(prefixes), decoder.vocab_size), -10.0)
        logits[:, 9] = 100.0
        if step == 0:
            logits[:, 1] = 2.0
            logits[:, 4] = 1.0
            return logits
        for row_index, prefix in enumerate(prefixes):
            if prefix == (1,):
                logits[row_index, 3] = 3.0
                logits[row_index, 2] = 2.0
            elif prefix == (4,):
                logits[row_index, 5] = 4.0
        return logits

    results = constrained_beam_search(
        decoder=decoder,
        depth=2,
        beam_size=3,
        logits_provider=logits_provider,
    )

    assert results[0].semantic_id == (1, 3)
    assert {result.semantic_id for result in results} <= {(1, 2), (1, 3), (4, 5)}
    assert all(result.score > 0 for result in results)


def test_constrained_beam_search_rejects_bad_logits_shape() -> None:
    decoder = StaticTransitionMatrixDecoder.from_semantic_ids([(1, 2)], vocab_size=3)

    with pytest.raises(ValueError, match="logits shape"):
        constrained_beam_search(
            decoder=decoder,
            depth=2,
            beam_size=1,
            logits_provider=lambda _prefixes, _states, _step: np.zeros((1, 2)),
        )
