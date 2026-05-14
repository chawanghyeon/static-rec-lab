from pathlib import Path

import pytest

from recsys.benchmark import (
    DecoderBenchmarkConfig,
    DecoderBenchmarkError,
    generate_synthetic_semantic_ids,
    run_decoder_benchmark,
    write_decoder_benchmark_report,
)


def test_generate_synthetic_semantic_ids_is_unique_and_reproducible() -> None:
    semantic_ids = generate_synthetic_semantic_ids(
        num_semantic_ids=64,
        depth=3,
        vocab_size=16,
        random_seed=7,
    )
    repeated = generate_synthetic_semantic_ids(
        num_semantic_ids=64,
        depth=3,
        vocab_size=16,
        random_seed=7,
    )

    assert semantic_ids == repeated
    assert len(semantic_ids) == 64
    assert len(set(semantic_ids)) == 64
    assert {len(semantic_id) for semantic_id in semantic_ids} == {3}


def test_generate_synthetic_semantic_ids_rejects_impossible_cardinality() -> None:
    with pytest.raises(DecoderBenchmarkError, match="고유 Semantic ID"):
        generate_synthetic_semantic_ids(
            num_semantic_ids=5,
            depth=2,
            vocab_size=2,
            random_seed=7,
        )


def test_run_decoder_benchmark_supports_configured_batch_sizes() -> None:
    summary = run_decoder_benchmark(
        config=DecoderBenchmarkConfig(
            batch_sizes=(1, 4),
            warmup_iterations=1,
            iterations=2,
            num_semantic_ids=32,
            semantic_id_depth=3,
            vocab_size=16,
            random_seed=7,
        )
    )

    assert summary.source == "synthetic"
    assert summary.num_semantic_ids == 32
    assert [result.batch_size for result in summary.results] == [1, 4]
    for result in summary.results:
        assert result.masks_identical is True
        assert result.naive_latency_ms >= 0
        assert result.static_latency_ms >= 0
        assert result.naive_throughput_rows_per_s > 0
        assert result.static_throughput_rows_per_s > 0


def test_run_decoder_benchmark_rejects_invalid_config() -> None:
    with pytest.raises(DecoderBenchmarkError, match="vocab_size"):
        run_decoder_benchmark(
            config=DecoderBenchmarkConfig(
                batch_sizes=(1,),
                warmup_iterations=0,
                iterations=1,
                num_semantic_ids=4,
                semantic_id_depth=2,
                vocab_size=0,
            )
        )


def test_write_decoder_benchmark_report(tmp_path: Path) -> None:
    summary = run_decoder_benchmark(
        config=DecoderBenchmarkConfig(
            batch_sizes=(1,),
            warmup_iterations=1,
            iterations=1,
            num_semantic_ids=16,
            semantic_id_depth=2,
            vocab_size=8,
            random_seed=7,
        )
    )
    report_path = write_decoder_benchmark_report(tmp_path / "decoder_benchmark.md", summary)

    report = report_path.read_text(encoding="utf-8")
    assert "# Decoder Benchmark" in report
    assert "batch_size" in report
    assert "mask 일치" in report
    assert "STATIC rows/s" in report
