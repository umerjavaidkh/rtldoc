"""RAGBench -- an ingestion-quality benchmark for rtldoc.

Modelled on run-llama/ParseBench (five equally-weighted capability dimensions,
deterministic rule-based scoring, no LLM judge), but re-aimed at the thing
rtldoc is actually for: feeding a retrieval pipeline.

See RAGBENCH.md for the design argument and how to read the numbers.
"""
__all__ = ["chunking", "rules", "metrics", "retrieval", "runner", "report"]
