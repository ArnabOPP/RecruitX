"""Shared pytest fixtures/config.

Defaults set here only apply if the environment doesn't already specify a
value (`setdefault`), so CI can still override — e.g. run the full suite
with `CV_PARSER_ENABLE_TRANSFORMER=1` to also exercise the BERT ensemble
path — without editing this file.
"""

import os

# Fast by default: skip the ~10s transformer model load for the bulk of the
# suite. A few tests explicitly re-enable it to cover the ensemble path.
os.environ.setdefault("CV_PARSER_ENABLE_TRANSFORMER", "0")
# Avoid cross-test 429s: the rate limiter's in-memory store is shared across
# every request the TestClient makes within a test session.
os.environ.setdefault("CV_PARSER_RATE_LIMIT_ENABLED", "0")
