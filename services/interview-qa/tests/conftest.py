"""Shared pytest fixtures/config.

Defaults set here only apply if the environment doesn't already specify a
value (`setdefault`), so CI can still override.
"""

import os

os.environ.setdefault("INTERVIEW_QA_RATE_LIMIT_ENABLED", "0")
