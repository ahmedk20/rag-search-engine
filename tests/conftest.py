import os
import sys
from pathlib import Path

# Several lib modules validate OPENROUTER_API_KEY at import time (they build an
# OpenAI client for LLM-backed features). Set a placeholder so pure-function
# tests can import them without requiring a real key or making a network call.
os.environ.setdefault("OPENROUTER_API_KEY", "test-key-not-a-real-key")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "cli"))
