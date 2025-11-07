"""Lightweight shim: ingest utilities were consolidated into `your-agent/ingest.py`.

This file preserves top-level imports referencing `ingest` by dynamically
loading the implementation from the `your-agent/ingest.py` file and re-exporting
the common entrypoints. This avoids duplicating logic while keeping imports
working for scripts that run from the repository root.
"""

import importlib.util
import os
import sys

_impl_path = os.path.join(os.path.dirname(__file__), "your-agent", "ingest.py")
if os.path.exists(_impl_path):
	spec = importlib.util.spec_from_file_location("your_agent_ingest", _impl_path)
	_mod = importlib.util.module_from_spec(spec)
	try:
		spec.loader.exec_module(_mod)  # type: ignore
	except Exception:
		# If loading fails, surface a helpful error when functions are called
		_mod = None
else:
	_mod = None


def _no_impl(*args, **kwargs):
	raise RuntimeError("ingest implementation not available. Check your-agent/ingest.py or restore the file.")


# Re-export functions if present, otherwise provide stubs that raise.
ingest_pdfs = getattr(_mod, "ingest_pdfs", _no_impl)
ingest_url = getattr(_mod, "ingest_url", _no_impl)
