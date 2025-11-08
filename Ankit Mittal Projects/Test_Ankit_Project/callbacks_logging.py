# callbacks_logging.py

# Annotations for better type hints (Python 3.7+). Notes for humans and tools that explain what kind of data is expected, making code easier to read, debug, and maintain.
from __future__ import annotations

import time
from typing import Any, List, Dict, Optional, Tuple
from project_logger import get_logger, _log
from langchain_core.callbacks import BaseCallbackHandler

class LoggerCallback(BaseCallbackHandler):
    """Logs retriever, LLM, chain, tool events with timings + stats."""
    def __init__(self, name: str = "lc-callback"):
        self.lg = get_logger(name)
        self._t = {}

    # --- helpers ---
    def _start(self, key: str):
        self._t[key] = time.time()
    def _end(self, key: str) -> int:
        t = self._t.pop(key, None)
        return int((time.time() - t) * 1000) if t else -1

    # --- retriever ---
    def on_retriever_start(self, serialized: Dict[str, Any], query: str, **kwargs) -> None:
        self._start("retr")
        _log(self.lg, "INFO", "retriever-start", query=query, retriever=str(serialized.get("id", "")))

    def on_retriever_end(self, documents: List[Any], **kwargs) -> None:
        dur = self._end("retr")
        # Try to include relevance scores if caller attached them on docs metadata
        scores = []
        for d in documents or []:
            s = None
            try:
                s = d.metadata.get("relevance_score")
            except Exception:
                pass
            scores.append(s)
        _log(self.lg, "INFO", "retriever-end", hits=len(documents or []), elapsed_ms=dur, scores=scores[:10])

    # --- LLM ---
    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs) -> None:
        self._start("llm")
        _log(self.lg, "INFO", "llm-start", model=str(serialized.get("kwargs", {}).get("model", "")),
             prompts_count=len(prompts), prompt_preview=(prompts[0][:240] if prompts else ""))

    def on_llm_end(self, response, **kwargs) -> None:
        dur = self._end("llm")
        # Try to get token usage if present
        usage = None
        try:
            usage = response.llm_output.get("token_usage")
        except Exception:
            pass
        _log(self.lg, "INFO", "llm-end", elapsed_ms=dur, usage=usage)

    # --- chain ---
    def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs) -> None:
        self._start("chain")
        _log(self.lg, "DEBUG", "chain-start", chain=str(serialized.get("id", "")))

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs) -> None:
        dur = self._end("chain")
        preview = str(outputs)[:240] if outputs is not None else ""
        _log(self.lg, "DEBUG", "chain-end", elapsed_ms=dur, output_preview=preview)
