# callbacks_logging.py
from __future__ import annotations
import time, statistics as st
from typing import Any, List, Dict
from project_logger import get_logger, _log
from langchain_core.callbacks import BaseCallbackHandler

class LoggerCallback(BaseCallbackHandler):
    """Concise, actionable RAG telemetry."""
    def __init__(self, name: str = "lc-callback"):
        self.lg = get_logger(name)
        self.t: Dict[str, float] = {}

    def _start(self, k): self.t[k] = time.time()
    def _end(self, k):
        t = self.t.pop(k, None)
        return int((time.time()-t)*1000) if t else -1

    # Retrieval
    def on_retriever_start(self, serialized: Dict[str, Any], query: str, **_):
        self._start("retr")
        _log(self.lg, "INFO", "retriever-start", query_len=len(query))

    def on_retriever_end(self, documents: List[Any], **_):
        dur = self._end("retr")
        scores = [d.metadata.get("relevance_score") for d in (documents or []) if d.metadata.get("relevance_score") is not None]
        stats = {}
        if scores:
            stats = dict(
                mean=round(st.mean(scores),3),
                median=round(st.median(scores),3),
                min=round(min(scores),3),
                max=round(max(scores),3),
                std=round(st.pstdev(scores),3),
            )
        _log(self.lg, "INFO", "retriever-end", hits=len(documents or []), elapsed_ms=dur, score_stats=stats)

    # LLM
    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **_):
        self._start("llm")
        model = serialized.get("kwargs", {}).get("model", "")
        ctxt = prompts[0] if prompts else ""
        _log(self.lg, "INFO", "llm-start", model=model, prompts=len(prompts), context_chars=len(ctxt))

    def on_llm_end(self, response, **_):
        dur = self._end("llm")
        usage = getattr(response, "llm_output", {}).get("token_usage", None)
        _log(self.lg, "INFO", "llm-end", elapsed_ms=dur, usage=usage)
