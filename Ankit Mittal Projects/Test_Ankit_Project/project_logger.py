# project_logger.py
"""
Production-grade logging utility with GLOBAL per-call tracing.

What you get:
- Rotating file handlers (JSON + text) and colorized console
- Structured fields: timestamp(ms), project, module, path, function, purpose,
  caller(file, func, line), start/end/duration(ms), outcome, exception/warning
- Decorator (@log_execution) and context manager (log_block) with timing
- Global warnings + uncaught exception capture
- Import hook: logs module imports (module name, origin/path)
- Auto-instrumentation for modules/classes/functions you own
- 🔥 Global per-call tracer using sys.setprofile to log EVERY function call/return
  inside your project directory, without adding decorators

Standard library only.
"""

from __future__ import annotations
import logging
import warnings
import sys
import os
import time
import json
import inspect
import traceback
import threading
from datetime import datetime
from functools import wraps
from logging.handlers import RotatingFileHandler
from types import ModuleType
from typing import Any, Callable, Dict, Optional

# ------------------------------
# Configuration (env-overridable)
# ------------------------------
PROJECT_NAME = os.getenv("PROJECT_NAME", "Test_Ankit_Project")
LOG_DIR = os.getenv("LOG_DIR", "logs")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
MAX_BYTES = int(os.getenv("LOG_MAX_BYTES", 5 * 1024 * 1024))   # 5MB
BACKUP_COUNT = int(os.getenv("LOG_BACKUP_COUNT", 7))
ENABLE_CONSOLE = os.getenv("LOG_CONSOLE", "true").lower() == "true"
ENABLE_JSON_FILE = os.getenv("LOG_JSON_FILE", "true").lower() == "true"
ENABLE_TEXT_FILE = os.getenv("LOG_TEXT_FILE", "true").lower() == "true"
COLOR_CONSOLE = os.getenv("LOG_COLOR", "true").lower() == "true"

# tracing filters
DEFAULT_PROJECT_ROOT = os.path.abspath(os.getenv("PROJECT_ROOT", os.getcwd()))
TRACE_INCLUDE = [DEFAULT_PROJECT_ROOT]  # paths we allow tracing
TRACE_EXCLUDE_SUBSTR = [os.path.sep + "site-packages" + os.path.sep, os.path.sep + "dist-packages" + os.path.sep]

os.makedirs(LOG_DIR, exist_ok=True)
DATE = datetime.now().strftime("%Y-%m-%d")
JSON_LOG_PATH = os.path.join(LOG_DIR, f"project_log_{DATE}.jsonl")
TEXT_LOG_PATH = os.path.join(LOG_DIR, f"project_log_{DATE}.log")

# ------------------------------
# Global state
# ------------------------------
_initialized = False
_loggers: Dict[str, logging.Logger] = {}
_global_extra: Dict[str, Any] = {}  # attached to every record
_import_hook_installed = False

# per-thread call timing store for tracer
_tls = threading.local()

# ------------------------------
# Formatters
# ------------------------------
class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.fromtimestamp(record.created).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            "level": record.levelname,
            "project": PROJECT_NAME,
            "logger": record.name,
            "module": record.module,
            #"module_path": getattr(record, "pathname", ""),
            "function": record.funcName,
            "line": record.lineno,
            "message": record.getMessage(),
        }
        extra_data = getattr(record, "extra_data", {})
        if isinstance(extra_data, dict):
            payload.update(extra_data)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)

class TextFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        ts = datetime.fromtimestamp(record.created).strftime("%H:%M:%S.%f")[:-3]
        base = f"{ts} {record.levelname:<8} [{PROJECT_NAME}] {record.name}: {record.getMessage()}"
        extra_data = getattr(record, "extra_data", None)
        if isinstance(extra_data, dict) and extra_data:
            base += f" | {extra_data}"
        if record.exc_info:
            base += "\n" + self.formatException(record.exc_info)
        return base

class ColorFormatter(TextFormatter):
    COLORS = {"DEBUG":"\033[37m","INFO":"\033[36m","WARNING":"\033[33m","ERROR":"\033[31m","CRITICAL":"\033[41m","RESET":"\033[0m"}
    def format(self, record: logging.LogRecord) -> str:
        s = super().format(record)
        if not COLOR_CONSOLE:
            return s
        return f"{self.COLORS.get(record.levelname,'')}{s}{self.COLORS['RESET']}"

# ------------------------------
# Core setup
# ------------------------------
def _build_handlers() -> list[logging.Handler]:
    handlers: list[logging.Handler] = []
    if ENABLE_JSON_FILE:
        json_fh = RotatingFileHandler(JSON_LOG_PATH, maxBytes=MAX_BYTES, backupCount=BACKUP_COUNT, encoding="utf-8")
        json_fh.setFormatter(JsonFormatter())
        handlers.append(json_fh)
    if ENABLE_TEXT_FILE:
        txt_fh = RotatingFileHandler(TEXT_LOG_PATH, maxBytes=MAX_BYTES, backupCount=BACKUP_COUNT, encoding="utf-8")
        txt_fh.setFormatter(TextFormatter())
        handlers.append(txt_fh)
    if ENABLE_CONSOLE:
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setFormatter(ColorFormatter())
        handlers.append(ch)
    return handlers

def _ensure_root_logger():
    global _initialized
    if _initialized:
        return
    root = logging.getLogger()
    root.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
    for h in _build_handlers():
        root.addHandler(h)
    logging.captureWarnings(True)   # warnings → logging
    sys.excepthook = _excepthook    # uncaught exceptions → logging
    _initialized = True

def get_logger(name: str = "app") -> logging.Logger:
    _ensure_root_logger()
    if name in _loggers:
        return _loggers[name]
    lg = logging.getLogger(name)
    lg.propagate = True
    _loggers[name] = lg
    return lg

def set_context(**kv):
    """Attach key/values to every subsequent log entry."""
    _global_extra.update(kv)

def _log(lg: logging.Logger, level: str, msg: str, **extra):
    merged = dict(_global_extra)
    merged.update(extra)
    lg.log(getattr(logging, level, logging.INFO), msg, extra={"extra_data": merged})

# ------------------------------
# Exception & warnings
# ------------------------------
def _excepthook(exc_type, exc, tb):
    lg = get_logger("uncaught")
    _log(lg, "ERROR", "uncaught-exception", exception=str(exc))
    lg.error("uncaught-exception", exc_info=(exc_type, exc, tb))

def enable_warning_capture():
    logging.captureWarnings(True)
    warnings.simplefilter("default")

# ------------------------------
# Import logging
# ------------------------------
class _ImportLoggerFinder:
    def __init__(self):
        self._lg = get_logger("import")
    def find_spec(self, fullname, path=None, target=None):
        for finder in sys.meta_path:
            if finder is self:
                continue
            try:
                spec = finder.find_spec(fullname, path, target) if hasattr(finder, "find_spec") else None
            except Exception:
                continue
            if spec is not None:
                origin = getattr(spec, "origin", None)
                _log(self._lg, "DEBUG", "module-import", module=fullname, origin=origin)
                return spec
        return None

def enable_import_logging():
    global _import_hook_installed
    if _import_hook_installed:
        return
    sys.meta_path.insert(0, _ImportLoggerFinder())
    _import_hook_installed = True

# ------------------------------
# Decorators & Context managers
# ------------------------------
def _caller_info():
    try:
        frame = inspect.stack()[2]
        return {"caller_func": frame.function, "caller_line": frame.lineno}
    except Exception:
        return {}

def log_execution(purpose: Optional[str] = None, level: str = "INFO"):
    """Decorator to log start/end, duration, outcome, exceptions for a function."""
    def deco(fn: Callable):
        lg = get_logger(fn.__module__ or "app")
        fn_purpose = purpose or (fn.__doc__.strip() if fn.__doc__ else "")
        @wraps(fn)
        def wrapper(*args, **kwargs):
            start = time.time()
            ci = _caller_info()
            _log(lg, level, "start", purpose=fn_purpose, function=fn.__name__, **ci)
            try:
                res = fn(*args, **kwargs)
                elapsed = int((time.time() - start) * 1000)
                _log(lg, level, "end", purpose=fn_purpose, function=fn.__name__, elapsed_ms=elapsed, outcome="success")
                return res
            except Exception as e:
                elapsed = int((time.time() - start) * 1000)
                etb = traceback.format_exc()
                _log(lg, "ERROR", "exception", purpose=fn_purpose, function=fn.__name__, elapsed_ms=elapsed,
                     outcome="error", exception=str(e), traceback=etb)
                raise
        return wrapper
    return deco

class log_block:
    """Context manager to time any code block."""
    def __init__(self, purpose: str, logger: Optional[logging.Logger] = None, level: str = "INFO"):
        self.purpose = purpose
        self.lg = logger or get_logger("block")
        self.level = level
        self.start = None
    def __enter__(self):
        self.start = time.time()
        _log(self.lg, self.level, "start", purpose=self.purpose, **_caller_info())
        return self
    def __exit__(self, exc_type, exc, tb):
        elapsed = int((time.time() - self.start) * 1000)
        if exc:
            etb = "".join(traceback.format_exception(exc_type, exc, tb))
            _log(self.lg, "ERROR", "exception", purpose=self.purpose, elapsed_ms=elapsed,
                 outcome="error", exception=str(exc), traceback=etb)
            return False
        _log(self.lg, self.level, "end", purpose=self.purpose, elapsed_ms=elapsed, outcome="success")
        return False

# ------------------------------
# Auto-instrumentation helpers
# ------------------------------
def _wrap_function(func: Callable, logger_name: str) -> Callable:
    if getattr(func, "__pl_wrapped__", False):
        return func
    lg = get_logger(logger_name)
    purpose = (func.__doc__ or "").strip()
    @wraps(func)
    def wrapped(*args, **kwargs):
        start = time.time()
        ci = _caller_info()
        _log(lg, "INFO", "start", purpose=purpose, function=func.__name__, **ci)
        try:
            out = func(*args, **kwargs)
            elapsed = int((time.time() - start) * 1000)
            _log(lg, "INFO", "end", purpose=purpose, function=func.__name__, elapsed_ms=elapsed, outcome="success")
            return out
        except Exception as e:
            elapsed = int((time.time() - start) * 1000)
            etb = traceback.format_exc()
            _log(lg, "ERROR", "exception", purpose=purpose, function=func.__name__, elapsed_ms=elapsed,
                 outcome="error", exception=str(e), traceback=etb)
            raise
    setattr(wrapped, "__pl_wrapped__", True)
    return wrapped

def auto_instrument(module: ModuleType,
                    include_private: bool = False,
                    instrument_classes: bool = True,
                    instrument_functions: bool = True,
                    logger_name: Optional[str] = None):
    """Wraps functions and class methods in a module so calls are logged."""
    lname = logger_name or module.__name__
    for attr_name, obj in list(vars(module).items()):
        if not include_private and attr_name.startswith("_"):
            continue
        if instrument_functions and inspect.isfunction(obj) and obj.__module__ == module.__name__:
            setattr(module, attr_name, _wrap_function(obj, lname))
        if instrument_classes and inspect.isclass(obj) and obj.__module__ == module.__name__:
            for m_name, m_obj in list(vars(obj).items()):
                if not include_private and (m_name.startswith("_") and not (m_name.startswith("__") and m_name.endswith("__"))):
                    continue
                if inspect.isfunction(m_obj):
                    setattr(obj, m_name, _wrap_function(m_obj, lname))

# ------------------------------
# 🔥 GLOBAL per-call tracer
# ------------------------------
def _should_trace_file(filename: str) -> bool:
    if not filename:
        return False
    fn = os.path.abspath(filename)
    if any(s in fn for s in TRACE_EXCLUDE_SUBSTR):
        return False
    return any(fn.startswith(os.path.abspath(root)) for root in TRACE_INCLUDE)

def _frame_key(frame) -> int:
    # unique key per frame object
    return id(frame)

def _trace_profile(frame, event, arg):
    # Only trace Python function calls/returns
    if event not in ("call", "return", "exception"):
        return
    code = frame.f_code
    filename = code.co_filename
    if not _should_trace_file(filename):
        return

    # prepare TLS store
    if not hasattr(_tls, "starts"):
        _tls.starts = {}
    starts: Dict[int, float] = _tls.starts

    lg = get_logger("trace")
    func_name = code.co_name
    module = frame.f_globals.get("__name__", "")
    line_no = frame.f_lineno
    now = time.time()
    key = _frame_key(frame)

    if event == "call":
        starts[key] = now
        # try to get first comment or docstring line
        purpose = ""
        try:
            doc = inspect.getdoc(frame.f_globals.get(func_name)) or ""
            if doc:
                purpose = doc.splitlines()[0][:200]
        except Exception:
            pass
        # caller info
        ci = {}
        try:
            caller = frame.f_back
            if caller:
                ci = {"caller_func": caller.f_code.co_name, "caller_line": caller.f_lineno}
        except Exception:
            pass
        _log(lg, "DEBUG", "start",
             project_module=module,
             #module_path=filename,
             function=func_name,
             line=line_no,
             purpose=purpose,
             **ci)
        return

    if event in ("return", "exception"):
        start = starts.pop(key, None)
        elapsed_ms = int((now - start) * 1000) if start else None
        if event == "return":
            _log(lg, "DEBUG", "end",
                 project_module=module,
                 #module_path=filename,
                 function=func_name,
                 line=line_no,
                 elapsed_ms=elapsed_ms,
                 outcome="success")
        else:
            etype, exc, tb = arg
            _log(lg, "ERROR", "exception",
                 project_module=module,
                 #module_path=filename,
                 function=func_name,
                 line=line_no,
                 elapsed_ms=elapsed_ms,
                 outcome="error",
                 exception=str(exc),
                 traceback="".join(traceback.format_exception(etype, exc, tb)))

def enable_global_tracing(project_root: Optional[str] = None, include_paths: Optional[list[str]] = None):
    """
    Enable per-function call/return logging across your project (no decorators needed).
    Only files under project_root / include_paths are traced.
    """
    if include_paths:
        TRACE_INCLUDE[:] = [os.path.abspath(p) for p in include_paths]
    else:
        root = os.path.abspath(project_root or DEFAULT_PROJECT_ROOT)
        TRACE_INCLUDE[:] = [root]
    sys.setprofile(_trace_profile)   # covers all threads by default on CPython

def disable_global_tracing():
    sys.setprofile(None)

# ------------------------------
# Initialization helper
# ------------------------------
def init_logging(project_name: Optional[str] = None, log_dir: Optional[str] = None, level: str = LOG_LEVEL,
                 enable_imports: bool = True, enable_warnings: bool = True):
    """Call once at app startup."""
    global PROJECT_NAME
    global LOG_DIR
    if project_name:
        PROJECT_NAME = project_name
    if log_dir:
        LOG_DIR = log_dir
    _ensure_root_logger()
    logging.getLogger().setLevel(getattr(logging, level, logging.INFO))
    if enable_warnings:
        enable_warning_capture()
    if enable_imports:
        enable_import_logging()
