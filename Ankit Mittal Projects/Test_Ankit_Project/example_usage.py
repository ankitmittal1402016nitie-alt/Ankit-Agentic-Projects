# example_usage.py
"""
Demonstrates:
- init_logging and set_context
- @log_execution decorator
- log_block context manager
- warnings & exception logging
- import logging
- auto_instrumentation
Run:
    python example_usage.py
"""
import warnings
import time
import importlib

import project_logger as pl

# 1) Initialize logger
pl.init_logging(project_name="Test_Ankit_Project", log_dir="logs", level="DEBUG")
pl.set_context(environment="dev", component="example")

# 2) Decorator example
@pl.log_execution(purpose="demo: add two numbers")
def add(a, b):
    """Add numbers (demo docstring as purpose)."""
    time.sleep(0.05)
    return a + b

# 3) Context manager example
def do_block():
    with pl.log_block("demo: complex block"):
        time.sleep(0.03)
        warnings.warn("This is a demo warning from inside a block")

# 4) Exception logging (caught)
@pl.log_execution(purpose="demo: raising error")
def oops():
    raise ValueError("Something went wrong")

# 5) Import logging (load a stdlib module dynamically)
def test_import_logging():
    importlib.import_module("json")
    importlib.import_module("pathlib")

# 6) Auto-instrument a small dynamic module
def test_auto_instrument():
    # create a tiny module on the fly
    import types
    demo = types.ModuleType("demo_module")

    def greet(name: str):
        """Say hello."""
        time.sleep(0.02)
        return f"Hello, {name}"

    class Math:
        """Simple math class."""
        def mul(self, x, y):
            time.sleep(0.01)
            return x * y

    demo.greet = greet
    demo.Math = Math

    pl.auto_instrument(demo, instrument_classes=True, instrument_functions=True)
    print(demo.greet("Ankit"))
    print(demo.Math().mul(3, 4))

if __name__ == "__main__":
    print("add:", add(2, 3))
    do_block()
    try:
        oops()
    except Exception:
        pass
    test_import_logging()
    test_auto_instrument()
    print("Done. Check the logs/ folder (JSON and text files).")
