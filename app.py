"""Hugging Face Spaces entry point.

Spaces (SDK: streamlit) runs `app.py` at the repository root. This file just
executes the real app in `app/app.py`, so local and deployed behaviour stay
identical.
"""

from pathlib import Path
import runpy

runpy.run_path(str(Path(__file__).resolve().parent / "app" / "app.py"),
               run_name="__main__")
