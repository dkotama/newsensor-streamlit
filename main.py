"""Entry point for running the application."""
from __future__ import annotations

import sys
from pathlib import Path

# Add src to Python path for development
sys.path.insert(0, str(Path(__file__).parent / "src"))

from newsensor_streamlit.ui.app import start_streamlit_app

if __name__ == "__main__":
    start_streamlit_app()