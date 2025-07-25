def run() -> None:
    """Entry point for uv tool."""
    from newsensor_streamlit.ui.app import start_streamlit_app
    start_streamlit_app()