from .frame import BUILTIN_METRICS, EvalResult, Evalframe

__all__ = ["Evalframe", "EvalResult", "BUILTIN_METRICS", "launch_gui"]
__version__ = "0.1.0"


def launch_gui() -> None:
    """Open the evalframe graphical interface (requires tkinter)."""
    try:
        from .gui import main  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError(
            "The evalframe GUI requires tkinter. "
            "Install it with your system package manager "
            "(e.g. 'apt install python3-tk' on Debian/Ubuntu)."
        ) from exc
    main()
