"""User-facing absolute paths that retain symbolic-link components."""
import os
from pathlib import Path


def absolute_path(value, directory=None, *, strict=False):
    """Expand home and normalize dot components without dereferencing links.

    Relative paths use the declaring document's logical directory, or the
    shell's logical PWD when it still names the actual working directory.
    Existence checks follow links but do not replace their spelling.
    """
    path = Path(value).expanduser()
    if not path.is_absolute():
        if directory is None:
            directory = Path.cwd()
            logical = os.environ.get('PWD', '')
            try:
                if Path(logical).is_absolute() and os.path.samefile(logical, directory):
                    directory = Path(logical)
            except OSError:
                pass
        path = Path(directory) / path
    path = Path(os.path.abspath(path))
    if strict:
        path.stat()
    return path
