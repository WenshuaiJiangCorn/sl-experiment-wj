"""This module stores miscellaneous tools and utilities shared by other packages in the library."""

import sys

from importlib_metadata import metadata as _metadata


def get_version_data() -> tuple[str, str]:
    """Determines and returns the current Python and sl-experiment versions as a string of two tuples.

    The first element of the returned tuple is the Python version, while the second element is the sl-experiment
    version.
    """
    # Determines the local Python version and the version of the sl-experiment library.
    sl_experiment_version = _metadata("sl-experiment")["version"]  # type: ignore
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"  # Python version
    return python_version, sl_experiment_version
