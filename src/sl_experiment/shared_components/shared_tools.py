"""This module stores miscellaneous tools and utilities shared by other packages in the library."""

import sys
from pathlib import Path

from sl_shared_assets import SessionData, VersionData
from importlib_metadata import metadata as _metadata


def write_version_data(session_data: SessionData) -> None:
    """Writes the current Python and sl-experiment version to disk as a version_data.yaml file.

    This service function is used to cache the version data used at runtime.

    Args:
        session_data: An initialized SessionData instance for the session whose data is being acquired.
    """

    # Determines where to cache the version data.
    output_path: Path = session_data.raw_data.version_data_path

    # Determines the local Python version and the version of the sl-experiment library.
    # sl-experiment version
    sl_experiment_version = _metadata("sl-experiment")["version"]  # type: ignore
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"  # Python version

    # Packages version data inside the VersionData object
    version_data = VersionData(python_version=python_version, sl_experiment_version=sl_experiment_version)

    # Caches the version data to disk as a version_data.yaml file.
    version_data.to_yaml(file_path=output_path)
