from sl_shared_assets import SessionData

def write_version_data(session_data: SessionData) -> None:
    """Writes the current Python and sl-experiment version to disk as a version_data.yaml file.

    This service function is used to cache the version data used at runtime.

    Args:
        session_data: An initialized SessionData instance for the session whose data is being acquired.
    """
