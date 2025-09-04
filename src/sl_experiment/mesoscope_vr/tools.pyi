from pathlib import Path
from dataclasses import field, dataclass

import numpy as np
from _typeshed import Incomplete
from PyQt6.QtGui import QCloseEvent
from numpy.typing import NDArray as NDArray
from PyQt6.QtWidgets import QMainWindow
from sl_shared_assets import SessionData, MesoscopeSystemConfiguration
from ataraxis_data_structures import SharedMemoryArray

def get_system_configuration() -> MesoscopeSystemConfiguration:
    """Verifies that the current data acquisition system is the Mesoscope-VR and returns its configuration data.

    Raises:
        ValueError: If the local data acquisition system is not a Mesoscope-VR system.
    """