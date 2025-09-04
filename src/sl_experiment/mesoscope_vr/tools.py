"""This module provides additional tools and classes used by other modules of the mesoscope_vr package. Primarily, this
includes various dataclasses specific to the Mesoscope-VR systems and utility functions used by other package modules.
The contents of this module are not intended to be used outside the mesoscope_vr package."""

import sys
from pathlib import Path
from dataclasses import field, dataclass
from multiprocessing import Process

import numpy as np
from PyQt6.QtGui import QFont, QCloseEvent
from PyQt6.QtCore import Qt, QTimer
from numpy.typing import NDArray
from PyQt6.QtWidgets import (
    QLabel,
    QWidget,
    QGroupBox,
    QHBoxLayout,
    QMainWindow,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QApplication,
    QDoubleSpinBox,
)
from sl_shared_assets import SessionData, SessionTypes, MesoscopeSystemConfiguration, get_system_configuration_data
from ataraxis_base_utilities import console, ensure_directory_exists
from ataraxis_data_structures import SharedMemoryArray


def get_system_configuration() -> MesoscopeSystemConfiguration:
    """Verifies that the current data acquisition system is the Mesoscope-VR and returns its configuration data.

    Raises:
        ValueError: If the local data acquisition system is not a Mesoscope-VR system.
    """
    system_configuration = get_system_configuration_data()
    if not isinstance(system_configuration, MesoscopeSystemConfiguration):
        message = (
            f"Unable to instantiate the MesoscopeData class, as the local data acquisition system is not a "
            f"Mesoscope-VR system. This either indicates a user error (calling incorrect Data class) or local data "
            f"acquisition system misconfiguration. To reconfigured the data-acquisition system, use the "
            f"sl-create-system-config' CLI command."
        )
        console.error(message, error=ValueError)
    return system_configuration