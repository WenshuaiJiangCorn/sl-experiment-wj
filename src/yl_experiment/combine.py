import time
from pathlib import Path

import numpy as np
import keyboard
from ataraxis_base_utilities import LogLevel, console
from ataraxis_data_structures import DataLogger, assemble_log_archives

from .data_processing import process_microcontroller_log
from .microcontroller import AMCInterface
from visualizers import BehaviorVisualizer