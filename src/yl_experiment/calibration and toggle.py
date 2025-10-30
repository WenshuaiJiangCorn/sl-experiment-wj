# Calibration script for the microcontroller
# 9/19/2025 calibration result: 80ms = 10uL

import time
from pathlib import Path
import tempfile

import numpy as np
from ataraxis_base_utilities import LogLevel, console
from ataraxis_data_structures import DataLogger

from microcontroller import AMCInterface

with tempfile.TemporaryDirectory(delete=False) as temp_dir_path:
    output_dir = Path(temp_dir_path).joinpath("test_output")

if not console.enabled:
    console.enable()
_CALIBRATION_PULSE_DURATION = np.uint32(60000) # microseconds
_TOGGLE_DURATION = 2  # seconds

def calibrate_valve():
    """Calibrates the valve by sending a pulse of specified duration."""
    data_logger = DataLogger(output_directory=output_dir, instance_name="calibration_test")
    mc = AMCInterface(data_logger=data_logger)
    console.echo(mc._controller._port)

    try:
        data_logger.start()  # Has to be done before starting any data-generation processes
        mc.start()

        console.echo('Calibration starts')

        mc.left_valve.calibrate(_CALIBRATION_PULSE_DURATION)

    finally:
        mc.left_valve.toggle(state=False)
        mc.stop()
        console.echo("Test: ended.", level=LogLevel.SUCCESS)
        data_logger.stop()

def toggle_valve():
    """Toggles the valve state for a specified duration."""
    data_logger = DataLogger(output_directory=output_dir, instance_name="toggle_test")
    mc = AMCInterface(data_logger=data_logger)
    console.echo(mc._controller._port)

    try:
        data_logger.start()
        mc.start()
        mc.left_valve.toggle(state=True)
        time.sleep(_TOGGLE_DURATION)
    finally:
        mc.left_valve.toggle(state=False)
        mc.stop()
        console.echo("Test: ended.", level=LogLevel.SUCCESS)
        data_logger.stop()

if __name__ == "__main__":
    calibrate_valve()
    # toggle_valve()