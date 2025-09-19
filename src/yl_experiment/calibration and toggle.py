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

def calibrate_valve():
    """Calibrates the valve by sending a pulse of specified duration."""
    data_logger = DataLogger(output_directory=output_dir, exist_ok=True)
    mc = AMCInterface(data_logger=data_logger)
    console.echo(mc._controller._port)

    try:
        data_logger.start()  # Has to be done before starting any data-generation processes
        mc.start()

        console.echo('Calibration starts')

        #mc.left_valve.calibrate(_CALIBRATION_PULSE_DURATION)
        mc.left_valve.toggle(state=True)
        time.sleep(0.2)
    finally:
        mc.left_valve.toggle(state=False)
        mc.stop()
        console.echo("Test: ended.", level=LogLevel.SUCCESS)

        data_logger.stop()

if __name__ == "__main__":
    calibrate_valve()