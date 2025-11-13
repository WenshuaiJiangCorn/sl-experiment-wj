# Calibration script for the microcontroller
from pathlib import Path
import tempfile

import numpy as np
import keyboard
from ataraxis_time import PrecisionTimer
from microcontroller import AMCInterface
from ataraxis_base_utilities import LogLevel, console
from ataraxis_data_structures import DataLogger


def calibrate_valve(valve_side) -> None:
    """Calibrates the valve by sending a pulse of specified duration."""
    data_logger = DataLogger(output_directory=output_dir, instance_name="calibration_test")
    mc = AMCInterface(data_logger=data_logger)
    console.echo(mc._controller._port)

    if valve_side == "left":
        valve = mc.left_valve
    elif valve_side == "right":
        valve = mc.right_valve
    
    else:
        console.echo("Invalid valve side specified.", level=LogLevel.ERROR)
        return

    try:
        data_logger.start()  # Has to be done before starting any data-generation processes
        mc.start()
        mc.connect_to_smh()

        console.echo("Calibration starts")

        valve.calibrate(_CALIBRATION_PULSE_DURATION)

    finally:
        valve.toggle(state=False)
        mc.disconnect_to_smh()
        mc.stop()
        console.echo("Test: ended.", level=LogLevel.SUCCESS)
        data_logger.stop()


def toggle_valve(valve_side, duration = 1) -> None:
    """Opens the valve state for a specified duration. Default is 1 second"""
    data_logger = DataLogger(output_directory=output_dir, instance_name="toggle_test")
    mc = AMCInterface(data_logger=data_logger)
    console.echo(mc._controller._port)

    if valve_side == "left":
        valve = mc.left_valve
    elif valve_side == "right":
        valve = mc.right_valve
    else:
        console.echo("Invalid valve side specified.", level=LogLevel.ERROR)
        return

    try:
        timer = PrecisionTimer("s")
        data_logger.start()
        mc.start()
        mc.connect_to_smh()
        console.echo(f"Open {valve_side} valve. Press 'q' to close.", level=LogLevel.SUCCESS)
        valve.toggle(state=True)
        timer.delay(duration, block=True)

    finally:
        valve.toggle(state=False)
        mc.disconnect_to_smh()
        mc.stop()
        console.echo("Test: ended.", level=LogLevel.SUCCESS)
        data_logger.stop()


def deliver_test(valve_side, volume=np.float64(30)) -> None:
    """Delivers a specified volume (default 30uL) of fluid through the specified valve to test dispensing."""
    data_logger = DataLogger(output_directory=output_dir, instance_name="deliver_test")
    mc = AMCInterface(data_logger=data_logger)
    console.echo(mc._controller._port)

    if valve_side == "left":
        valve = mc.left_valve
    elif valve_side == "right":
        valve = mc.right_valve
    else:
        console.echo("Invalid valve side specified.", level=LogLevel.ERROR)
        return

    try:
        data_logger.start()
        mc.start()
        mc.connect_to_smh()
        valve.dispense_volume(volume=volume)
    finally:
        mc.disconnect_to_smh()
        mc.stop()
        console.echo("Test: ended.", level=LogLevel.SUCCESS)
        data_logger.stop()


if __name__ == "__main__":
    if not console.enabled:
        console.enable()

    _CALIBRATION_PULSE_DURATION = np.uint32(15000)  # microseconds
    with tempfile.TemporaryDirectory(delete=False) as temp_dir_path:
        output_dir = Path(temp_dir_path).joinpath("test_output")

    calibrate_valve(valve_side='left')
    #calibrate_valve(valve_side="right")
    # deliver_test(valve_side='left')
    # deliver_test(valve_side='right')
    # toggle_valve(valve_side='right')
    # toggle_valve(valve_side='left')
