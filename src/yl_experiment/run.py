"""This module provides the executable script used to run experiments in the Yapici lab."""

# WJ: Run this script to start the experiment
import time
from pathlib import Path
import tempfile

import numpy as np
import keyboard
from ataraxis_base_utilities import LogLevel, console
from ataraxis_data_structures import DataLogger

from .data_processing import process_microcontroller_log
from .microcontroller import AMCInterface

# Note, prevents the context manager from automatically deleting the temporary directory.
with tempfile.TemporaryDirectory(delete=False) as temp_dir_path:
    output_dir = Path(temp_dir_path).joinpath("test_output")


_REWARD_VOLUME = np.float64(0.5)  # 500 microliters


def run_experiment() -> None:
    """Initializes, manages, and terminates an experiment runtime cycle in the Yapici lab."""
    data_logger = DataLogger(output_directory=output_dir)
    mc = AMCInterface(data_logger=data_logger)

    try:
        data_logger.start()  # Has to be done before starting any data-generation processes
        mc.start()
        mc.left_lick_sensor.check_state()
        mc.right_lick_sensor.check_state()
        console.echo("Experiment: started. Press 'q' to quit.", level=LogLevel.SUCCESS)

        # Initial valve availability
        valve_left_active = True
        valve_right_active = True

        prev_lick_left = mc.left_lick_sensor.lick_count
        prev_lick_right = mc.right_lick_sensor.lick_count

        while True:
            lick_left = mc.left_lick_sensor.lick_count
            lick_right = mc.right_lick_sensor.lick_count

            if lick_left > prev_lick_left and valve_left_active:
                mc.left_valve.dispense_volume(volume=_REWARD_VOLUME)
                valve_left_active = False
                valve_right_active = True

            elif lick_right > prev_lick_right and valve_right_active:
                mc.right_valve.dispense_volume(volume=_REWARD_VOLUME)
                valve_right_active = False
                valve_left_active = True

            prev_lick_left, prev_lick_right = lick_left, lick_right

            if keyboard.is_pressed("q"):
                console.echo("Breaking the experiment loop due to the 'q' key press.")

                # Stops monitoring lick sensors before entering the termination clause
                mc.left_lick_sensor.reset_command_queue()
                mc.right_lick_sensor.reset_command_queue()
                break

            time.sleep(0.01)

    finally:
        mc.stop()
        console.echo("Experiment: ended.", level=LogLevel.SUCCESS)

        data_logger.stop()  # Data logger needs to be stopped last

        # Combines all log entries into a single .npz log file for each source.
        data_logger.compress_logs(
            remove_sources=True, memory_mapping=False, verbose=True, compress=False, verify_integrity=False
        )

        # Extracts all logged data as module-specific .feather files. These files can be read via
        # Polars' 'read_ipc' function. Use memory-mapping mode for efficiency.
        process_microcontroller_log(
            data_logger=data_logger, microcontroller=mc, output_directory=output_dir.joinpath("processed")
        )


# Run experiment
if __name__ == "__main__":
    run_experiment()
