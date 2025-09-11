"""This module provides the executable script used to run test experiments with only left valve and lick in the Yapici lab."""

# WJ: Run this script to start the test
import time
from pathlib import Path
import tempfile

import numpy as np
import keyboard
from ataraxis_base_utilities import LogLevel, console
from ataraxis_data_structures import DataLogger

from .data_processing import process_microcontroller_log
from .microcontroller import AMCInterface, LickInterface

# Note, prevents the context manager from automatically deleting the temporary directory.
with tempfile.TemporaryDirectory(delete=False) as temp_dir_path:
    output_dir = Path(temp_dir_path).joinpath("test_output")


_REWARD_VOLUME = np.float64(0.5)  # 500 microliters


def run_test() -> None:
    """Initializes, manages, and terminates a test runtime cycle in the Yapici lab. 
    Valve deactivated for 5 seconds after dispensing reward."""

    data_logger = DataLogger(output_directory=output_dir)
    mc = AMCInterface(data_logger=data_logger)

    # Enable debug mode for left lick sensor to read the voltage 
    mc.left_lick_sensor = LickInterface(
            module_id=np.uint8(1),
            debug=True,
        )
    try:
        data_logger.start()  # Has to be done before starting any data-generation processes
        mc.start()
        mc.left_lick_sensor.check_state()
        console.echo("Test: started. Press 'q' to quit.", level=LogLevel.SUCCESS)

        # Initial valve availability
        valve_left_active = True

        prev_lick_left = mc.left_lick_sensor.lick_count
        valve_left_deactivated_time = None  # Track when the left valve was deactivated

        while True:
            lick_left = mc.left_lick_sensor.lick_count

            if lick_left > prev_lick_left and valve_left_active:
                mc.left_valve.dispense_volume(volume=_REWARD_VOLUME)
                valve_left_active = False
                valve_left_deactivated_time = time.time()

            # check if 5 seconds passed since deactivation
            if not valve_left_active and valve_left_deactivated_time is not None:
                if time.time() - valve_left_deactivated_time >= 5:
                    valve_left_active = True
                    valve_left_deactivated_time = None  # reset timer

            prev_lick_left = lick_left

            if keyboard.is_pressed("q"):
                console.echo("Breaking the test loop due to the 'q' key press.")

                # Stops monitoring lick sensors before entering the termination clause
                mc.left_lick_sensor.reset_command_queue()
                break

            time.sleep(0.01)

    finally:
        mc.stop()
        console.echo("Test: ended.", level=LogLevel.SUCCESS)

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


# Run test
if __name__ == "__main__":
    run_test()
