"""This module provides the executable script used to run test experiments with only left valve in the Yapici lab."""
# WJ: Run this script to start the test
import time
from pathlib import Path
import tempfile

import keyboard
from ataraxis_base_utilities import LogLevel, console
from ataraxis_data_structures import DataLogger

from yl_experiment.data_processing import process_microcontroller_log
from microcontroller import AMCInterface

# Note, prevents the context manager from automatically deleting the temporary directory.
with tempfile.TemporaryDirectory(delete=False) as temp_dir_path:
    output_dir = Path(temp_dir_path).joinpath("test_output")


def run_test() -> None:
    """Initializes, manages, and terminates a test runtime cycle in the Yapici lab. 
    Valve deactivated for 5 seconds after dispensing reward."""

    data_logger = DataLogger(output_directory=output_dir, exist_ok=True)
    mc = AMCInterface(data_logger=data_logger)
    console.echo(mc._controller._port)

    try:
        data_logger.start()  # Has to be done before starting any data-generation processes
        mc.start()  # Starts the microcontroller data acquisition and control processes
        console.echo("Test: started. Press 'q' to quit.", level=LogLevel.SUCCESS)
        start_time = time.time()

        while True:
            elapsed_time = time.time() - start_time

            if elapsed_time % 5 < 0.01:
                mc.left_valve.dispense_volume(volume=5)

            if keyboard.is_pressed("q"):
                console.echo("Breaking the test loop due to the 'q' key press.")
                break

            time.sleep(0.01)

    finally:
        mc.left_valve.toggle(state=False)
        mc.stop()
        console.echo("Test: ended.", level=LogLevel.SUCCESS)

        data_logger.stop()  # Data logger needs to be stopped last

        # # Combines all log entries into a single .npz log file for each source.
        # data_logger.compress_logs(
        #     remove_sources=True, memory_mapping=False, verbose=True, compress=False, verify_integrity=False
        # )

        # # Extracts all logged data as module-specific .feather files. These files can be read via
        # # Polars' 'read_ipc' function. Use memory-mapping mode for efficiency.
        # process_microcontroller_log(
        #     data_logger=data_logger, microcontroller=mc, output_directory=output_dir.joinpath("processed")
        # )


# Run test
if __name__ == "__main__":
    run_test()
