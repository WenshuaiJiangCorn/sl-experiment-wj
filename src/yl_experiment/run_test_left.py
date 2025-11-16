"""This module provides the executable script used to run test experiments with only left valve and lick in the Yapici lab."""

# WJ: Run this script to start the test
import time
from pathlib import Path

import numpy as np
import keyboard
from visualizers import BehaviorVisualizer
from data_processing import process_microcontroller_log
from microcontroller import AMCInterface
from ataraxis_base_utilities import LogLevel, console, ensure_directory_exists
from ataraxis_data_structures import DataLogger, assemble_log_archives

# Note, prevents the context manager from automatically deleting the temporary directory.
# with tempfile.TemporaryDirectory(delete=False) as temp_dir_path:
# output_dir = Path(temp_dir_path).joinpath("test_output")

output_dir = Path("C:\\Users\\Changwoo\\Dropbox\\Research_projects\\dopamine\\mazes\\linear_track\\lickometer_test").joinpath("test_output")

_REWARD_VOLUME = np.float64(10)  # 5 microliters


def run_test() -> None:
    """Initializes, manages, and terminates a test runtime cycle in the Yapici lab.
    Valve deactivated for 5 seconds after dispensing reward.
    """
    if not console.enabled:
        console.enable()

    data_logger = DataLogger(output_directory=output_dir, instance_name="valve_lick_test")
    mc = AMCInterface(data_logger=data_logger)
    visualizer = BehaviorVisualizer()
    console.echo(mc._controller._port)

    try:
        data_logger.start()  # Has to be done before starting any data-generation processes
        mc.start()
        mc.connect_to_smh()  # Establishes connections to SharedMemoryArray for all modules
        mc.left_lick_sensor.check_state()
        visualizer.open()  # Open the visualizer window
        console.echo("Test: started. Press 'q' to quit.", level=LogLevel.SUCCESS)

        # Initial valve availability
        valve_left_active = True

        prev_lick_left = mc.left_lick_sensor.lick_count
        valve_left_deactivated_time = None  # Track when the left valve was deactivated

        while True:
            visualizer.update()
            lick_left = mc.left_lick_sensor.lick_count

            if lick_left > prev_lick_left:
                visualizer.add_left_lick_event()

                if valve_left_active:
                    mc.left_valve.dispense_volume(volume=_REWARD_VOLUME)
                    valve_left_active = False
                    visualizer.add_left_valve_event()

                    valve_left_deactivated_time = time.time()

            # check if 3 seconds passed since deactivation
            if not valve_left_active and valve_left_deactivated_time is not None:
                if time.time() - valve_left_deactivated_time >= 3:
                    valve_left_active = True
                    valve_left_deactivated_time = None  # reset timer

            prev_lick_left = lick_left

            if keyboard.is_pressed("q"):
                console.echo("Ending the test.")

                # Stops monitoring lick sensors before entering the termination clause
                mc.left_lick_sensor.reset_command_queue()
                break

            time.sleep(0.03)

    finally:
        mc.disconnect_to_smh()  # Disconnects from SharedMemoryArray for all modules
        mc.stop()
        visualizer.close()
        console.echo("Test: ended.", level=LogLevel.SUCCESS)

        data_logger.stop()  # Data logger needs to be stopped last

        # Combines all log entries into a single .npz log file for each source.
        assemble_log_archives(
            log_directory=data_logger.output_directory,
            remove_sources=True,
            memory_mapping=False,
            verbose=True,
            verify_integrity=False,
        )

        processed_dir = output_dir.joinpath("processed")
        ensure_directory_exists(processed_dir)

        # Extracts all logged data as module-specific .feather files. These files can be read via
        # Polars' 'read_ipc' function. Use memory-mapping mode for efficiency.
        process_microcontroller_log(data_logger=data_logger, microcontroller=mc, output_directory=processed_dir)


# Run test
if __name__ == "__main__":
    run_test()
