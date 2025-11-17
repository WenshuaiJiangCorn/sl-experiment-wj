from pathlib import Path

import numpy as np
import keyboard
from datetime import datetime

from binding_classes import VideoSystems
from visualizers import BehaviorVisualizer
from ataraxis_time import PrecisionTimer
from data_processing import process_microcontroller_log
from microcontroller import AMCInterface

from ataraxis_base_utilities import LogLevel, console, ensure_directory_exists
from ataraxis_data_structures import DataLogger, assemble_log_archives


mouse = input("Input experiment mouse ID. (e.g., DATM1)")
exp_day = input("Input experiment day (e.g., day_1): ")

date = datetime.now().strftime("%Y%m%d")
exp_day = f"{exp_day}_{date}"

output_dir = Path("C:\\Users\\Changwoo\\Dropbox\\Research_projects\\dopamine\\mazes\\linear_track\\water_reward") / exp_day / mouse
ensure_directory_exists(output_dir)

_REWARD_VOLUME = np.float64(10)  # 10uL


def run_experiment() -> None:
    """Initializes, manages, and terminates an experiment runtime cycle in the Yapici lab."""
    if not console.enabled:
        console.enable()

    data_logger = DataLogger(output_directory=output_dir, instance_name="linear_track")
    mc = AMCInterface(data_logger=data_logger)
    vs = VideoSystems(data_logger=data_logger, output_directory=output_dir)
    visualizer = BehaviorVisualizer()

    try:
        data_logger.start()  # Has to be done before starting any data-generation processes
        vs.start()

        # Start the microcontroller, execute reward delivery logic
        mc.start()
        mc.connect_to_smh()  # Establishes connections to SharedMemoryArray for all modules
        visualizer.open()  # Open the visualizer window

        mc.left_lick_sensor.check_state()
        mc.right_lick_sensor.check_state()
        mc.analog_input.check_state()

        console.echo("Experiment: started. Press 'q' to stop.", level=LogLevel.SUCCESS)

        # Initial valve availability
        valve_left_active = True
        valve_right_active = True

        prev_lick_left = mc.left_lick_sensor.lick_count
        prev_lick_right = mc.right_lick_sensor.lick_count

        while True:
            visualizer.update()
            lick_left = mc.left_lick_sensor.lick_count
            lick_right = mc.right_lick_sensor.lick_count

            if lick_left > prev_lick_left:
                visualizer.add_left_lick_event()
                if valve_left_active:
                    mc.left_valve.dispense_volume(volume=_REWARD_VOLUME)
                    valve_left_active = False
                    valve_right_active = True
                    visualizer.add_left_valve_event()

            if lick_right > prev_lick_right:
                visualizer.add_right_lick_event()
                if valve_right_active:
                    mc.right_valve.dispense_volume(volume=_REWARD_VOLUME)
                    valve_left_active = True
                    valve_right_active = False
                    visualizer.add_right_valve_event()

            prev_lick_left, prev_lick_right = lick_left, lick_right
            
            if keyboard.is_pressed("q"):
                console.echo("Breaking the experiment loop due to the 'q' key press.")

                # Stops monitoring lick sensors before entering the termination clause
                mc.left_lick_sensor.reset_command_queue()
                mc.right_lick_sensor.reset_command_queue()
                mc.analog_input.reset_command_queue()
                break

            timer = PrecisionTimer("ms")
            timer.delay(delay=20, block=False)  # 10ms delay to prevent CPU overuse

    finally:
        vs.stop()
        total_volume = mc.dispensed_volume() # Store total dispensed volume before stopping the microcontroller
        mc.disconnect_to_smh()  # Disconnects from SharedMemoryArray for all modules
        mc.stop()
        visualizer.close()
        console.echo("Experiment: ended.", level=LogLevel.SUCCESS)

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

        # Extract and save video frame timestamps
        vs.extract_video_time_stamps(output_directory=processed_dir)

        # Summarize total dispensed volume
        console.echo(f"Total dispensed volume: {total_volume:.2f} uL", level=LogLevel.SUCCESS)

if __name__ == "__main__":
    run_experiment()
