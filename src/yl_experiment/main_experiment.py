from pathlib import Path
from datetime import datetime

import numpy as np
import keyboard
from visualizers import BehaviorVisualizer
from ataraxis_time import PrecisionTimer
from binding_classes import VideoSystems
from data_processing import process_microcontroller_log
from microcontroller import AMCInterface
from ataraxis_base_utilities import LogLevel, console, ensure_directory_exists
from ataraxis_data_structures import DataLogger, assemble_log_archives

_REWARD_VOLUME = np.float64(10)  # 10uL
_EXPERIMENT_DIR = Path(
    "C:\\Users\\yapici\\Dropbox\\Research_projects\\dopamine\\mazes\\linear_track\\0.1M_scrose_reward\\2026Jan_AgRP\\raw_data"
    )


def run_experiment() -> None:
    """Initializes, manages, and terminates an experiment runtime cycle in the Yapici lab.
    The experiment starts with a 8 minutes acclimation period, experimenter should attach fiber
    and let the animal acclimates to the experiment arena during this period.

    Task opens after 8 minutes. Press 'q' to terminate the process.
    """
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

        # Start monitoring lickings and photometry analog input before the task opens
        mc.left_lick_sensor.check_state()
        mc.right_lick_sensor.check_state()
        mc.analog_input.check_state()

        # Initialize the timers
        acclimation_timer = PrecisionTimer("s")
        cycle_timer = PrecisionTimer("ms")

        acclimation_timer.reset()

        # During acclimation period, the valves are closed
        valve_left_active = False
        valve_right_active = False

        prev_lick_left = mc.left_lick_sensor.lick_count
        prev_lick_right = mc.right_lick_sensor.lick_count

        # Before experiment tasak starts, wait for 8 minutes for experimenter to attach fiber to
        # the mouse and acclimate the animal to the arena
        # Cut off this period in the data processing if necessary
        _once = False

        console.echo("Experiment starts. Press 'q' to stop the experiment.", level=LogLevel.SUCCESS)
        console.echo("8 minutes of pre-task acclimation period starts. Press 'p' to manually proceed")
        while True:
            cycle_timer.delay(delay=20)  # 20ms delay to prevent CPU overuse

            visualizer.update()

            lick_left = mc.left_lick_sensor.lick_count
            lick_right = mc.right_lick_sensor.lick_count

            # Check if acclimation period has passed, or 'p' has been pressed to proceed
            # If it has, activate valves
            if not _once:
                if acclimation_timer.elapsed >= 480 or keyboard.is_pressed("p"):
                    valve_left_active = True
                    valve_right_active = True
                    _once = True
                    console.echo("Task opens.", level=LogLevel.SUCCESS)

            if keyboard.is_pressed("e"):
                mc.left_valve.dispense_volume(volume=_REWARD_VOLUME)
                visualizer.add_left_valve_event()

            if keyboard.is_pressed("r"):
                mc.right_valve.dispense_volume(volume=_REWARD_VOLUME)
                visualizer.add_right_valve_event()

            if lick_left > prev_lick_left:
                visualizer.add_left_lick_event()
                if valve_left_active:
                    mc.left_valve.dispense_volume(volume=_REWARD_VOLUME)
                    visualizer.add_left_valve_event()

                    valve_left_active = False
                    valve_right_active = True

            if lick_right > prev_lick_right:
                visualizer.add_right_lick_event()
                if valve_right_active:
                    mc.right_valve.dispense_volume(volume=_REWARD_VOLUME)
                    visualizer.add_right_valve_event()

                    valve_left_active = True
                    valve_right_active = False

            prev_lick_left, prev_lick_right = lick_left, lick_right

            if keyboard.is_pressed("q"):
                console.echo("Stopping the experiment due to the 'q' key press.")

                # Stops monitoring lick sensors before entering the termination clause
                mc.left_lick_sensor.reset_command_queue()
                mc.right_lick_sensor.reset_command_queue()
                mc.analog_input.reset_command_queue()
                break

    finally:
        total_volume = mc.dispensed_volume()  # Store total dispensed volume before stopping the microcontroller
        console.echo(f"Total dispensed volume: {total_volume:.2f} uL", level=LogLevel.SUCCESS)

        vs.stop()
        mc.disconnect_to_smh()  # Disconnects from SharedMemoryArray for all modules
        mc.stop()
        visualizer.close()
        data_logger.stop()  # Data logger needs to be stopped last
        console.echo("Experiment: ended.", level=LogLevel.SUCCESS)

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


if __name__ == "__main__":
    # Configure the mouse and experiment info
    mouse = input("Input experiment mouse ID (e.g., DATM1): ")
    exp_day = input("Input experiment day (e.g., day_1): ")

    date = datetime.now().strftime("%Y%m%d")
    exp_day = f"{exp_day}_{date}"

    # Create output directory
    output_dir = _EXPERIMENT_DIR / mouse / exp_day
    ensure_directory_exists(output_dir)

    # Run experiment
    run_experiment()