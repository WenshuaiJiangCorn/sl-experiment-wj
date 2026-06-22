from pathlib import Path
import threading

import numpy as np
from visualizers import BehaviorVisualizer
from ataraxis_time import PrecisionTimer
from binding_classes import VideoSystems
from data_processing import process_microcontroller_log
from microcontroller import AMCInterface
from ataraxis_base_utilities import LogLevel, console, ensure_directory_exists
from ataraxis_data_structures import DataLogger, assemble_log_archives


class ExperimentControl:
    """Thread-safe signals shared between the GUI and the experiment loop."""

    def __init__(self):
        self.stop = threading.Event()
        self.dispense_left = threading.Event()
        self.dispense_right = threading.Event()
        self.proceed = threading.Event()  # skip acclimation (replaces 'p')


def run_experiment(output_dir: Path, reward_volume: np.float64, control: ExperimentControl) -> None:
    """Initializes, manages, and terminates an experiment runtime cycle in the Yapici lab.
    The experiment starts with a 8 minutes acclimation period, experimenter should attach fiber
    and let the animal acclimates to the experiment arena during this period.

    The GUI drives reward dispensing and stopping via ExperimentControl signals.
    """
    if not console.enabled:
        console.enable()

    data_logger = DataLogger(output_directory=output_dir, instance_name="linear_track")
    mc = AMCInterface(data_logger=data_logger)
    vs = VideoSystems(data_logger=data_logger, output_directory=output_dir)
    visualizer = BehaviorVisualizer()

    try:
        data_logger.start()
        vs.start()

        mc.start()
        mc.connect_to_smh()
        visualizer.open()

        mc.left_lick_sensor.check_state()
        mc.right_lick_sensor.check_state()
        mc.analog_input.check_state()

        acclimation_timer = PrecisionTimer("s")
        cycle_timer = PrecisionTimer("ms")
        valve_delay_timer = PrecisionTimer("ms")

        valve_left_active = False
        valve_right_active = False
        valve_delay_active = False
        valve_triggered_side = None

        prev_lick_left = mc.left_lick_sensor.lick_count
        prev_lick_right = mc.right_lick_sensor.lick_count

        acclimation_timer.reset()
        valve_delay_timer.reset()

        _once = False

        console.echo("Experiment starts. Click Stop in the GUI to end.", level=LogLevel.SUCCESS)
        console.echo("8 minutes of pre-task acclimation period starts. Click Proceed in the GUI to skip.")

        while True:
            cycle_timer.delay(delay=20)

            visualizer.update()

            lick_left = mc.left_lick_sensor.lick_count
            lick_right = mc.right_lick_sensor.lick_count

            if not _once:
                if acclimation_timer.elapsed >= 480 or control.proceed.is_set():
                    valve_left_active = True
                    valve_right_active = True
                    _once = True
                    console.echo("Task opens.", level=LogLevel.SUCCESS)

            if valve_delay_active and valve_delay_timer.elapsed >= 1000:
                if valve_triggered_side == "left":
                    valve_right_active = True
                elif valve_triggered_side == "right":
                    valve_left_active = True
                valve_delay_active = False
                valve_triggered_side = None

            if control.dispense_left.is_set():
                control.dispense_left.clear()
                mc.left_valve.dispense_volume(volume=reward_volume)
                visualizer.add_left_valve_event()
                valve_triggered_side = "left"
                valve_left_active = False
                valve_delay_active = True
                valve_delay_timer.reset()

            if control.dispense_right.is_set():
                control.dispense_right.clear()
                mc.right_valve.dispense_volume(volume=reward_volume)
                visualizer.add_right_valve_event()
                valve_triggered_side = "right"
                valve_right_active = False
                valve_delay_active = True
                valve_delay_timer.reset()

            if lick_left > prev_lick_left:
                visualizer.add_left_lick_event()
                if valve_left_active:
                    mc.left_valve.dispense_volume(volume=reward_volume)
                    visualizer.add_left_valve_event()
                    valve_left_active = False
                    valve_right_active = False
                    valve_delay_active = True
                    valve_triggered_side = "left"
                    valve_delay_timer.reset()

            if lick_right > prev_lick_right:
                visualizer.add_right_lick_event()
                if valve_right_active:
                    mc.right_valve.dispense_volume(volume=reward_volume)
                    visualizer.add_right_valve_event()
                    valve_left_active = False
                    valve_right_active = False
                    valve_delay_active = True
                    valve_triggered_side = "right"
                    valve_delay_timer.reset()

            prev_lick_left, prev_lick_right = lick_left, lick_right

            if control.stop.is_set():
                console.echo("Stopping the experiment.")
                mc.left_lick_sensor.reset_command_queue()
                mc.right_lick_sensor.reset_command_queue()
                mc.analog_input.reset_command_queue()
                break

    finally:
        total_volume = mc.dispensed_volume()

        vs.stop()
        mc.disconnect_to_smh()
        mc.stop()
        visualizer.close()
        data_logger.stop()
        console.echo("Experiment: ended.", level=LogLevel.SUCCESS)
        console.echo(f"Total dispensed volume: {total_volume:.2f} uL", level=LogLevel.SUCCESS)

        assemble_log_archives(
            log_directory=data_logger.output_directory,
            remove_sources=True,
            memory_mapping=False,
            verbose=True,
            verify_integrity=False,
        )

        processed_dir = output_dir.joinpath("processed")
        ensure_directory_exists(processed_dir)

        process_microcontroller_log(data_logger=data_logger, microcontroller=mc, output_directory=processed_dir)
        vs.extract_video_time_stamps(output_directory=processed_dir)
