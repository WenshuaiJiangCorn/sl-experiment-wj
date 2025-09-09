# WJ: Run this script to start the experiment
import time

import keyboard
from ataraxis_base_utilities import console
from ataraxis_data_structures import DataLogger

from src.sl_experiment.binding_classes import MicroControllerInterfaces

output_dir = r"test_output"


def run_experiment():
    data_logger = DataLogger(output_directory=output_dir)
    mc = MicroControllerInterfaces(data_logger=data_logger)

    try:
        mc.start()
        mc.enable_lick_monitoring()
        console.echo("Experiment starts. Press 'q' to quit.")

        # Initial valve availability
        valve_left_active = True
        valve_right_active = True

        prev_lick_left = mc.lick_count_left()
        prev_lick_right = mc.lick_count_right()

        while True:
            lick_left = mc.lick_count_left()
            lick_right = mc.lick_count_right()

            if lick_left > prev_lick_left and valve_left_active:
                mc.deliver_reward_left(tone_duration=0)
                valve_left_active = False
                valve_right_active = True

            elif lick_right > prev_lick_right and valve_right_active:
                mc.deliver_reward_right(tone_duration=0)
                valve_right_active = False
                valve_left_active = True

            prev_lick_left, prev_lick_right = lick_left, lick_right

            if keyboard.is_pressed("q"):
                console.echo("Breaking loop due to 'q' key press.")
                break

            time.sleep(0.01)

    finally:
        mc.stop()
        console.echo("Experiment ends.")


# Run experiment
if __name__ == "__main__":
    run_experiment()
