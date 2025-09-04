# WJ: Run this script to start the experiment
from ataraxis_base_utilities import console
import keyboard
from mesoscope_vr.binding_classes import MicroControllerInterfaces
from ataraxis_data_structures import DataLogger

output_dir = r"test_output"

data_logger = DataLogger(output_directory=output_dir)
mc = MicroControllerInterfaces(data_logger=data_logger)

mc.start()
mc.enable_lick_monitoring()

console.echo("Experiment starts. Press 'q' to quit.")

#At first both valves are available
valve_status_left = True
valve_status_right = True

prev_lick_count_left = mc.lick_count_left()
prev_lick_count_right = mc.lick_count_right()

# Only one valve is available after the first reward, deactivated after use
# Valve is reactivated after the other valve is used 
while True:
    if mc.lick_count_left() > prev_lick_count_left and valve_status_left:
        mc.deliver_reward_left(tone_duration=0)
        valve_status_left = False
        valve_status_right = True
    elif mc.lick_count_right() > prev_lick_count_right and valve_status_right:
        mc.deliver_reward_right(tone_duration=0)
        valve_status_right = False
        valve_status_left = True

    prev_lick_count_left = mc.lick_count_left()
    prev_lick_count_right = mc.lick_count_right()

    if keyboard.is_pressed('q'):  # Check if 'q' key is pressed
        console.echo("Breaking loop due to 'q' key press.")
        break

mc.stop()
console.echo("Experiment ends.")