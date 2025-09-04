# WJ: Run this script to start the experiment
from ataraxis_base_utilities import console
import keyboard
from mesoscope_vr.binding_classes import MicroControllerInterfaces
from ataraxis_data_structures import DataLogger

output_dir = r"test_output"
valve_ids = (1, 2) # Has to match the valve IDs in the microcontroller firmware
lick_ids = (1, 2) # Has to match the lick IDs in the microcontroller firmware
valve_status = {valve_id: True for valve_id in valve_ids}

data_logger = DataLogger(output_directory=output_dir)
mc = MicroControllerInterfaces(valve_ids=valve_ids, lick_ids=lick_ids, data_logger=data_logger)

mc.start()
mc.enable_lick_monitoring()

console.echo("Experiment starts. Press 'q' to quit.")

# Only one valve is available after the first reward, deactivated after use
# Valve is reactivated after the other valve is used 
prev_lick_count_1 = mc.lick_count(lick_id=lick_ids[0])
prev_lick_count_2 = mc.lick_count(lick_id=lick_ids[1])

while True:
    if mc.lick_count(lick_id=lick_ids[0]) > prev_lick_count_1 and valve_status[valve_ids[0]]:
        mc.deliver_reward(valve_id=valve_ids[0], tone_duration=0)
        valve_status[valve_ids[0]] = False
        valve_status[valve_ids[1]] = True
    elif mc.lick_count(lick_id=lick_ids[1]) > prev_lick_count_2 and valve_status[valve_ids[1]]:
        mc.deliver_reward(valve_id=valve_ids[1], tone_duration=0)
        valve_status[valve_ids[1]] = False
        valve_status[valve_ids[0]] = True

    prev_lick_count_1 = mc.lick_count(lick_id=lick_ids[0])
    prev_lick_count_2 = mc.lick_count(lick_id=lick_ids[1])

    if keyboard.is_pressed('q'):  # Check if 'q' key is pressed
        console.echo("Breaking loop due to 'q' key press.")
        break

mc.stop()
console.echo("Experiment ends.")