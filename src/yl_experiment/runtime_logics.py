import numpy as np
from binding_classes import LinearTrackFunctions

if __name__ == "__main__":
    _CALIBRATION_PULSE_DURATION = np.uint32(15000)

    exp = LinearTrackFunctions()

    #exp.open_valve(valve_side='left', duration=1)
    exp.open_valve(valve_side='right', duration=1)
    
    #exp.calibrate_valve('left', _CALIBRATION_PULSE_DURATION)
    #exp.calibrate_valve("right", _CALIBRATION_PULSE_DURATION)

    #exp.first_day_training()
    #exp.second_day_training()


    #exp.delivery_test('left')
    #exp.delivery_test('right')

    #exp.test_noise()