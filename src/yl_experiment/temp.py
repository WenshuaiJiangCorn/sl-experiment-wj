
import time
from pathlib import Path
import tempfile

import numpy as np
from ataraxis_base_utilities import LogLevel, console
from ataraxis_data_structures import DataLogger

from microcontroller import AMCInterface


if __name__ == "__main__":
    if not console.enabled:
        console.enable()

    with tempfile.TemporaryDirectory(delete=False) as temp_dir_path:
        output_dir = Path(temp_dir_path).joinpath("test_output")

    data_logger = DataLogger(output_directory=output_dir, instance_name="calibration_test")
    mc = AMCInterface(data_logger=data_logger)
    console.echo(mc._controller._port)

    lick_sensor = mc.left_lick_sensor

    try:
        data_logger.start()  # Has to be done before starting any data-generation processes
        mc.start()

        console.echo('Starts')
        c = lick_sensor.check_state()
        print(c)


    finally:
        mc.stop()
        console.echo("Test: ended.", level=LogLevel.SUCCESS)
        data_logger.stop()