"""This module provides the main VR class that abstracts working with Sun lab's mesoscope-VR system"""

from ataraxis_data_structures import DataLogger
from ataraxis_communication_interface import MicroControllerInterface
from .module_interfaces import (
    TTLInterface,
    EncoderInterface,
    BreakInterface,
    ValveInterface,
    LickInterface,
    TorqueInterface,
)
from pathlib import Path
import numpy as np


class VR:
    def __init__(self, output_directory: Path) -> None:

        self._started: bool = False

        # Initializes the microcontroller data logger. This datalogger works exclusively with microcontroller-generated
        # data.
        self.amc_logger: DataLogger = DataLogger(output_directory=output_directory, instance_name="vr", sleep_timer=0)

        # Actor AMC controls the hardware that needs to be triggered by PC at irregular intervals. Most of such hardware
        # is designed to produce some form of an output: deliver water reward, engage wheel breaks, issue a TTL trigger,
        # etc.
        self._actor_interfaces: tuple[TTLInterface, BreakInterface, ValveInterface] = (
            TTLInterface(),  # Mesoscope trigger
            BreakInterface(
                minimum_break_strength=43.2047,  # 0.6 in oz
                maximum_break_strength=1152.1246,  # 16 in oz
                object_diameter=15.0333,  # 15 cm diameter + 0.0333 to account for the wrap
            ),  # Wheel break
            ValveInterface(valve_calibration_data=((10, 10), (20, 20))),  # Reward solenoid valve
        )
        self._actor: MicroControllerInterface = MicroControllerInterface(
            controller_id=np.uint8(101),
            microcontroller_serial_buffer_size=8192,
            microcontroller_usb_port="/dev/ttyACM1",
            data_logger=self.amc_logger,
            module_interfaces=self._actor_interfaces,
        )

        # Sensor AMC controls the hardware that collects data at regular intervals. This includes lick sensors, torque
        # sensors, and input TTL recorders. Critically, all managed hardware does not rely on hardware interrupt logic
        # to maintain the necessary precision.
        self._sensor_interfaces: tuple[TTLInterface, LickInterface, TorqueInterface] = (
            TTLInterface(),  # Mesoscope frame recorder
            LickInterface(lick_threshold=2000),  # Main lick sensor
            TorqueInterface(
                baseline_voltage=2046,  # ~1.65 V
                maximum_voltage=4095,  # ~3.3 V
                sensor_capacity=720.0779,  # 10 in oz
                object_diameter=15.0333,  # 15 cm diameter + 0.0333 to account for the wrap
            ),  # Wheel torque sensor
        )
        self._sensor: MicroControllerInterface = MicroControllerInterface(
            controller_id=np.uint8(152),
            microcontroller_serial_buffer_size=8192,
            microcontroller_usb_port="/dev/ttyACM0",
            data_logger=self.amc_logger,
            module_interfaces=self._sensor_interfaces,
        )

        # Encoder AMC is specifically designed to interface with a rotary encoder connected to the running wheel. The
        # encoder uses hardware interrupt logic to maintain high precision and, therefore, it is isolated to a
        # separate microcontroller to ensure adequate throughput.
        self._encoder_interfaces: tuple[EncoderInterface] = (
            EncoderInterface(encoder_ppr=8192, object_diameter=15.0333, cm_per_unity_unit=10.0),
        )
        self._encoder: MicroControllerInterface = MicroControllerInterface(
            controller_id=np.uint8(203),
            microcontroller_serial_buffer_size=8192,
            microcontroller_usb_port="/dev/ttyACM2",
            data_logger=self.amc_logger,
            module_interfaces=self._encoder_interfaces,
        )

    def __del__(self):
        self.stop()

    def start(self):
        if not self._started:
            self.amc_logger.start()
            self._actor.start()
            self._actor.unlock_controller()  # Only Actor outputs data, so no need to unlock other controllers.
            self._sensor.start()
            self._encoder.start()

            # Engages the modules that need to stay on during the entire experiment:

            # The mesoscope acquires frames at ~10 Hz and sends triggers with the on-phase duration of ~50 ms, so a
            # 100 Hz polling frequency should be enough to detect all triggers.
            # noinspection PyTypeChecker
            self._sensor.send_message(self._sensor_interfaces[0].check_state(repetition_delay=np.uint32(10000)))

            # Starts monitoring licks. Uses 100 Hz polling frequency, since mice are expected to lick at ~10 Hz rate.
            # noinspection PyTypeChecker
            self._sensor.send_message(self._sensor_interfaces[1].check_state(repetition_delay=np.uint32(10000)))

            # Water Valve receives triggers directly from unity, so we do not need to manipulate the valve state
            # manually. Mesoscope is triggered via a dedicated method.

            self._started = True

    def stop(self):
        if self._started:
            self._started = False
            self.amc_logger.stop()
            self._actor.stop()
            self._sensor.stop()
            self._encoder.stop()

    def vr_rest(self):
        # Engages the break to prevent the mouse from moving the wheel
        # noinspection PyTypeChecker
        self._actor.send_message(self._actor_interfaces[1].toggle(True))

        # Initiates torque monitoring at 100 Hz. The torque can only be accurately measured when the wheel is locked,
        # as it requires a resistance force to trigger the sensor. Since we downsample all data to the mesoscope
        # acquisition rate of ~10 Hz, and do not use torque data in real time, the sampling rate is seto to a 100 Hz.
        # noinspection PyTypeChecker
        self._sensor.send_message(self._sensor_interfaces[2].check_state(repetition_delay=np.uint32(10000)))

        # Temporarily suspends encoder monitoring. Since the wheel is locked, the mouse should not be able to produce
        # meaningful motion data.
        # noinspection PyTypeChecker
        self._encoder.send_message(self._encoder_interfaces[0].dequeue_command)

    def vr_run(self):
        # Initializes encoder monitoring at 1 kHz rate. The encoder aggregates wheel data at native speeds; this rate
        # only determines how often the aggregated data is sent to PC and Unity.
        # noinspection PyTypeChecker
        self._encoder.send_message(self._encoder_interfaces[0].check_state(repetition_delay=np.uint32(1000)))

        # Disables torque monitoring. To accurately measure torque, the sensor requires a resistance force provided by
        # the break. During running, measuring torque is not very reliable and adds little in addition to the
        # encoder.
        # noinspection PyTypeChecker
        self._sensor.send_message(self._sensor_interfaces[2].dequeue_command)

        # Disengages the break to allow the mouse to move the wheel
        # noinspection PyTypeChecker
        self._actor.send_message(self._actor_interfaces[1].toggle(False))

    def mesoscope_on(self):

        # Toggles the mesoscope acquisition trigger to continuously deliver a HIGH signal. The mesoscope will
        # continuously acquire frames as long as the trigger is HIGH.
        # noinspection PyTypeChecker
        self._actor.send_message(self._actor_interfaces[0].toggle(True))

    def mesoscope_off(self):

        # Toggles the mesoscope acquisition trigger to continuously deliver a LOW signal. When the trigger is LOW,
        # the mesoscope will not acquire frames.
        # noinspection PyTypeChecker
        self._actor.send_message(self._actor_interfaces[0].toggle(False))
