"""This module provides ModuleInterface implementations for the hardware used by the Sun lab VR-mesoscope."""

from json import dumps
import math
from multiprocessing import Queue as MPQueue

import numpy as np
from numpy.typing import NDArray
from ataraxis_base_utilities import console
from numpy.polynomial.polynomial import polyfit

from ataraxis_communication_interface import (
    ModuleData,
    ModuleState,
    ModuleParameters,
    UnityCommunication,
    OneOffModuleCommand,
    RepeatedModuleCommand,
    ModuleInterface
)


class EncoderInterface(ModuleInterface):
    """Interfaces with EncoderModule instances running on Ataraxis MicroControllers.

    EncoderModule allows interfacing with quadrature encoders used to monitor the direction and magnitude of connected
    object's rotation. To achieve the highest resolution, the module relies on hardware interrupt pins to detect and
    handle the pulses sent by the two encoder channels.

    Notes:
        This interface automatically sends CW and CCW motion data to Unity via 'LinearTreadmill/Data' MQTT topic.

        The default initial encoder readout is 0 (no CW or CCW motion). The class instance is zeroed at communication
        initialization, and it is safe to assume the displacement readout is 0 until the encoder sends the first
        code 51 or 52 data message.

    Args:
        encoder_ppr: The resolution of the managed quadrature encoder, in Pulses Per Revolution (PPR). This is the
            number of quadrature pulses the encoder emits per full 360-degree rotation. If this number is not known,
            provide a placeholder value and use get_ppr() command to estimate the PPR using the index channel of the
            encoder.
        object_diameter: The diameter of the rotating object connected to the encoder, in centimeters. This is used to
            convert encoder pulses into rotated distance in cm.
        cm_per_unity_unit: The conversion factor to translate the distance traveled by the edge of the circular object
             into Unity units. This value works together with object_diameter and encoder_ppr to translate raw
             encoder pulses received from the microcontroller into Unity-compatible units.

    Attributes:
        _motion_topic: Stores the MQTT motion topic.
        _ppr: Stores the resolution of the managed quadrature encoder.
        _object_diameter: Stores the diameter of the object connected to the encoder.
        _cm_per_unity_unit: Stores the conversion factor that translates centimeters into Unity units.
        _unity_unit_per_pulse: Stores the conversion factor to translate encoder pulses into Unity units.
    """

    def __init__(
            self,
            encoder_ppr: int = 8192,
            object_diameter: float = 15.0333,  # 0333 is to account for the wheel wrap
            cm_per_unity_unit: float = 10.0,
    ) -> None:
        data_codes = {np.uint8(51), np.uint8(52), np.uint8(53)}  # kRotatedCCW, kRotatedCW, kPPR

        super().__init__(
            module_type=np.uint8(2),
            module_id=np.uint8(1),
            mqtt_communication=True,
            data_codes=data_codes,
            unity_command_topics=None,
            error_codes=None
        )

        # Saves additional data to class attributes.
        self._motion_topic = "LinearTreadmill/Data"  # Hardcoded output topic
        self._ppr = encoder_ppr
        self._object_diameter = object_diameter
        self._cm_per_unity_unit = cm_per_unity_unit

        # Computes the conversion factor to translate encoder pulses into unity units. Rounds to 12 decimal places for
        # consistency and to ensure repeatability.
        self._unity_unit_per_pulse = np.round(
            a=np.float64((math.pi * object_diameter) / (encoder_ppr * cm_per_unity_unit)),
            decimals=12,
        )

    def process_received_data(
            self,
            message: ModuleState | ModuleData,
            unity_communication: UnityCommunication,
            mp_queue: MPQueue,  # type: ignore
    ) -> None:

        # If the incoming message is the PPR report, sends the data to the output queue
        if message.event == 53:
            topic = "encoder ppr"
            ppr = message.data_object
            mp_queue.put((topic, ppr))

        # Otherwise, the message necessarily has to be reporting rotation into CCW or CW direction
        # (event code 51 or 52).

        # The rotation direction is encoded via the message event code. CW rotation (code 51) is interpreted as negative
        # and CCW as positive.
        sign = 1 if message.event == np.uint8(51) else -1

        # Translates the absolute motion into the CW / CCW vector and converts from raw pulse count to Unity units
        # using the precomputed conversion factor. Uses float64 and rounds to 12 decimal places for consistency and
        # precision
        signed_motion = np.round(
            a=np.float64(message.data_object) * self._unity_unit_per_pulse * sign,
            decimals=12,
        )

        # Encodes the motion data into the format expected by the GIMBL Unity module and serializes it into a
        # byte-string.
        json_string = dumps(obj={"movement": signed_motion})
        byte_array = json_string.encode("utf-8")

        # Publishes the motion to the appropriate MQTT topic.
        unity_communication.send_data(topic=self._motion_topic, payload=byte_array)

    def parse_unity_command(self, topic: str, payload: bytes | bytearray) -> None:
        """Not used."""
        return

    def set_parameters(
            self,
            report_ccw: np.bool | bool = np.bool(True),
            report_cw: np.bool | bool = np.bool(True),
            delta_threshold: np.uint32 | int = np.uint32(10),
    ) -> ModuleParameters:
        """Changes the PC-addressable runtime parameters of the EncoderModule instance.

        Use this method to package and apply new PC-addressable parameters to the EncoderModule instance managed by
        this Interface class.

        Args:
            report_ccw: Determines whether to report rotation in the CCW (positive) direction.
            report_cw: Determines whether to report rotation in the CW (negative) direction.
            delta_threshold: The minimum number of pulses required for the motion to be reported. Depending on encoder
                resolution, this allows setting the 'minimum rotation distance' threshold for reporting. Note, if the
                change is 0 (the encoder readout did not change), it will not be reported, regardless of the
                value of this parameter. Sub-threshold motion will be aggregated (summed) across readouts until a
                significant overall change in position is reached to justify reporting it to the PC.

        Returns:
            The ModuleParameters message that can be sent to the microcontroller via the send_message() method of
            the MicroControllerInterface class.
        """
        return ModuleParameters(
            module_type=self._module_type,
            module_id=self._module_id,
            return_code=np.uint8(0),
            parameter_data=(np.bool(report_ccw), np.bool(report_cw), np.uint32(delta_threshold)),
        )

    def check_state(self, repetition_delay: np.uint32 = np.uint32(0)) -> OneOffModuleCommand | RepeatedModuleCommand:
        """Returns the number of pulses accumulated by the EncoderModule since the last check or reset.

        If there has been a significant change in the absolute count of pulses, reports the change and direction to the
        PC. It is highly advised to issue this command to repeat (recur) at a desired interval to continuously monitor
        the encoder state, rather than repeatedly calling it as a one-off command for best runtime efficiency.

        This command allows continuously monitoring the rotation of the object connected to the encoder. It is designed
        to return the absolute raw count of pulses emitted by the encoder in response to the object ration. This allows
        avoiding floating-point arithmetic on the microcontroller and relies on the PC to convert pulses to standard
        units,s uch as centimeters. The specific conversion algorithm depends on the encoder and motion diameter.

        Args:
            repetition_delay: The time, in microseconds, to delay before repeating the command. If set to 0, the
            command will only run once.

        Returns:
            The RepeatedModuleCommand or OneOffModuleCommand message that can be sent to the microcontroller via the
            send_message() method of the MicroControllerInterface class.
        """
        if repetition_delay == 0:
            return OneOffModuleCommand(
                module_type=self._module_type,
                module_id=self._module_id,
                return_code=np.uint8(0),
                command=np.uint8(1),
                noblock=np.bool(False),
            )

        return RepeatedModuleCommand(
            module_type=self._module_type,
            module_id=self._module_id,
            return_code=np.uint8(0),
            command=np.uint8(1),
            noblock=np.bool(False),
            cycle_delay=np.uint32(repetition_delay),
        )

    def reset_pulse_count(self) -> OneOffModuleCommand:
        """Resets the EncoderModule pulse tracker to 0.

        This command allows resetting the encoder without evaluating its current pulse count. Currently, this command
        is designed ot only run once.

        Returns:
            The OneOffModuleCommand message that can be sent to the microcontroller via the send_message() method of
            the MicroControllerInterface class.
        """
        return OneOffModuleCommand(
            module_type=self._module_type,
            module_id=self._module_id,
            return_code=np.uint8(0),
            command=np.uint8(2),
            noblock=np.bool(False),
        )

    def get_ppr(self) -> OneOffModuleCommand:
        """Uses the index channel of the EncoderModule to estimate its Pulse-per-Revolution (PPR).

        The PPR allows converting raw pulse counts the EncoderModule sends to the PC to accurate displacement in
        standard distance units, such as centimeters. This is a service command not intended to be used during most
        runtimes if the PPR is already known. It relies on the object tracked by the encoder completing up to 11 full
        revolutions and uses the index channel of the encoder to measure the number of pulses per each revolution.

        Notes:
            Make sure the evaluated encoder rotates at a slow and stead speed until this command completes. Similar to
            other service commands, it is designed to deadlock the controller until the command completes. Note, the
            EncoderModule does not provide the rotation, this needs to be done manually.

            The direction of the rotation is not relevant for this command, as long as the object makes the full
            360-degree revolution.

            The command is optimized for the object to be rotated with a human hand at a steady rate, so it delays
            further index pin polling for 100 milliseconds each time the index pin is triggered. Therefore, if the
            object is moving too fast (or too slow), the command will not work as intended.

        Returns:
            The OneOffModuleCommand message that can be sent to the microcontroller via the send_message() method of
            the MicroControllerInterface class.
        """
        return OneOffModuleCommand(
            module_type=self._module_type,
            module_id=self._module_id,
            return_code=np.uint8(0),
            command=np.uint8(3),
            noblock=np.bool(False),
        )

    @property
    def mqtt_topic(self) -> str:
        """Returns the MQTT topic used to transfer motion data from the interface to Unity."""
        return self._motion_topic

    @property
    def cm_per_pulse(self) -> np.float64:
        """Returns the conversion factor to translate raw encoder pulse count to real world centimeters of motion."""
        return np.round(
            a=np.float64((math.pi * self._object_diameter) / self._ppr),
            decimals=12,
        )


class TTLInterface(ModuleInterface):
    """Interfaces with TTLModule instances running on Ataraxis MicroControllers.

    TTLModule facilitates exchanging Transistor-to-Transistor Logic (TTL) signals between various hardware systems, such
    as microcontrollers, cameras and recording devices. The module contains methods for both sending and receiving TTL
    pulses, but each TTLModule instance can only perform one of these functions at a time.

    Notes:
        When the TTLModule is configured to output a signal, it will notify the PC about the initial signal state
        (HIGH or LOW) after setup.
    """

    def __init__(self) -> None:

        error_codes = {np.uint8(51), np.uint8(54)}  # kOutputLocked, kInvalidPinMode

        # kInputOn, kInputOff, kOutputOn, kOutputOff
        # data_codes = {np.uint8(52), np.uint8(53), np.uint8(55), np.uint8(56)}

        super().__init__(
            module_type=np.uint8(1),
            module_id=np.uint8(1),
            mqtt_communication=False,
            data_codes=None,  # None of the data codes needs additional processing, so statically set to None
            unity_command_topics=None,
            error_codes=error_codes
        )

    def process_received_data(
            self,
            message: ModuleData | ModuleState,
            unity_communication: UnityCommunication,
            mp_queue: MPQueue,  # type: ignore
    ) -> None:
        """Not used."""
        return

    def parse_unity_command(self, topic: str, payload: bytes | bytearray) -> None:
        """Not used."""
        return

    def set_parameters(
            self, pulse_duration: np.uint32 = np.uint32(10000), averaging_pool_size: np.uint8 = np.uint8(0)
    ) -> ModuleParameters:
        """Changes the PC-addressable runtime parameters of the TTLModule instance.

        Use this method to package and apply new PC-addressable parameters to the TTLModule instance managed by
        this Interface class.

        Args:
            pulse_duration: The duration, in microseconds, of each emitted TTL pulse HIGH phase. This determines
                how long the TTL pin stays ON when emitting a pulse.
            averaging_pool_size: The number of digital pin readouts to average together when checking pin state. This
                is used during the execution of check_state() command to debounce the pin readout and acts in addition
                to any built-in debouncing.

        Returns:
            The ModuleParameters message that can be sent to the microcontroller via the send_message() method of
            the MicroControllerInterface class.
        """
        return ModuleParameters(
            module_type=self._module_type,
            module_id=self._module_id,
            return_code=np.uint8(0),
            parameter_data=(pulse_duration, averaging_pool_size),
        )

    def send_pulse(
            self, repetition_delay: np.uint32 = np.uint32(0), noblock: bool = True
    ) -> RepeatedModuleCommand | OneOffModuleCommand:
        """Triggers TTLModule to deliver a one-off or recurrent (repeating) digital TTL pulse.

        This command is well-suited to carry out most forms of TTL communication, but it is adapted for comparatively
        low-frequency communication at 10-200 Hz. This is in-contrast to PWM outputs capable of mHz or even Khz pulse
        oscillation frequencies.

        Args:
            repetition_delay: The time, in microseconds, to delay before repeating the command. If set to 0, the command
                will only run once. The exact repetition delay will be further affected by other modules managed by the
                same microcontroller and may not be perfectly accurate.
            noblock: Determines whether the command should block the microcontroller while emitting the high phase of
                the pulse or not. Blocking ensures precise pulse duration, non-blocking allows the microcontroller to
                perform other operations while waiting, increasing its throughput.

        Returns:
            The RepeatedModuleCommand or OneOffModuleCommand message that can be sent to the microcontroller via the
            send_message() method of the MicroControllerInterface class.
        """
        if repetition_delay == 0:
            return OneOffModuleCommand(
                module_type=self._module_type,
                module_id=self._module_id,
                return_code=np.uint8(0),
                command=np.uint8(1),
                noblock=np.bool(noblock),
            )

        return RepeatedModuleCommand(
            module_type=self._module_type,
            module_id=self._module_id,
            return_code=np.uint8(0),
            command=np.uint8(1),
            noblock=np.bool(noblock),
            cycle_delay=repetition_delay,
        )

    def toggle(self, state: bool) -> OneOffModuleCommand:
        """Triggers the TTLModule to continuously deliver a digital HIGH or LOW signal.

        This command locks the TTLModule managed by this Interface into delivering the desired logical signal.

        Args:
            state: The signal to output. Set to True for HIGH and False for LOW.

        Returns:
            The OneOffModuleCommand message that can be sent to the microcontroller via the send_message() method of the
            MicroControllerInterface class.
        """
        return OneOffModuleCommand(
            module_type=self._module_type,
            module_id=self._module_id,
            return_code=np.uint8(0),
            command=np.uint8(2 if state else 3),
            noblock=np.bool(False),
        )

    def check_state(self, repetition_delay: np.uint32 = np.uint32(0)) -> OneOffModuleCommand | RepeatedModuleCommand:
        """Checks the state of the TTL signal received by the TTLModule.

        This command evaluates the state of the TTLModule's input pin and, if it is different from the previous state,
        reports it to the PC. This approach ensures that the module only reports signal level shifts (edges), preserving
        communication bandwidth.

        Args:
            repetition_delay: The time, in microseconds, to delay before repeating the command. If set to 0, the command
            will only run once.

        Returns:
            The RepeatedModuleCommand or OneOffModuleCommand message that can be sent to the microcontroller via the
            send_message() method of the MicroControllerInterface class.
        """
        if repetition_delay == 0:
            return OneOffModuleCommand(
                module_type=self._module_type,
                module_id=self._module_id,
                return_code=np.uint8(0),
                command=np.uint8(4),
                noblock=np.bool(False),
            )

        return RepeatedModuleCommand(
            module_type=self._module_type,
            module_id=self._module_id,
            return_code=np.uint8(0),
            command=np.uint8(4),
            noblock=np.bool(False),
            cycle_delay=repetition_delay,
        )


class BreakInterface(ModuleInterface):
    """Interfaces with BreakModule instances running on Ataraxis MicroControllers.

    BreakModule allows interfacing with a break to dynamically control the motion of break-coupled objects. The module
    is designed to send PWM signals that trigger Field-Effect-Transistor (FET) gated relay hardware to deliver voltage
    that variably engages the break. The module can be used to either fully engage or disengage the breaks or to output
    a PWM signal to engage the break with the desired strength.

    Notes:
        The break will notify the PC about its initial state (Engaged or Disengaged) after setup.

    Args:
        minimum_break_strength: The minimum torque applied by the break in gram centimeter. This is the torque the
            break delivers at minimum voltage (break is disabled).
        maximum_break_strength: The maximum torque applied by the break in gram centimeter. This is the torque the
            break delivers at maximum voltage (break is fully engaged).

    Attributes:
        _minimum_break_strength: The minimum torque the break delivers at minimum voltage (break is disabled).
        _maximum_break_strength: The maximum torque the break delivers at maximum voltage (break is fully engaged).
        _newton_per_gram_centimeter: Conversion factor from torque force in g cm to torque force in N cm.
        _torque_per_pwm: Conversion factor from break pwm levels to breaking force in N cm.
    """

    def __init__(
            self,
            minimum_break_strength: float = 43.2047,  # 0.6 in iz
            maximum_break_strength: float = 1152.1246,  # 16 in oz
    ) -> None:
        error_codes = {np.uint8(51)}  # kOutputLocked
        # data_codes = {np.uint8(52), np.uint8(53), np.uint8(54)}  # kEngaged, kDisengaged, kVariable

        # Initializes the subclassed ModuleInterface using the input instance data. Type data is hardcoded.
        super().__init__(
            module_type=np.uint8(3),
            module_id=np.uint8(1),
            mqtt_communication=False,
            data_codes=None,  # None of the data codes need additional processing, so set to None.
            unity_command_topics=None,
            error_codes=error_codes
        )

        # Hardcodes the conversion factor used to translate torque force in g cm to N cm
        self._newton_per_gram_centimeter: float = 0.0000981

        # Stores additional data into class attributes. Rounds to 12 decimal places for consistency and to ensure
        # repeatability.
        self._minimum_break_strength: np.float64 = np.round(
            a=minimum_break_strength / self._newton_per_gram_centimeter,
            decimals=12,
        )
        self._maximum_break_strength: np.float64 = np.round(
            a=maximum_break_strength / self._newton_per_gram_centimeter,
            decimals=12,
        )

        # Computes the conversion factor to translate break pwm levels into breaking force in Newtons cm. Rounds
        # to 12 decimal places for consistency and to ensure repeatability.
        self._torque_per_pwm = np.round(
            a=(self._maximum_break_strength - self._minimum_break_strength) / 255,
            decimals=12,
        )

    def process_received_data(
            self,
            message: ModuleData | ModuleState,
            unity_communication: UnityCommunication,
            queue: MPQueue,  # type: ignore
    ) -> None:
        """Not used."""
        return

    def parse_unity_command(self, topic: str, payload: bytes | bytearray) -> None:
        """Not used."""
        return

    def get_pwm_from_force(self, target_force_n_cm: float) -> np.uint8:
        """Converts the desired breaking force in Newtons centimeter to the required PWM value (0-255) to be delivered
        to the break hardware by the BreakModule.

        Use this method to convert the desired breaking force into the PWM value that can be submitted to the
        BreakModule via the set_parameters() class method.

        Args:
            target_force_n_cm: Desired force in Newtons centimeter at the edge of the object.

        Returns:
            The byte PWM value that would generate the desired amount of torque.

        Raises:
            ValueError: If the input force is not within the valid range for the BreakModule.
        """
        if self._maximum_break_strength > target_force_n_cm or self._minimum_break_strength < target_force_n_cm:
            message = (
                f"The requested force {target_force_n_cm} N cm is outside the valid range for the BreakModule "
                f"{self._module_id}. Valid breaking force range is from {self._minimum_break_strength} to "
                f"{self._maximum_break_strength}."
            )
            console.error(message=message, error=ValueError)

        # Calculates PWM using the pre-computed torque_per_pwm conversion factor
        pwm_value = np.uint8(round((target_force_n_cm - self._minimum_break_strength) / self._torque_per_pwm))

        return pwm_value

    def set_parameters(self, breaking_strength: np.uint8 = np.uint8(255)) -> ModuleParameters:
        """Changes the PC-addressable runtime parameters of the BreakModule instance.

        Use this method to package and apply new PC-addressable parameters to the BreakModule instance managed by this
        Interface class.

        Notes:
            Use set_breaking_power() command to apply the breaking-strength transmitted in this parameter message to the
            break. Until the command is called, the new breaking_strength will not be applied to the break hardware.

        Args:
            breaking_strength: The Pulse-Width-Modulation (PWM) value to use when the BreakModule delivers adjustable
                breaking power. Depending on this value, the breaking power can be adjusted from none (0) to maximum
                (255). Use get_pwm_from_force() to translate desired breaking force into the required PWM value.

        Returns:
            The ModuleParameters message that can be sent to the microcontroller via the send_message() method of
            the MicroControllerInterface class.
        """
        return ModuleParameters(
            module_type=self._module_type,
            module_id=self._module_id,
            return_code=np.uint8(0),  # Generally, return code is only helpful for debugging.
            parameter_data=(breaking_strength,),
        )

    def toggle(self, state: bool) -> OneOffModuleCommand:
        """Triggers the BreakModule to be permanently engaged at maximum strength or permanently disengaged.

        This command locks the BreakModule managed by this Interface into the desired state.

        Notes:
            This command does NOT use the breaking_strength parameter and always uses either maximum or minimum breaking
            power. To set the break to a specific torque level, set the level via the set_parameters() method and then
            switch the break into the variable torque mode by using the set_breaking_power() method.

        Args:
            state: The desired state of the break. True means the break is engaged; False means the break is disengaged.

        Returns:
            The OneOffModuleCommand message that can be sent to the microcontroller via the send_message() method of the
            MicroControllerInterface class.
        """
        return OneOffModuleCommand(
            module_type=self._module_type,
            module_id=self._module_id,
            return_code=np.uint8(0),
            command=np.uint8(1 if state else 2),
            noblock=np.bool(False),
        )

    def set_breaking_power(self) -> OneOffModuleCommand:
        """Triggers the BreakModule to engage with the strength (torque) defined by the breaking_strength runtime
        parameter.

        Unlike the toggle() method, this method allows precisely controlling the torque applied by the break. This
        is achieved by pulsing the break control pin at the PWM level specified by breaking_strength runtime parameter
        stored in BreakModule's memory (on the microcontroller).

        Notes:
            This command switches the break to run in the variable strength mode and applies the current value of the
            breaking_strength parameter to the break, but it does not determine the breaking power. To adjust the power,
            use the set_parameters() class method to issue updated breaking_strength value. By default, the break power
            is set to 50% (PWM value 128).

        Returns:
            The OneOffModuleCommand message that can be sent to the microcontroller via the send_message() method of the
            MicroControllerInterface class.
        """
        return OneOffModuleCommand(
            module_type=self._module_type,
            module_id=self._module_id,
            return_code=np.uint8(0),
            command=np.uint8(3),
            noblock=np.bool(False),
        )

    @property
    def force_per_pwm(self) -> np.float64:
        """Returns the conversion factor to translate break pwm levels into breaking force in Newtons centimeter."""
        return self._torque_per_pwm


class ValveInterface(ModuleInterface):
    """Interfaces with ValveModule instances running on Ataraxis MicroControllers.

    ValveModule allows interfacing with a solenoid valve to controllably dispense precise amounts of fluid. The module
    is designed to send digital signals that trigger Field-Effect-Transistor (FET) gated relay hardware to deliver
    voltage that opens or closes the controlled valve. The module can be used to either permanently open or close the
    valve or to cycle opening and closing in a way that ensures a specific amount of fluid passes through the
    valve.

    Notes:
        This interface comes pre-configured to receive valve pulse triggers from Unity via the "Gimbl/Reward/"
        topic.

        The valve will notify the PC about its initial state (Open or Closed) after setup.

    Args:
        valve_calibration_data: A tuple of tuples that contains the data required to map pulse duration to delivered
            fluid volume. Each sub-tuple should contain the integer that specifies the pulse duration in microseconds
            and a float that specifies the delivered fluid volume in microliters. If you do not know this data,
            initialize the class using a placeholder calibration tuple and use calibration() class method to collect
            this data using the ValveModule.

    Attributes:
        _microliters_per_microsecond: The conversion factor from desired fluid volume in microliters to the pulse
            valve duration in microseconds.
        _reward_topic: Stores the topic used by Unity to issue reward commadns to the module.
    """

    def __init__(
            self,
            valve_calibration_data: tuple[tuple[int | float, int | float], ...]
    ) -> None:
        error_codes = {np.uint8(51)}  # kOutputLocked
        # data_codes = {np.uint8(52), np.uint8(53), np.uint8(54)}  # kOpen, kClosed, kCalibrated
        data_codes = {np.uint8(54)}  # The only code that requires additional processing is kCalibrated
        unity_command_topics = {"Gimbl/Reward/"}

        super().__init__(
            module_type=np.uint8(5),
            module_id=np.uint8(1),
            mqtt_communication=True,
            data_codes=data_codes,
            unity_command_topics=unity_command_topics,
            error_codes=error_codes
        )

        # Extracts pulse durations and fluid volumes into separate arrays
        pulse_durations: NDArray[np.float64] = np.array([x[0] for x in valve_calibration_data], dtype=np.float64)
        fluid_volumes: NDArray[np.float64] = np.array([x[1] for x in valve_calibration_data], dtype=np.float64)

        # Computes the conversion factor by finding the slope of the calibration curve.
        # Rounds to 12 decimal places for consistency and to ensure repeatability
        slope: np.float64 = polyfit(x=pulse_durations, y=fluid_volumes, deg=1)[0]  # type: ignore
        self._microliters_per_microsecond: np.float64 = np.round(a=slope, decimals=12)

        # Stores the reward topic separately to make it accessible via property
        self._reward_topic = "Gimbl/Reward/"

    def process_received_data(
            self,
            message: ModuleData | ModuleState,
            unity_communication: UnityCommunication,
            mp_queue: MPQueue,  # type: ignore
    ) -> None:
        # Since the only data code that requires further processing is code 54 (kCalibrated), this method statically
        # puts 'calibrated' into the queue as a one-element tuple.
        mp_queue.put(('Calibrated',))

    def parse_unity_command(self, topic: str, payload: bytes | bytearray) -> OneOffModuleCommand:
        # If the received message was sent to the reward topic, this is a binary (empty payload) trigger to
        # pulse the valve. It is expected that the valve parameters are configured so that this delivers the
        # desired amount of water reward.
        return OneOffModuleCommand(
            module_type=self._module_type,
            module_id=self._module_id,
            return_code=np.uint8(0),
            command=np.uint8(1),
            noblock=np.bool(False),  # Blocks to ensure reward delivery precision.
        )

    def get_duration_from_volume(self, volume: float) -> np.uint32:
        """Converts the desired fluid volume in microliters to the valve pulse duration in microseconds that ValveModule
        will use to deliver that fluid volume.

        Use this method to convert the desired fluid volume into the pulse_duration value that can be submitted to the
        ValveModule via the set_parameters() class method.

        Args:
            volume: Desired fluid volume in microliters.

        Returns:
            The microsecond pulse duration that would be used to deliver the specified volume.
        """
        return np.uint32(np.round(volume / self._microliters_per_microsecond))

    def set_parameters(
            self,
            pulse_duration: np.uint32 = np.uint32(10000),
            calibration_delay: np.uint32 = np.uint32(10000),
            calibration_count: np.uint16 = np.uint16(100),
    ) -> ModuleParameters:
        """Changes the PC-addressable runtime parameters of the ValveModule instance.

        Use this method to package and apply new PC-addressable parameters to the ValveModule instance managed by this
        Interface class.

        Args:
            pulse_duration: The time, in microseconds, the valve stays open when it is pulsed (opened and closed). This
                is used during the execution of send_pulse() command to control the amount of dispensed fluid. Use
                get_duration_from_volume() method to convert the desired fluid volume into the pulse_duration value.
            calibration_delay: The time, in microseconds, to wait between consecutive pulses during calibration.
                Calibration works by repeatedly pulsing the valve the requested number of times. Delaying after closing
                the valve (ending the pulse) ensures the valve hardware has enough time to respond to the inactivation
                phase before starting the next calibration cycle.
            calibration_count: The number of times to pulse the valve during calibration. A number between 10 and 100 is
                enough for most use cases.

        Returns:
            The ModuleParameters message that can be sent to the microcontroller via the send_message() method of
            the MicroControllerInterface class.
        """
        return ModuleParameters(
            module_type=self._module_type,
            module_id=self._module_id,
            return_code=np.uint8(0),
            parameter_data=(pulse_duration, calibration_delay, calibration_count),
        )

    def send_pulse(
            self, repetition_delay: np.uint32 = np.uint32(0), noblock: bool = False
    ) -> RepeatedModuleCommand | OneOffModuleCommand:
        """Triggers ValveModule to deliver a precise amount of fluid by cycling opening and closing the valve once or
        repetitively (recurrently).

        After calibration, this command allows delivering precise amounts of fluid with, depending on the used valve and
        relay hardware microliter or nanoliter precision. This command is optimized to change valve states at a
        comparatively low frequency in the 10-200 Hz range.

        Notes:
            To ensure the accuracy of fluid delivery, it is recommended to run the valve in the blocking mode
            and, if possible, isolate it to a controller that is not busy with running other tasks.

        Args:
            repetition_delay: The time, in microseconds, to delay before repeating the command. If set to 0, the command
                will only run once. The exact repetition delay will be further affected by other modules managed by the
                same microcontroller and may not be perfectly accurate.
            noblock: Determines whether the command should block the microcontroller while the valve is kept open or
                not. Blocking ensures precise pulse duration and, by extension, delivered fluid volume. Non-blocking
                allows the microcontroller to perform other operations while waiting, increasing its throughput.

        Returns:
            The RepeatedModuleCommand or OneOffModuleCommand message that can be sent to the microcontroller via the
            send_message() method of the MicroControllerInterface class.
        """
        if repetition_delay == 0:
            return OneOffModuleCommand(
                module_type=self._module_type,
                module_id=self._module_id,
                return_code=np.uint8(0),
                command=np.uint8(1),
                noblock=np.bool(noblock),
            )

        return RepeatedModuleCommand(
            module_type=self._module_type,
            module_id=self._module_id,
            return_code=np.uint8(0),
            command=np.uint8(1),
            noblock=np.bool(noblock),
            cycle_delay=repetition_delay,
        )

    def toggle(self, state: bool) -> OneOffModuleCommand:
        """Triggers the ValveModule to be permanently open or closed.

        This command locks the ValveModule managed by this Interface into the desired state.

        Args:
            state: The desired state of the valve. True means the valve is open; False means the valve is closed.

        Returns:
            The OneOffModuleCommand message that can be sent to the microcontroller via the send_message() method of the
            MicroControllerInterface class.
        """
        return OneOffModuleCommand(
            module_type=self._module_type,
            module_id=self._module_id,
            return_code=np.uint8(0),
            command=np.uint8(2 if state else 3),
            noblock=np.bool(False),
        )

    def calibrate(self) -> OneOffModuleCommand:
        """Triggers ValveModule to repeatedly pulse the valve using the duration defined by the pulse_duration runtime
        parameter.

        This command is used to build the calibration map of the valve that matches pulse_duration to the volume of
        fluid dispensed during the time the valve is open. To do so, the command repeatedly pulses the valve to dispense
        a large volume of fluid which can be measured and averaged to get the volume of fluid delivered during each
        pulse. The number of pulses carried out during this command is specified by the calibration_count parameter, and
        the delay between pulses is specified by the calibration_delay parameter.

        Notes:
            When activated, this command will block in-place until the calibration cycle is completed. Currently, there
            is no way to interrupt the command, and it may take a prolonged period of time (minutes) to complete.

            This command does not set any of the parameters involved in the calibration process. Make sure the
            parameters are submitted to the ValveModule's hardware memory via the set_parameters() class method before
            running the calibration() command.

        Returns:
            The OneOffModuleCommand message that can be sent to the microcontroller via the send_message() method of
            the MicroControllerInterface class.
        """
        return OneOffModuleCommand(
            module_type=self._module_type,
            module_id=self._module_id,
            return_code=np.uint8(0),
            command=np.uint8(4),
            noblock=np.bool(False),
        )

    @property
    def mqtt_topic(self) -> str:
        """Returns the MQTT topic monitored by the module to receive reward commands from Unity."""
        return self._reward_topic

    @property
    def microliter_per_microsecond(self) -> np.float64:
        """Returns the conversion factor to translate valve open time, in microseconds, into the volume of dispensed
        fluid, in microliters."""
        return self._microliters_per_microsecond


class LickInterface(ModuleInterface):
    """Interfaces with LickModule instances running on Ataraxis MicroControllers.

    LickModule allows interfacing with conductive lick sensors used in the Sun Lab to detect mouse interaction with
    water dispensing tubes. The sensor works by sending a small direct current through the mouse, which is picked up by
    the sensor connected to the metal lick tube. When the mouse completes the circuit by making the contact with the
    tube, the sensor determines whether the resultant voltage matches the threshold expected for a tongue contact and,
    if so, notifies the PC about the contact.

    Notes:
        The sensor is calibrated to work with very small currents the animal does not detect, so it does not interfere
        with behavior during experiments. The sensor will, however, interfere with electrophysiological recordings.

        The resolution of the sensor is high enough to distinguish licks from paw touches. By default, the
        microcontroller is configured in a way that will likely send both licks and non-lick interactions to the PC.
        Use lick_threshold argument to provide a more exclusive lick threshold.

        The interface automatically sends significant lick triggers to Unity via the "LickPort/" MQTT topic. This only
        includes the 'onset' triggers, the interface does not report voltage level reductions (associated with the end
        of the mouse-to-tube contact).

        The default state of the sensor after setup or reset is 0. Until the sensor sends a state message communicating
        a non-zero detected value, it can be safely assumed that the sensor detects the voltage of 0.

    Args:
        lick_threshold: The threshold voltage, in raw analog units recorded by a 12-bit ADC, for detecting the tongue
            contact. Note, 12-bit ADC only supports values between 0 and 4095, so setting the threshold above 4095 will
            result in no licks being reported to Unity.

    Attributes:
        _sensor_topic: Stores the output MQTT topic.
        _lick_threshold: The threshold voltage for detecting a tongue contact.
        _volt_per_adc: The conversion factor to translate the raw analog values recorded by the 12-bit ADC into
            voltage in Volts.
    """

    def __init__(
            self,
            lick_threshold: int = 2000,
    ) -> None:
        data_codes = {np.uint8(51)}  # kChanged

        # Initializes the subclassed ModuleInterface using the input instance data. Type data is hardcoded.
        super().__init__(
            module_type=np.uint8(4),
            module_id=np.uint8(1),
            mqtt_communication=True,
            data_codes=data_codes,
            unity_command_topics=None,
            error_codes=None
        )

        self._sensor_topic: str = "LickPort/"
        self._lick_threshold: np.uint16 = np.uint16(lick_threshold)

        # Statically computes the voltage resolution of each analog step, assuming a 3.3V ADC with 12-bit resolution.
        self._volt_per_adc = np.round(
            a=np.float64(3.3 / (2 ** 12)),
            decimals=12
        )

    def process_received_data(
            self,
            message: ModuleData | ModuleState,
            unity_communication: UnityCommunication,
            mp_queue: MPQueue,  # type: ignore
    ) -> None:

        # Currently, the only data_code that requires additional processing is code 51 (sensor readout change code).
        if message.event == 51 and message.data_object >= self._lick_threshold:  # Threshold is inclusive

            # If the sensor detects a significantly high voltage, sends an empty message to the sensor MQTT topic,
            # which acts as a binary lick trigger.
            unity_communication.send_data(topic=self._sensor_topic, payload=None)

    def parse_unity_command(self, topic: str, payload: bytes | bytearray) -> None:
        """Not used."""
        return

    def set_parameters(
            self,
            lower_threshold: np.uint16 = np.uint16(1000),
            upper_threshold: np.uint16 = np.uint16(4095),
            delta_threshold: np.uint16 = np.uint16(500),
            averaging_pool_size: np.uint8 = np.uint8(50),
    ) -> ModuleParameters:
        """Changes the PC-addressable runtime parameters of the LickModule instance.

        Use this method to package and apply new PC-addressable parameters to the LickModule instance managed by this
        Interface class.

        Notes:
            All threshold parameters are inclusive! if you need help determining appropriate threshold levels for
            specific targeted voltages, use convert_to_adc() method of the interface instance.

        Args:
            lower_threshold: The minimum voltage level, in raw analog units of 12-bit Analog-to-Digital-Converter (ADC),
                that needs to be reported to the PC. Setting this threshold to a number above zero allows high-pass
                filtering the incoming signals. Note, the threshold only applies to the rising edge of the signal,
                going from a high to low value does not respect this threshold.
            upper_threshold: The maximum voltage level, in raw analog units of 12-bit Analog-to-Digital-Converter (ADC),
                that needs to be reported to the PC. Setting this threshold to a number below 4095 allows low-pass
                filtering the incoming signals.
            delta_threshold: The minimum value by which the signal has to change, relative to the previous check, for
                the change to be reported to the PC. Note, if the change is 0, the signal will not be reported to the
                PC, regardless of this parameter value.
            averaging_pool_size: The number of analog pin readouts to average together when checking pin state. This
                is used to smooth the recorded values to avoid communication line noise. It is highly advised to
                have this enabled and set to at least 10 readouts.

        Returns:
            The ModuleParameters message that can be sent to the microcontroller via the send_message() method of
            the MicroControllerInterface class.
        """
        return ModuleParameters(
            module_type=self._module_type,
            module_id=self._module_id,
            return_code=np.uint8(0),  # Generally, return code is only helpful for debugging.
            parameter_data=(upper_threshold, lower_threshold, delta_threshold, averaging_pool_size),
        )

    def get_adc_from_volts(self, voltage: int | float) -> np.uint16:
        """Converts the input voltage to raw analog units of 12-bit Analog-to-Digital-Converter (ADC).

        Use this method to determine the appropriate raw analog units for the threshold arguments of the
        set_parameters() method, based on the desired voltage thresholds.

        Notes:
            This method assumes a 3.3V ADC with 12-bit resolution.

        Args:
            voltage: The voltage to convert to raw analog units, in Volts.

        Returns:
            The raw analog units of 12-bit ADC for the input voltage.
        """
        return np.uint16(np.round(voltage / self._volt_per_adc))

    def check_state(self, repetition_delay: np.uint32 = np.uint32(0)) -> OneOffModuleCommand | RepeatedModuleCommand:
        """ Returns the voltage signal detected by the analog pin monitored by the LickModule.

        If there has been a significant change in the detected voltage level and the level is within the reporting
        thresholds, reports the change to the PC. It is highly advised to issue this command to repeat (recur) at a
        desired interval to continuously monitor the lick sensor state, rather than repeatedly calling it as a one-off
        command for best runtime efficiency.

        This command allows continuously monitoring the mouse interaction with the lickport tube. It is designed
        to return the raw analog units, measured by a 3.3V ADC with 12-bit resolution. To avoid floating-point math, the
        value is returned as an unsigned 16-bit integer.

        Args:
            repetition_delay: The time, in microseconds, to delay before repeating the command. If set to 0, the
            command will only run once.

        Returns:
            The RepeatedModuleCommand or OneOffModuleCommand message that can be sent to the microcontroller via the
            send_message() method of the MicroControllerInterface class.
        """
        if repetition_delay == 0:
            return OneOffModuleCommand(
                module_type=self._module_type,
                module_id=self._module_id,
                return_code=np.uint8(0),
                command=np.uint8(1),
                noblock=np.bool(False),
            )

        return RepeatedModuleCommand(
            module_type=self._module_type,
            module_id=self._module_id,
            return_code=np.uint8(0),
            command=np.uint8(1),
            noblock=np.bool(False),
            cycle_delay=repetition_delay,
        )

    @property
    def mqtt_topic(self) -> str:
        """Returns the MQTT topic used to transfer lick events from the interface to Unity."""
        return self._sensor_topic

    @property
    def volts_per_adc(self) -> np.float64:
        """Returns the conversion factor to translate the raw analog values recorded by the 12-bit ADC into voltage in
        Volts."""
        return self._volt_per_adc


class TorqueInterface(ModuleInterface):
    """Interfaces with TorqueModule instances running on Ataraxis MicroControllers.

    TorqueModule interfaces with a bipolar torque sensor. The sensor uses bipolar coding in the millivolt range to
    communicate torque in the CW and the CCW direction. To convert and amplify the output of the torque sensor, it is
    wired to an AD620 microvolt amplifier instrument, that converts the output signal into a positive vector and
    amplifies its strength to Volts range.

    The TorqueModule further refines the signal by converting it to a range from 0 to baseline, ensuring that CCW and
    CW torque signals behave identically (go from 0 up to baseline as the torque increases).

    Notes:
        This interface receives torque as a positive uint16_t value from 0 to at most 2046 raw analog units of 3.3v
        12-bit ADC converter. The direction of the torque is reported by the event-code of the received message.

        The default state of the sensor after setup or reset is 0. Until the sensor sends a state message communicating
        a non-zero detected value, it can be safely assumed that the sensor detects the torque of 0. The torque of 0
        essentially has no direction, as it means there is no CW or CCW torque.

    Args:
        baseline_voltage: The voltage level, in raw analog units measured by 3.3v ADC at 12-bit resolution after the
            AD620 amplifier, that corresponds to 0 torque readout. Usually, for a 3.3v ADC, this would be around
            2046 (the midpoint, ~1.65 V).
        maximum_voltage: The voltage level, in raw analog units measured by 3.3v ADC at 12-bit resolution after the
            AD620 amplifier, that corresponds to the absolute maximum torque detectable by the sensor. The best way
            to get this value is to measure the positive voltage level after applying the maximum CW (positive) torque.
            At most, this value can be 4095 (~3.3 V).
        sensor_capacity: The maximum torque detectable by the sensor, in grams centimeter (g cm).

    Attributes:
        _newton_per_gram_centimeter: Stores the hardcoded conversion factor from gram centimeter to Newton centimeter.
        _capacity_in_newtons_cm: The maximum torque detectable by the sensor in Newtons centimeter.
        _torque_per_adc: The conversion factor to translate raw analog 3.3v 12-bit ADC values to torque in Newtons
            centimeter.
    """

    def __init__(
            self,
            baseline_voltage: np.uint16 = np.uint16(2046),
            maximum_voltage: np.uint16 = np.uint16(4095),
            sensor_capacity: np.float64 = np.float64(720.0779)  # 10 oz in
    ) -> None:
        # data_codes = {np.uint8(51), np.uint8(52)}  # kCCWTorque, kCWTorque

        # Initializes the subclassed ModuleInterface using the input instance data. Type data is hardcoded.
        super().__init__(
            module_type=np.uint8(6),
            module_id=np.uint8(1),
            mqtt_communication=False,
            data_codes=None,
            unity_command_topics=None,
            error_codes=None
        )

        # Hardcodes the conversion factor used to translate torque force in g cm to N cm
        self._newton_per_gram_centimeter: float = 0.0000981

        # Determines the capacity of the torque sensor in Newtons centimeter.
        self._capacity_in_newtons_cm: np.float64 = np.round(
            a=sensor_capacity / self._newton_per_gram_centimeter,
            decimals=12,
        )

        # Determines the conversion factor to go from ADC raw units to volts.
        volts_per_adc = np.float64(3.3 / (2 ** 12))

        # Computes the conversion factor to translate the recorded raw analog readouts of the 3.3V 12-bit ADC to
        # unidirectional torque force in Newton centimeter. Rounds to 12 decimal places for consistency and to ensure
        # repeatability.
        self._torque_per_adc = np.round(
            a=((self._capacity_in_newtons_cm / (maximum_voltage - baseline_voltage)) / volts_per_adc),
            decimals=12,
        )

    def process_received_data(
            self,
            message: ModuleData | ModuleState,
            unity_communication: UnityCommunication,
            mp_queue: MPQueue,  # type: ignore
    ) -> None:
        """Not used."""
        return

    def parse_unity_command(self, topic: str, payload: bytes | bytearray) -> None:
        """Not used."""
        return

    def set_parameters(
            self,
            report_ccw: np.bool = np.bool(True),
            report_cw: np.bool = np.bool(True),
            lower_threshold: np.uint16 = np.uint16(200),
            upper_threshold: np.uint16 = np.uint16(2046),
            delta_threshold: np.uint16 = np.uint16(100),
            averaging_pool_size: np.uint8 = np.uint8(50)
    ) -> ModuleParameters:
        """Changes the PC-addressable runtime parameters of the TorqueModule instance.

        Use this method to package and apply new PC-addressable parameters to the TorqueModule instance managed by this
        Interface class.

        Notes:
            All threshold parameters are inclusive! If you need help determining appropriate threshold levels for
            specific targeted torque levels, use convert_to_adc() method of the interface instance.

        Args:
            report_ccw: Determines whether the sensor should report torque in the CounterClockwise (CCW) direction.
            report_cw: Determines whether the sensor should report torque in the Clockwise (CW) direction.
            lower_threshold: The minimum torque level, in raw analog units of 12-bit Analog-to-Digital-Converter (ADC),
                that needs to be reported to the PC. Setting this threshold to a number above zero allows high-pass
                filtering the incoming signals.
            upper_threshold: The maximum torque level, in raw analog units of 12-bit Analog-to-Digital-Converter (ADC),
                that needs to be reported to the PC. Setting this threshold to a number below 4095 allows low-pass
                filtering the incoming signals.
            delta_threshold: The minimum value by which the signal has to change, relative to the previous check, for
                the change to be reported to the PC. Note, if the change is 0, the signal will not be reported to the
                PC, regardless of this parameter value.
            averaging_pool_size: The number of analog pin readouts to average together when checking pin state. This
                is used to smooth the recorded values to avoid communication line noise. It is highly advised to
                have this enabled and set to at least 10 readouts.

        Returns:
            The ModuleParameters message that can be sent to the microcontroller via the send_message() method of
            the MicroControllerInterface class.
        """
        return ModuleParameters(
            module_type=self._module_type,
            module_id=self._module_id,
            return_code=np.uint8(0),  # Generally, return code is only helpful for debugging.
            parameter_data=(
                report_ccw, report_cw, upper_threshold, lower_threshold, delta_threshold, averaging_pool_size),
        )

    def check_state(self, repetition_delay: np.uint32 = np.uint32(0)) -> OneOffModuleCommand | RepeatedModuleCommand:
        """Returns the torque signal detected by the analog pin monitored by the TorqueModule.

        If there has been a significant change in the detected signal (voltage) level and the level is within the
        reporting thresholds, reports the change to the PC. It is highly advised to issue this command to repeat
        (recur) at a desired interval to continuously monitor the lick sensor state, rather than repeatedly calling it
        as a one-off command for best runtime efficiency.

        This command allows continuously monitoring the CW and CCW torque experienced by the object connected to the
        torque sensor. It is designed to return the raw analog units, measured by a 3.3V ADC with 12-bit resolution.
        To avoid floating-point math, the value is returned as an unsigned 16-bit integer.

        Notes:
            Due to how the torque signal is measured and processed, the returned value will always be between 0 and
            the baseline ADC value. For a 3.3V 12-bit ADC, this is between 0 and ~1.65 Volts.

        Args:
            repetition_delay: The time, in microseconds, to delay before repeating the command. If set to 0, the
            command will only run once.

        Returns:
            The RepeatedModuleCommand or OneOffModuleCommand message that can be sent to the microcontroller via the
            send_message() method of the MicroControllerInterface class.
        """
        if repetition_delay == 0:
            return OneOffModuleCommand(
                module_type=self._module_type,
                module_id=self._module_id,
                return_code=np.uint8(0),
                command=np.uint8(1),
                noblock=np.bool(False),
            )

        return RepeatedModuleCommand(
            module_type=self._module_type,
            module_id=self._module_id,
            return_code=np.uint8(0),
            command=np.uint8(1),
            noblock=np.bool(False),
            cycle_delay=repetition_delay,
        )

    def convert_to_adc(self, torque: int | float) -> np.uint16:
        """Converts the input torque to raw analog units of 12-bit Analog-to-Digital-Converter (ADC).

        Use this method to determine the appropriate raw analog units for the threshold arguments of the
        set_parameters() method.

        Notes:
            This method assumes a 3.3V ADC with 12-bit resolution.

        Args:
            torque: The vtarget torque in Newton centimeter, to convert to an ADC threshold.

        Returns:
            The raw analog units of 12-bit ADC for the input torque.
        """
        return np.uint16(np.round(torque / self._torque_per_adc))

    @property
    def torque_per_adc(self) -> np.float64:
        """Returns the conversion factor to translate the raw analog values recorded by the 12-bit ADC into torque in
        Newton centimeter."""
        return self._torque_per_adc
