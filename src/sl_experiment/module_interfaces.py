"""This module provides ModuleInterface implementations for the hardware used by the Sun lab VR-Mesoscope system."""

from json import dumps
import math

import numpy as np
from ataraxis_data_structures.shared_memory.shared_memory_array import SharedMemoryArray
from numpy.typing import NDArray
from ataraxis_base_utilities import console
from ataraxis_communication_interface import (
    ModuleData,
    ModuleState,
    ModuleInterface,
    ModuleParameters,
    MQTTCommunication,
    OneOffModuleCommand,
    RepeatedModuleCommand,
)
from typing import Any
from scipy.optimize import curve_fit
from ataraxis_time import PrecisionTimer


class EncoderInterface(ModuleInterface):
    """Interfaces with EncoderModule instances running on Ataraxis MicroControllers.

    EncoderModule allows interfacing with quadrature encoders used to monitor the direction and magnitude of a connected
    object's rotation. To achieve the highest resolution, the module relies on hardware interrupt pins to detect and
    handle the pulses sent by the two encoder channels.

    Notes:
        This interface sends CW and CCW motion data to Unity via 'LinearTreadmill/Data' MQTT topic.

        The default initial encoder readout is zero (no CW or CCW motion). The class instance is zeroed at communication
        initialization.

    Args:
        encoder_ppr: The resolution of the managed quadrature encoder, in Pulses Per Revolution (PPR). This is the
            number of quadrature pulses the encoder emits per full 360-degree rotation. If this number is not known,
            provide a placeholder value and use the get_ppr () command to estimate the PPR using the index channel of
            the encoder.
        object_diameter: The diameter of the rotating object connected to the encoder, in centimeters. This is used to
            convert encoder pulses into rotated distance in cm.
        cm_per_unity_unit: The length of each Unity 'unit' in centimeters. This is used to translate raw encoder pulses
            into Unity 'units' before sending the data to Unity.
        debug: A boolean flag that configures the interface to dump certain data received from the microcontroller into
            the terminal. This is used during debugging and system calibration and should be disabled for most runtimes.

    Attributes:
        _motion_topic: Stores the MQTT motion topic.
        _ppr: Stores the resolution of the managed quadrature encoder.
        _object_diameter: Stores the diameter of the object connected to the encoder.
        _cm_per_pulse: Stores the conversion factor that translates encoder pulses into centimeters.
        _unity_unit_per_pulse: Stores the conversion factor to translate encoder pulses into Unity units.
        _communication: Stores the communication class used to send data to Unity over MQTT.
        _debug: Stores the debug flag.
        _speed_tracker: Stores the SharedMemoryArray that stores the average running speed of the animal.
        _speed_timer: Stores the PrecisionTimer instance when the class is in the Communication process. The timer is
            used to calculate the average running speed from the received encoder data.
        _previous_position: Stores the previous absolute position of the animal in centimeters relative to the runtime
            onset. This value is updated to the _current_position value every 100 milliseconds.
        _current_position: Stores the current absolute position of the animal in centimeters relative to the runtime
            onset. This value is updated each time the encoder sends data to the PC.
    """

    def __init__(
        self,
        encoder_ppr: int = 8192,
        object_diameter: float = 15.0333,  # 0333 is to account for the wheel wrap
        cm_per_unity_unit: float = 10.0,
        debug: bool = False,
    ) -> None:
        data_codes: set[np.uint8] = {np.uint8(51), np.uint8(52), np.uint8(53)}  # kRotatedCCW, kRotatedCW, kPPR

        super().__init__(
            module_type=np.uint8(2),
            module_id=np.uint8(1),
            mqtt_communication=True,
            data_codes=data_codes,
            mqtt_command_topics=None,
            error_codes=None,
        )

        # Saves additional data to class attributes.
        self._motion_topic: str = "LinearTreadmill/Data"  # Hardcoded output topic
        self._ppr: int = encoder_ppr
        self._object_diameter: float = object_diameter
        self._debug: bool = debug

        # Computes the conversion factor to go from pulses to centimeters
        self._cm_per_pulse: np.float64 = np.round(
            a=np.float64((math.pi * self._object_diameter) / self._ppr),
            decimals=8,
        )

        # Computes the conversion factor to translate encoder pulses into unity units. Rounds to 8 decimal places for
        # consistency and to ensure repeatability.
        self._unity_unit_per_pulse: np.float64 = np.round(
            a=np.float64((math.pi * object_diameter) / (encoder_ppr * cm_per_unity_unit)),
            decimals=8,
        )

        # The communication class used to send data to Unity over MQTT. Initializes to a placeholder due to pickling
        # issues
        self._communication: MQTTCommunication | None = None

        # Precreates a shared memory array used to track and share the running speed of the animal. This is primarily
        # used for data visualization and during run training.
        self._speed_tracker: SharedMemoryArray = SharedMemoryArray.create_array(
            name=f"{self._module_type}_{self._module_id}_speed_tracker",
            prototype=np.empty(shape=1, dtype=np.float64),
            exist_ok=True,
        )

        # Initializes additional assets used to calculate running speed
        self._speed_timer: None | PrecisionTimer = None  # Placeholder
        self._previous_position: np.float64 = np.float64(0)
        self._current_position: np.float64 = np.float64(0)

    def initialize_remote_assets(self) -> None:
        """Initializes the MQTTCommunication class and connects to the MQTT broker.

        Also connects to the speed_tracker SharedMemoryArray and initializes the PrecisionTimer used in running speed
        calculation.
        """
        # MQTT Client is used to send motion data to Unity over MQTT
        self._communication = MQTTCommunication()
        self._communication.connect()
        self._speed_tracker.connect()
        self._speed_timer = PrecisionTimer("ms")

    def terminate_remote_assets(self) -> None:
        """Destroys the MQTTCommunication class and disconnects from the speed_tracker SharedMemoryArray."""
        self._communication.disconnect()
        self._speed_tracker.disconnect()

    def process_received_data(self, message: ModuleState | ModuleData) -> None:
        """Processes incoming data in real time.

        Motion data (codes 51 and 52) is converted into CW / CCW vectors, translated from pulses to Unity units, and
        is sent to Unity via MQTT. Encoder PPR data (code 53) is printed via console, so make sure the console is
        enabled. Also calculates the average running speed of the animal using 100 ms smoothing window and sends it to
        the central process via the _speed_tracker SharedMemoryArray.

        Notes:
            If debug mode is enabled, motion data is also converted to centimeters and printed via console.
        """
        # If the incoming message is the PPR report, prints the data via console.
        if message.event == 53:
            console.echo(f"Encoder ppr: {message.data_object}")

        # Otherwise, the message necessarily has to be reporting rotation into CCW or CW direction
        # (event code 51 or 52).

        # The rotation direction is encoded via the message event code. CW rotation (code 52) is interpreted as negative
        # and CCW (code 51) as positive.
        sign = 1 if message.event == np.uint8(51) else -1

        # Translates the absolute motion into the CW / CCW vector and converts from raw pulse count to Unity units
        # using the precomputed conversion factor. Uses float64 and rounds to 8 decimal places for consistency and
        # precision
        signed_motion = np.round(
            a=np.float64(message.data_object) * self._unity_unit_per_pulse * sign,
            decimals=8,
        )

        # Converts the motion into centimeters
        cm_motion = np.round(
            a=np.float64(message.data_object) * self._cm_per_pulse * sign,
            decimals=8,
        )

        # Aggregates all motion data into the _current_position attribute.
        self._current_position += cm_motion

        # Every 100 milliseconds, calculates the average running speed using the absolute difference between the current
        # and previous positions of the animal and updates the _speed_tracker with the data.
        elapsed_time = np.float64(self._speed_timer.elapsed)
        self._speed_timer.reset()  # Resets to start timing the next window while processing the data
        if elapsed_time > 100:
            position_change = np.abs(self._current_position - self._previous_position, dtype=np.float64)
            average_speed = np.float64(position_change / elapsed_time) * 1000  # In cm / s
            self._speed_tracker.write_data(index=0, data=average_speed)
            self._previous_position = self._current_position

        # If the class is in the debug mode, prints the motion data via console
        if self._debug:
            console.echo(message=f"Encoder moved {cm_motion} cm.")

        # Encodes the motion data into the format expected by the GIMBL Unity module and serializes it into a
        # byte-string.
        json_string = dumps(obj={"movement": signed_motion})
        byte_array = json_string.encode("utf-8")

        # Publishes the motion to the appropriate MQTT topic.
        self._communication.send_data(topic=self._motion_topic, payload=byte_array)  # type: ignore

    def parse_mqtt_command(self, topic: str, payload: bytes | bytearray) -> None:
        """Not used."""
        return

    def set_parameters(
        self,
        report_ccw: np.bool | bool = np.bool(True),
        report_cw: np.bool | bool = np.bool(True),
        delta_threshold: np.uint32 | int = np.uint32(10),
    ) -> None:
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
        """
        message = ModuleParameters(
            module_type=self._module_type,
            module_id=self._module_id,
            return_code=np.uint8(0),
            parameter_data=(np.bool(report_ccw), np.bool(report_cw), np.uint32(delta_threshold)),
        )
        self._input_queue.put(message)  # type: ignore

    def check_state(self, repetition_delay: np.uint32 = np.uint32(200)) -> None:
        """Returns the number of pulses accumulated by the EncoderModule since the last check or reset.

        If there has been a significant change in the absolute count of pulses, reports the change and direction to the
        PC. It is highly advised to issue this command to repeat (recur) at a desired interval to continuously monitor
        the encoder state, rather than repeatedly calling it as a one-off command for best runtime efficiency.

        This command allows continuously monitoring the rotation of the object connected to the encoder. It is designed
        to return the absolute raw count of pulses emitted by the encoder in response to the object ration. This allows
        avoiding floating-point arithmetic on the microcontroller and relies on the PC to convert pulses to standard
        units, such as centimeters. The specific conversion algorithm depends on the encoder and motion diameter.

        Args:
            repetition_delay: The time, in microseconds, to delay before repeating the command. If set to 0, the
                command will only run once.
        """
        if repetition_delay == 0:
            command = OneOffModuleCommand(
                module_type=self._module_type,
                module_id=self._module_id,
                return_code=np.uint8(0),
                command=np.uint8(1),
                noblock=np.bool(False),
            )
        else:
            command = RepeatedModuleCommand(
                module_type=self._module_type,
                module_id=self._module_id,
                return_code=np.uint8(0),
                command=np.uint8(1),
                noblock=np.bool(False),
                cycle_delay=np.uint32(repetition_delay),
            )
        self._input_queue.put(command)  # type: ignore

    def reset_pulse_count(self) -> None:
        """Resets the EncoderModule pulse tracker to 0.

        This command allows resetting the encoder without evaluating its current pulse count. Currently, this command
        is designed to only run once.
        """
        command = OneOffModuleCommand(
            module_type=self._module_type,
            module_id=self._module_id,
            return_code=np.uint8(0),
            command=np.uint8(2),
            noblock=np.bool(False),
        )

        self._input_queue.put(command)  # type: ignore

    def get_ppr(self) -> None:
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
        """
        command = OneOffModuleCommand(
            module_type=self._module_type,
            module_id=self._module_id,
            return_code=np.uint8(0),
            command=np.uint8(3),
            noblock=np.bool(False),
        )
        self._input_queue.put(command)  # type: ignore

    @property
    def mqtt_topic(self) -> str:
        """Returns the MQTT topic used to transfer motion data from the interface to Unity."""
        return self._motion_topic

    @property
    def cm_per_pulse(self) -> np.float64:
        """Returns the conversion factor to translate raw encoder pulse count to distance moved in centimeters."""
        return self._cm_per_pulse

    @property
    def speed_tracker(self) -> SharedMemoryArray:
        """Returns the SharedMemoryArray that stores the average running speed of the animal in centimeters per second.

        The running speed is computed over a window of 100 milliseconds. It is stored under the index 0 of the tracker
        and uses a float64 datatype.
        """
        return self._speed_tracker

    def parse_logged_data(self) -> tuple[NDArray[np.uint64], NDArray[np.float64]]:
        """Extracts and prepares the data acquired by the module during runtime for further analysis.

        Returns:
            A tuple with two elements. The first element is a numpy array that stores the timestamps, as microseconds
            elapsed since UTC epoch onset. The second element is a numpy array that stores the absolute position of the
            animal in centimeters at each timestamp, relative to the beginning of the VR track.
        """
        # Reads the data logged during runtime as a dictionary of dictionaries.
        log_data: dict[Any, list[dict[str, Any]]] = self.extract_logged_data()

        # Here, we only look for event-codes 51 (CCW displacement) and event-codes 52 (CW displacement).

        # Gets the data, defaulting to an empty list if the data is missing
        ccw_data = log_data.get(np.uint8(51), [])
        cw_data = log_data.get(np.uint8(52), [])

        # The way EncoderModule is implemented guarantees there is at least one CW code message with the displacement
        # of 0 that is received by the PC. In the worst case scenario, there will be no CCW codes and the parsing will
        # not work. To avoid that issue, we generate an artificial zero-code CCW value at the same timestamp + 1
        # microsecond as the original CW zero-code value. This does not affect the accuracy of our data, just makes the
        # code work for edge-cases.
        if not ccw_data:
            first_timestamp = cw_data[0]["timestamp"]
            ccw_data = [{"timestamp": first_timestamp + 1, "data": 0}]
        elif not cw_data:
            first_timestamp = ccw_data[0]["timestamp"]
            cw_data = [{"timestamp": first_timestamp + 1, "data": 0}]

        # Precreates the output arrays, based on the number of recorded CW and CCW displacements.
        total_length = len(ccw_data) + len(cw_data)
        timestamps: NDArray[np.uint64] = np.empty(total_length, dtype=np.uint64)
        displacements = np.empty(total_length, dtype=np.float64)

        # Processes CCW rotations (Code 51). CCW rotation is interpreted as positive displacement
        n_ccw = len(ccw_data)
        timestamps[:n_ccw] = [value["timestamp"] for value in ccw_data]  # Extracts timestamps for each value
        # The values are initially using the uint32 type. This converts them to float64 during the initial assignment
        displacements[:n_ccw] = [np.float64(value["data"]) for value in ccw_data]

        # Processes CW rotations (Code 52). CW rotation is interpreted as negative displacement
        timestamps[n_ccw:] = [value["timestamp"] for value in cw_data]  # CW data just fills remaining space after CCW.
        displacements[n_ccw:] = [-np.float64(value["data"]) for value in cw_data]

        # Sorts both arrays based on timestamps.
        sort_indices = np.argsort(timestamps)
        timestamps = timestamps[sort_indices]
        displacements = displacements[sort_indices]

        # Converts individual displacement vectors into aggregated absolute position of the mouse. The position is also
        # translated from encoder pulse counts into centimeters. The position is referenced to the start of the
        # experimental trial (beginning of the VR track) as 0-value. Positive positions mean moving forward along the
        # track, negative positions mean moving backward along the track.
        positions: NDArray[np.float64] = np.round(np.cumsum(displacements * self.cm_per_pulse), decimals=8)

        # Returns both timestamps and positions as numpy arrays.
        return timestamps, positions


class TTLInterface(ModuleInterface):
    """Interfaces with TTLModule instances running on Ataraxis MicroControllers.

    TTLModule facilitates exchanging Transistor-to-Transistor Logic (TTL) signals between various hardware systems, such
    as microcontrollers, cameras and recording devices. The module contains methods for both sending and receiving TTL
    pulses, but each TTLModule instance can only perform one of these functions at a time.

    Notes:
        When the TTLModule is configured to output a signal, it will notify the PC about the initial signal state
        (HIGH or LOW) after setup.

    Args:
        module_id: The unique byte-code identifier of the TTLModule instance. Since the mesoscope data acquisition
            pipeline uses multiple TTL modules on some microcontrollers, each instance running on the same
            microcontroller must have a unique identifier. The ID codes are not shared between AMC and other module
            types.
        report_pulses: A boolean flag that determines whether the class should report detecting HIGH signals to other
            processes. This is intended exclusively for the mesoscope frame acquisition recorder to notify the central
            process whether the mesoscope start trigger has been successfully received and processed by ScanImage
            software.
        debug: A boolean flag that configures the interface to dump certain data received from the microcontroller into
            the terminal. This is used during debugging and system calibration and should be disabled for most runtimes.

    Attributes:
        _report_pulses: Stores the report pulses flag.
        _debug: Stores the debug flag.
        _pulse_tracker: When the class is initialized with the report_pulses flag, stores the SharedMemoryArray used
            to transmit detected pulse status from the communication process back to the central process. Otherwise,
            this will be set to the None placeholder.
    """

    def __init__(self, module_id: np.uint8, report_pulses: bool = False, debug: bool = False) -> None:
        error_codes: set[np.uint8] = {np.uint8(51), np.uint8(54)}  # kOutputLocked, kInvalidPinMode
        # kInputOn, kInputOff, kOutputOn, kOutputOff
        # data_codes = {np.uint8(52), np.uint8(53), np.uint8(55), np.uint8(56)}

        self._debug: bool = debug
        self._report_pulses: bool = report_pulses

        # If the interface runs in the debug mode, configures the interface to monitor all incoming data codes.
        # Otherwise, the interface does not need to do any real-time processing of incoming data, so sets data_codes to
        # None.
        data_codes: set[np.uint8] | None = None
        if debug:
            data_codes = {np.uint8(52), np.uint8(53), np.uint8(55), np.uint8(56)}

        super().__init__(
            module_type=np.uint8(1),
            module_id=module_id,
            mqtt_communication=False,
            data_codes=data_codes,
            mqtt_command_topics=None,
            error_codes=error_codes,
        )

        # Precreates a shared memory array used to track and share the current received pulse status
        # (detected / not detected) with other processes. This tracking method is faster than using multiprocessing
        # queue, so it is preferred for time-critical application. Queue is easier to use, though, so we use it for
        # non-time-critical applications.
        self._pulse_tracker: SharedMemoryArray | None = None
        if report_pulses:
            self._pulse_tracker: SharedMemoryArray = SharedMemoryArray.create_array(
                name=f"{self._module_type}_{self._module_id}_pulse_tracker",
                prototype=np.empty(shape=1, dtype=np.uint8),
                exist_ok=True,
            )

    def initialize_remote_assets(self) -> None:
        """If the class is instructed to report detected HIGH incoming pulses, connects to the _pulse_tracker
        SharedMemoryArray.
        """
        if self._report_pulses and self._pulse_tracker is not None:
            self._pulse_tracker.connect()

    def terminate_remote_assets(self) -> None:
        """If the class is instructed to report detected HIGH incoming pulses, disconnects from the _pulse_tracker
        SharedMemoryArray.
        """
        if self._report_pulses and self._pulse_tracker is not None:
            self._pulse_tracker.disconnect()

    def process_received_data(self, message: ModuleData | ModuleState) -> None:
        """Processes incoming data when the class operates in debug or pulse reporting mode.

        During debug runtimes, this method dumps all received data into the terminal via the console class. During
        pulse reporting runtimes, the class sets the _pulse_tracker to 1 when it detects a HIGH incoming TTL pulse and
        to zero when it detects a LOW incoming TTL pulse.

        Notes:
            If the interface runs in debug mode, make sure the console is enabled, as it is used to print received
            data into the terminal.
        """
        if self._debug:
            if message.event == 52:
                console.echo(f"TTLModule {self.module_id} detects HIGH signal")
            if message.event == 53:
                console.echo(f"TTLModule {self.module_id} detects LOW signal")
            if message.event == 55:
                console.echo(f"TTLModule {self.module_id} emits HIGH signal")
            if message.event == 56:
                console.echo(f"TTLModule {self.module_id} emits LOW signal")

        if self._report_pulses:
            if message.event == 52:
                self._pulse_tracker.write_data(index=0, data=np.uint8(1))
            elif message.event == 52:
                self._pulse_tracker.write_data(index=0, data=np.uint8(0))

    def parse_mqtt_command(self, topic: str, payload: bytes | bytearray) -> None:
        """Not used."""
        return

    def set_parameters(
        self, pulse_duration: np.uint32 = np.uint32(10000), averaging_pool_size: np.uint8 = np.uint8(0)
    ) -> None:
        """Changes the PC-addressable runtime parameters of the TTLModule instance.

        Use this method to package and apply new PC-addressable parameters to the TTLModule instance managed by
        this Interface class.

        Args:
            pulse_duration: The duration, in microseconds, of each emitted TTL pulse HIGH phase. This determines
                how long the TTL pin stays ON when emitting a pulse.
            averaging_pool_size: The number of digital pin readouts to average together when checking pin state. This
                is used during the execution of the check_state () command to debounce the pin readout and acts in
                addition to any built-in debouncing.
        """
        message = ModuleParameters(
            module_type=self._module_type,
            module_id=self._module_id,
            return_code=np.uint8(0),
            parameter_data=(pulse_duration, averaging_pool_size),
        )
        self._input_queue.put(message)  # type: ignore

    def send_pulse(self, repetition_delay: np.uint32 = np.uint32(0), noblock: bool = True) -> None:
        """Triggers TTLModule to deliver a one-off or recurrent (repeating) digital TTL pulse.

        This command is well-suited to carry out most forms of TTL communication, but it is adapted for comparatively
        low-frequency communication at 10-200 Hz. This is in contrast to PWM outputs capable of mHz or even Khz pulse
        oscillation frequencies.

        Args:
            repetition_delay: The time, in microseconds, to delay before repeating the command. If set to 0, the command
                will only run once. The exact repetition delay will be further affected by other modules managed by the
                same microcontroller and may not be perfectly accurate.
            noblock: Determines whether the command should block the microcontroller while emitting the high phase of
                the pulse or not. Blocking ensures precise pulse duration, non-blocking allows the microcontroller to
                perform other operations while waiting, increasing its throughput.
        """
        if repetition_delay == 0:
            command = OneOffModuleCommand(
                module_type=self._module_type,
                module_id=self._module_id,
                return_code=np.uint8(0),
                command=np.uint8(1),
                noblock=np.bool(noblock),
            )
        else:
            command = RepeatedModuleCommand(
                module_type=self._module_type,
                module_id=self._module_id,
                return_code=np.uint8(0),
                command=np.uint8(1),
                noblock=np.bool(noblock),
                cycle_delay=repetition_delay,
            )

        self._input_queue.put(command)  # type: ignore

    def toggle(self, state: bool) -> None:
        """Triggers the TTLModule to continuously deliver a digital HIGH or LOW signal.

        This command locks the TTLModule managed by this Interface into delivering the desired logical signal.

        Args:
            state: The signal to output. Set to True for HIGH and False for LOW.
        """
        command = OneOffModuleCommand(
            module_type=self._module_type,
            module_id=self._module_id,
            return_code=np.uint8(0),
            command=np.uint8(2 if state else 3),
            noblock=np.bool(False),
        )

        self._input_queue.put(command)  # type: ignore

    def check_state(self, repetition_delay: np.uint32 = np.uint32(0)) -> None:
        """Checks the state of the TTL signal received by the TTLModule.

        This command evaluates the state of the TTLModule's input pin and, if it is different from the previous state,
        reports it to the PC. This approach ensures that the module only reports signal level shifts (edges), preserving
        communication bandwidth.

        Args:
            repetition_delay: The time, in microseconds, to delay before repeating the command. If set to 0, the command
                will only run once.
        """
        if repetition_delay == 0:
            command = OneOffModuleCommand(
                module_type=self._module_type,
                module_id=self._module_id,
                return_code=np.uint8(0),
                command=np.uint8(4),
                noblock=np.bool(False),
            )
        else:
            command = RepeatedModuleCommand(
                module_type=self._module_type,
                module_id=self._module_id,
                return_code=np.uint8(0),
                command=np.uint8(4),
                noblock=np.bool(False),
                cycle_delay=repetition_delay,
            )
        self._input_queue.put(command)  # type: ignore

    def parse_logged_data(self) -> NDArray[np.uint64]:
        """Extracts and prepares the data acquired by the module during runtime for further analysis.

        Notes:
            The returned array of timestamps is used as the 'seed' for the interpolation step of data alignment if
            this method is called for the TTLModuleInterface used to monitor mesoscope frame acquisition stamps.

            This parsing code includes the 'blip' correction. If the first detected pulse is less than 10 ms duration,
            the corresponding event is removed from the returned array. This is used to filter the 'blip' associated
            with starting the mesoscope frame acquisition.

        Returns:
            A numpy array that stores the timestamps for the beginning of each HIGH TTL phase event detected by the
            module. The timestamps correspond to the time when the LOW phase translates into the HIGH phase
            (rising edges). Currently, this method is only called for the TTLModule that monitors mesoscope frame
            acquisition stamps and, therefore, the timestamps denote the time when the mesoscope starts acquiring
            (scanning) each frame. If the module received no TTL pulses during runtime, this method will return an
            empty array.
        """
        # Reads the data logged during runtime as a dictionary of dictionaries.
        log_data: dict[Any, list[dict[str, Any]]] = self.extract_logged_data()

        # Here, we only look for event-codes 52 (InputON) and event-codes 53 (InputOFF).

        # Gets the data for both message types. The way the module is written guarantees that the PC receives code 53
        # at least once. No such guarantee is made for code 52, however. We still default to empty lists for both
        # to make this code a bit friendlier to future changes.
        on_data = log_data.get(np.uint8(52), [])
        off_data = log_data.get(np.uint8(53), [])

        # Since this code ultimately looks for rising edges, it will not find any unless there is at least one ON and
        # one OFF message. Therefore, if any of the codes is actually missing, shorts to returning an empty array.
        if len(on_data) == 0 or len(off_data) == 0:
            return np.array([], dtype=np.uint64)

        # Determines the total length of the output array using the length of ON and OFF data arrays.
        total_length = len(on_data) + len(off_data)

        # Precreates the storage numpy arrays for both message types. Timestamps use uint64 datatype and the trigger
        # values are boolean. We use uint8 as it has the same memory footprint as a boolean and allows us to use integer
        # types across the entire dataset.
        timestamps: NDArray[np.uint64] = np.empty(total_length, dtype=np.uint64)
        triggers: NDArray[np.uint8] = np.empty(total_length, dtype=np.uint8)

        # Extracts ON (Code 52) trigger codes. Statically assigns the value '1' to denote ON signals.
        n_on = len(on_data)
        timestamps[:n_on] = [value["timestamp"] for value in on_data]
        triggers[:n_on] = np.uint8(1)  # All code 52 signals are ON (High)

        # Extracts OFF (Code 53) trigger codes.
        timestamps[n_on:] = [value["timestamp"] for value in off_data]
        triggers[n_on:] = np.uint8(0)  # All code 53 signals are OFF (Low)

        # Sorts both arrays based on the timestamps, so that the data is in the chronological order.
        sort_indices = np.argsort(timestamps)
        timestamps = timestamps[sort_indices]
        triggers = triggers[sort_indices]

        # Finds falling edges (where the signal goes from 1 to 0). Then uses the indices for such events to extract the
        # timestamps associated with each falling edge, before returning them to the caller.
        # falling_edges = np.where((triggers[:-1] == 1) & (triggers[1:] == 0))[0] + 1

        # Recently we switched to using the rising edges instead of falling edges. The purpose and code are very
        # similar, though.
        rising_edges = np.where((triggers[:-1] == 0) & (triggers[1:] == 1))[0] + 1
        frame_timestamps = timestamps[rising_edges]

        # Calculate pulse durations by looking ahead to the next falling edge
        # pulse_durations = (timestamps[rising_edges + 1] - timestamps[rising_edges]).astype(np.float64)

        # Determines the durations of all detected pulses. This is needed to filter out the 'blip' in the mesoscope
        # frame stamps. The blip pulse is usually under 5 ms, vs. a real frame pulse that is ~100 ms, and it happens
        # at the very beginning of the mesoscope acquisition sequence. Since timestamps alternate rising and falling
        # edges, rising edge + 1 corresponds to the falling edge of that pulse.
        pulse_durations = (timestamps[rising_edges + 1] - timestamps[rising_edges]).astype(np.float64)

        # If the very first recorded pulse has a duration below 10 ms, removes the pulse from the returned array
        if pulse_durations[0] < 10000:  # The timestamps use microseconds, so the check uses 10,000 us
            frame_timestamps = frame_timestamps[1:]

        return frame_timestamps

    @property
    def pulse_status(self) -> bool:
        """Returns the current status of the incoming TTL pulse.

        If the TTLModule receives the HIGH phase of the incoming TTL pulse, it returns True. Otherwise, returns False.
        """

        # If pulse tracking is disabled, always returns False.
        if self._pulse_tracker is None:
            return False

        if self._pulse_tracker.read_data(index=0) == 1:
            return True
        else:
            return False


class BreakInterface(ModuleInterface):
    """Interfaces with BreakModule instances running on Ataraxis MicroControllers.

    BreakModule allows interfacing with a break to dynamically control the motion of break-coupled objects. The module
    is designed to send PWM signals that trigger Field-Effect-Transistor (FET) gated relay hardware to deliver voltage
    that variably engages the break. The module can be used to either fully engage or disengage the breaks or to output
    a PWM signal to engage the break with the desired strength.

    Notes:
        The break will notify the PC about its initial state (Engaged or Disengaged) after setup.

        This class is explicitly designed to work with an 8-bit Pulse Width Modulation (PWM) resolution. Specifically,
        it assumes that there are a total of 255 intervals covered by the whole PWM range when it calculates conversion
        factors to go from PWM levels to torque and force.

    Args:
        minimum_break_strength: The minimum torque applied by the break in gram centimeter. This is the torque the
            break delivers at minimum voltage (break is disabled).
        maximum_break_strength: The maximum torque applied by the break in gram centimeter. This is the torque the
            break delivers at maximum voltage (break is fully engaged).
        object_diameter: The diameter of the rotating object connected to the break, in centimeters. This is used to
            calculate the force at the end of the object associated with each torque level of the break.
        debug: A boolean flag that configures the interface to dump certain data received from the microcontroller into
            the terminal. This is used during debugging and system calibration and should be disabled for most runtimes.

    Attributes:
        _newton_per_gram_centimeter: Conversion factor from torque force in g cm to torque force in N cm.
        _minimum_break_strength: The minimum torque the break delivers at minimum voltage (break is disabled) in N cm.
        _maximum_break_strength: The maximum torque the break delivers at maximum voltage (break is fully engaged) in N
            cm.
        _torque_per_pwm: Conversion factor from break pwm levels to breaking torque in N cm.
        _force_per_pwm: Conversion factor from break pwm levels to breaking force in N at the edge of the object.
        _debug: Stores the debug flag.
    """

    def __init__(
        self,
        minimum_break_strength: float = 43.2047,  # 0.6 oz in
        maximum_break_strength: float = 1152.1246,  # 16 oz in
        object_diameter: float = 15.0333,
        debug: bool = False,
    ) -> None:
        error_codes: set[np.uint8] = {np.uint8(51)}  # kOutputLocked
        # data_codes = {np.uint8(52), np.uint8(53), np.uint8(54)}  # kEngaged, kDisengaged, kVariable

        self._debug: bool = debug

        # If the interface runs in the debug mode, configures the interface to monitor engaged and disengaged codes.
        data_codes: set[np.uint8] | None = None
        if debug:
            data_codes = {np.uint8(52), np.uint8(53)}

        # Initializes the subclassed ModuleInterface using the input instance data. Type data is hardcoded.
        super().__init__(
            module_type=np.uint8(3),
            module_id=np.uint8(1),
            mqtt_communication=False,
            data_codes=data_codes,
            mqtt_command_topics=None,
            error_codes=error_codes,
        )

        # Hardcodes the conversion factor used to translate torque force in g cm to N cm
        self._newton_per_gram_centimeter: float = 0.00981

        # Converts minimum and maximum break strength into Newton centimeter
        self._minimum_break_strength: np.float64 = np.round(
            a=minimum_break_strength * self._newton_per_gram_centimeter,
            decimals=8,
        )
        self._maximum_break_strength: np.float64 = np.round(
            a=maximum_break_strength * self._newton_per_gram_centimeter,
            decimals=8,
        )

        # Computes the conversion factor to translate break pwm levels into breaking torque in Newtons cm. Rounds
        # to 12 decimal places for consistency and to ensure repeatability.
        self._torque_per_pwm: np.float64 = np.round(
            a=(self._maximum_break_strength - self._minimum_break_strength) / 255,
            decimals=8,
        )

        # Also computes the conversion factor to translate break pwm levels into force in Newtons. To overcome the
        # breaking torque, the object has to experience that much force applied to its edge.
        self._force_per_pwm: np.float64 = np.round(
            a=self._torque_per_pwm / (object_diameter / 2),
            decimals=8,
        )

    def initialize_remote_assets(self) -> None:
        """Not used."""
        pass

    def terminate_remote_assets(self) -> None:
        """Not used."""
        return

    def process_received_data(self, message: ModuleData | ModuleState) -> None:
        """During debug runtime, dumps the data received from the module into the terminal.

        Currently, this method only works with codes 52 (Engaged) and 53 (Disengaged).

        Notes:
            The method is not used during non-debug runtimes. If the interface runs in debug mode, make sure the
            console is enabled, as it is used to print received data into the terminal.
        """
        # The method is ONLY called during debug runtime, so prints all received data via console.
        if message.event == 52:
            console.echo(f"Break is engaged")
        if message.event == 53:
            console.echo(f"Break is disengaged")

    def parse_mqtt_command(self, topic: str, payload: bytes | bytearray) -> None:
        """Not used."""
        return

    def set_parameters(self, breaking_strength: np.uint8 = np.uint8(255)) -> None:
        """Changes the PC-addressable runtime parameters of the BreakModule instance.

        Use this method to package and apply new PC-addressable parameters to the BreakModule instance managed by this
        Interface class.

        Notes:
            Use set_breaking_power() command to apply the breaking-strength transmitted in this parameter message to the
            break. Until the command is called, the new breaking_strength will not be applied to the break hardware.

        Args:
            breaking_strength: The Pulse-Width-Modulation (PWM) value to use when the BreakModule delivers adjustable
                breaking power. Depending on this value, the breaking power can be adjusted from none (0) to maximum
                (255). Use get_pwm_from_force() to translate the desired breaking torque into the required PWM value.
        """
        message = ModuleParameters(
            module_type=self._module_type,
            module_id=self._module_id,
            return_code=np.uint8(0),  # Generally, return code is only helpful for debugging.
            parameter_data=(breaking_strength,),
        )
        self._input_queue.put(message)  # type: ignore

    def toggle(self, state: bool) -> None:
        """Triggers the BreakModule to be permanently engaged at maximum strength or permanently disengaged.

        This command locks the BreakModule managed by this Interface into the desired state.

        Notes:
            This command does NOT use the breaking_strength parameter and always uses either maximum or minimum breaking
            power. To set the break to a specific torque level, set the level via the set_parameters() method and then
            switch the break into the variable torque mode by using the set_breaking_power() method.

        Args:
            state: The desired state of the break. True means the break is engaged; False means the break is disengaged.
        """
        command = OneOffModuleCommand(
            module_type=self._module_type,
            module_id=self._module_id,
            return_code=np.uint8(0),
            command=np.uint8(1 if state else 2),
            noblock=np.bool(False),
        )
        self._input_queue.put(command)  # type: ignore

    def set_breaking_power(self) -> None:
        """Triggers the BreakModule to engage with the strength (torque) defined by the breaking_strength runtime
        parameter.

        Unlike the toggle() method, this method allows precisely controlling the torque applied by the break. This
        is achieved by pulsing the break control pin at the PWM level specified by breaking_strength runtime parameter
        stored in BreakModule's memory (on the microcontroller).

        Notes:
            This command switches the break to run in the variable strength mode and applies the current value of the
            breaking_strength parameter to the break, but it does not determine the breaking power. To adjust the power,
            use the set_parameters() class method to issue an updated breaking_strength value. By default, the break
            power is set to 50% (PWM value 128).
        """
        command = OneOffModuleCommand(
            module_type=self._module_type,
            module_id=self._module_id,
            return_code=np.uint8(0),
            command=np.uint8(3),
            noblock=np.bool(False),
        )
        self._input_queue.put(command)  # type: ignore

    def get_pwm_from_torque(self, target_torque_n_cm: float) -> np.uint8:
        """Converts the desired breaking torque in Newtons centimeter to the required PWM value (0-255) to be delivered
        to the break hardware by the BreakModule.

        Use this method to convert the desired breaking torque into the PWM value that can be submitted to the
        BreakModule via the set_parameters() class method.

        Args:
            target_torque_n_cm: Desired torque in Newtons centimeter at the edge of the object.

        Returns:
            The byte PWM value that would generate the desired amount of torque.

        Raises:
            ValueError: If the input force is not within the valid range for the BreakModule.
        """
        if self._maximum_break_strength < target_torque_n_cm or self._minimum_break_strength > target_torque_n_cm:
            message = (
                f"The requested torque {target_torque_n_cm} N cm is outside the valid range for the BreakModule "
                f"{self._module_id}. Valid breaking torque range is from {self._minimum_break_strength} to "
                f"{self._maximum_break_strength}."
            )
            console.error(message=message, error=ValueError)

        # Calculates PWM using the pre-computed torque_per_pwm conversion factor
        pwm_value = np.uint8(round((target_torque_n_cm - self._minimum_break_strength) / self._torque_per_pwm))

        return pwm_value

    @property
    def torque_per_pwm(self) -> np.float64:
        """Returns the conversion factor to translate break pwm levels into breaking torque in Newton centimeters."""
        return self._torque_per_pwm

    @property
    def force_per_pwm(self) -> np.float64:
        """Returns the conversion factor to translate break pwm levels into breaking force in Newtons."""
        return self._force_per_pwm

    def parse_logged_data(self) -> tuple[NDArray[np.uint64], NDArray[np.float64]]:
        """Extracts and prepares the data acquired by the module during runtime for further analysis.

        Notes:
            This method assumes that the break was used in the absolute force mode. It does not extract variable
            breaking power data.

        Returns:
            A tuple with two elements. The first element is a numpy array that stores the timestamps, as microseconds
            elapsed since UTC epoch onset. The second element is a numpy array that stores the torque applied by the
            break to the running wheel at each timestamp in Newton centimeters.
        """
        # Reads the data logged during runtime as a dictionary of dictionaries.
        log_data: dict[Any, list[dict[str, Any]]] = self.extract_logged_data()

        # Here, we only look for event-codes 52 (Engaged) and event-codes 53 (Disengaged) as no experiment requires
        # variable breaking power. If we ever use variable breaking power, this section would need to be expanded to
        # allow parsing code 54 events.

        # Gets the data, defaulting to an empty list if the data is missing
        engaged_data = log_data.get(np.uint8(52), [])
        disengaged_data = log_data.get(np.uint8(53), [])

        # Precreates the storage numpy arrays for both message types. Timestamps use uint64 datatype. Although trigger
        # values are boolean, we translate them into the actual torque applied by the break in Newton centimeters and
        # store them as float 64 values.
        total_length = len(engaged_data) + len(disengaged_data)
        timestamps: NDArray[np.uint64] = np.empty(total_length, dtype=np.uint64)
        torques: NDArray[np.float64] = np.empty(total_length, dtype=np.float64)

        # Processes Engaged (code 52) triggers. When the motor is engaged, it applies the maximum possible torque to
        # the break.
        n_engaged = len(engaged_data)
        timestamps[:n_engaged] = [value["timestamp"] for value in engaged_data]  # Extracts timestamps for each value
        # Since engaged strength means that the torque is delivering maximum force, uses the maximum force in N cm as
        # the torque value for each 'engaged' state.
        torques[:n_engaged] = [self._maximum_break_strength for _ in engaged_data]  # Already in rounded float 64

        # Processes Disengaged (code 53) triggers. Contrary to naive expectation, the torque of a disengaged break is
        # NOT zero. Instead, it is at least the same as the minimum break strength, likely larger due to all mechanical
        # couplings in the system.
        timestamps[n_engaged:] = [value["timestamp"] for value in disengaged_data]
        torques[n_engaged:] = [self._minimum_break_strength for _ in disengaged_data]  # Already in rounded float 64

        # Sorts both arrays based on timestamps.
        sort_indices = np.argsort(timestamps)
        timestamps = timestamps[sort_indices]
        torques = torques[sort_indices]

        return timestamps, torques


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
            initialize the class using a placeholder calibration tuple and use the calibration() class method to
            collect this data using the ValveModule.
        debug: A boolean flag that configures the interface to dump certain data received from the microcontroller into
            the terminal. This is used during debugging and system calibration and should be disabled for most runtimes.

    Attributes:
        _scale_coefficient: Stores the scale coefficient derived from the calibration data. We use the power law to
            fit the data, which results in better overall fit than using the linera equation.
        _nonlinearity_exponent: The intercept of the valve calibration curve. This is used to account for the fact that
            some valves may have a minimum open time or dispensed fluid volume, which is captured by the intercept.
            This improves the precision of fluid-volume-to-valve-open-time conversions.
        _calibration_cov
        _reward_topic: Stores the topic used by Unity to issue reward commands to the module.
        _debug: Stores the debug flag.
        _reward_tracker: Stores the SharedMemoryArray that tracks the current valve status and the total volume of
            water dispensed by the valve.
        _cycle_timer: A PrecisionTimer instance initialized in the Communication process to track how long the valve
            stays open during cycling.
    """

    def __init__(
        self, valve_calibration_data: tuple[tuple[int | float, int | float], ...], debug: bool = False
    ) -> None:
        error_codes: set[np.uint8] = {np.uint8(51)}  # kOutputLocked
        data_codes = {np.uint8(52), np.uint8(53), np.uint8(54)}  # kOpen, kClosed, kCalibrated
        mqtt_command_topics: set[str] = {"Gimbl/Reward/"}

        self._debug: bool = debug

        # If the interface runs in the debug mode, expands the list of processed data codes to include all codes used
        # by the valve module.
        if debug:
            data_codes = {np.uint8(52), np.uint8(53), np.uint8(54)}

        super().__init__(
            module_type=np.uint8(5),
            module_id=np.uint8(1),
            mqtt_communication=True,
            data_codes=data_codes,
            mqtt_command_topics=mqtt_command_topics,
            error_codes=error_codes,
        )

        # Extracts pulse durations and fluid volumes into separate arrays
        pulse_durations: NDArray[np.float64] = np.array([x[0] for x in valve_calibration_data], dtype=np.float64)
        fluid_volumes: NDArray[np.float64] = np.array([x[1] for x in valve_calibration_data], dtype=np.float64)

        # Defines the power-law model. Our calibration data suggests that the Valve performs in a non-linear fashion
        # and is better calibrated using the power law, rather than a linear fit
        def power_law_model(pulse_duration, a, b):
            return a * np.power(pulse_duration, b)

        # Fits the power-law model to the input calibration data and saves the fit parameters and covariance matrix to
        # class attributes
        # noinspection PyTupleAssignmentBalance
        params, fit_cov_matrix = curve_fit(f=power_law_model, xdata=pulse_durations, ydata=fluid_volumes)
        scale_coefficient, nonlinearity_exponent = params
        self._calibration_cov: NDArray[np.float64] = fit_cov_matrix
        self._scale_coefficient: np.float64 = np.round(a=np.float64(scale_coefficient), decimals=8)
        self._nonlinearity_exponent: np.float64 = np.round(a=np.float64(nonlinearity_exponent), decimals=8)

        # Stores the reward topic separately to make it accessible via property
        self._reward_topic: str = "Gimbl/Reward/"

        # Precreates a shared memory array used to track and share valve state data. Index 1 is used to report the
        # current valve state (open or closed). Index 1 tracks the total amount of water dispensed by the valve
        self._reward_tracker: SharedMemoryArray = SharedMemoryArray.create_array(
            name=f"{self._module_type}_{self._module_id}_reward_tracker",
            prototype=np.empty(shape=2, dtype=np.float64),
            exist_ok=True,
        )

        # Placeholder
        self._cycle_timer: PrecisionTimer | None = None

    def __del__(self) -> None:
        """Ensures the reward_tracker is properly cleaned up when the class is garbage-collected."""
        self._reward_tracker.disconnect()
        self._reward_tracker.destroy()

    def initialize_remote_assets(self) -> None:
        """Connects to the reward tracker SharedMemoryArray and initializes the cycle PrecisionTimer from the
        Communication process."""
        self._reward_tracker.connect()
        self._cycle_timer = PrecisionTimer("us")

    def terminate_remote_assets(self) -> None:
        """Disconnects from the reward tracker SharedMemoryArray."""
        self._reward_tracker.disconnect()

    def process_received_data(self, message: ModuleData | ModuleState) -> None:
        """Processes incoming data.

        Valve calibration events (code 54) are sent to the terminal via console regardless of the debug flag. If the
        class was initialized in the debug mode, Valve opening (code 52) and closing (code 52) codes are also sent to
        the terminal. Also, updates the reward_tracker state (index 0) each time the valve sends a new state message
        and for each Open and Close cycle updates the total volume of dispensed waters stored under index 1 of the
        reward_tracker.

        Note:
            Make sure the console is enabled before calling this method.
        """

        # Extracts the previous valve state from the storage array
        previous_state = bool(self._reward_tracker.read_data(index=0, convert_output=True))

        if message.event == 54:
            console.echo(f"Valve Calibration: Complete")
        elif message.event == 52:
            if self._debug:
                console.echo(f"Valve Opened")

            # Updates the current valve state in the storage array
            self._reward_tracker.write_data(index=0, data=np.float64(1))

            # Resets the cycle timer each time the valve transitions from closed to open
            if not previous_state:
                self._cycle_timer.reset()

        elif message.event == 53:
            if self._debug:
                console.echo(f"Valve Closed")

            # Updates the current valve state in the storage array
            self._reward_tracker.write_data(index=0, data=np.float64(0))

            # Each time the valve transitions from open to closed state, records the period of time the valve was open
            # and uses it to estimate the volume of fluid delivered through the valve. Accumulates the total volume in
            # the tracker array.
            if previous_state:
                open_duration = self._cycle_timer.elapsed
                delivered_volume = np.float64(
                    self._scale_coefficient * np.power(open_duration, self._nonlinearity_exponent)
                )
                self._reward_tracker.write_data(index=1, data=delivered_volume)

    def parse_mqtt_command(self, topic: str, payload: bytes | bytearray) -> None:
        """When called, this method statically sends a reward delivery command to the ValveModule instance.

        Notes:
            The method does NOT evaluate the input message or topic. It is written to always send reward trigger
            commands when called. If future Sun lab pipelines need this method to evaluate the input message, the logic
            of the method needs to be rewritten.
        """

        # Currently, the only message that can be processed by this method is the reward trigger message from Unity.
        # Therefore, whenever this method is triggerd, regardless of the input message, sends a reward delivery
        # command to the ValveModule instance.
        command = OneOffModuleCommand(
            module_type=self._module_type,
            module_id=self._module_id,
            return_code=np.uint8(0),
            command=np.uint8(1),
            noblock=np.bool(False),  # Blocks to ensure reward delivery precision.
        )
        self._input_queue.put(command)  # type: ignore

    def set_parameters(
        self,
        pulse_duration: np.uint32 = np.uint32(35590),
        calibration_delay: np.uint32 = np.uint32(200000),
        calibration_count: np.uint16 = np.uint16(200),
    ) -> None:
        """Changes the PC-addressable runtime parameters of the ValveModule instance.

        Use this method to package and apply new PC-addressable parameters to the ValveModule instance managed by this
        Interface class.

        Note:
            Default parameters are configured to support 'reference' calibration run. When calibrate() is called with
            these default parameters, the Valve should deliver ~5 uL of water, which is the value used during Sun lab
            experiments. If the reference calibration fails, you have to fully recalibrate the valve!

        Args:
            pulse_duration: The time, in microseconds, the valve stays open when it is pulsed (opened and closed). This
                is used during the execution of the send_pulse() command to control the amount of dispensed fluid. Use
                the get_duration_from_volume() method to convert the desired fluid volume into the pulse_duration value.
            calibration_delay: The time, in microseconds, to wait between consecutive pulses during calibration.
                Calibration works by repeatedly pulsing the valve the requested number of times. Delaying after closing
                the valve (ending the pulse) ensures the valve hardware has enough time to respond to the inactivation
                phase before starting the next calibration cycle.
            calibration_count: The number of times to pulse the valve during calibration. A number between 10 and 100 is
                enough for most use cases.
        """
        message = ModuleParameters(
            module_type=self._module_type,
            module_id=self._module_id,
            return_code=np.uint8(0),
            parameter_data=(pulse_duration, calibration_delay, calibration_count),
        )
        self._input_queue.put(message)  # type: ignore

    def send_pulse(self, repetition_delay: np.uint32 = np.uint32(0), noblock: bool = False) -> None:
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
                not. Blocking ensures precise pulse duration and dispensed fluid volume. Non-blocking
                allows the microcontroller to perform other operations while waiting, increasing its throughput.
        """
        if repetition_delay == 0:
            command = OneOffModuleCommand(
                module_type=self._module_type,
                module_id=self._module_id,
                return_code=np.uint8(0),
                command=np.uint8(1),
                noblock=np.bool(noblock),
            )
        else:
            command = RepeatedModuleCommand(
                module_type=self._module_type,
                module_id=self._module_id,
                return_code=np.uint8(0),
                command=np.uint8(1),
                noblock=np.bool(noblock),
                cycle_delay=repetition_delay,
            )
        self._input_queue.put(command)  # type: ignore

    def toggle(self, state: bool) -> None:
        """Triggers the ValveModule to be permanently open or closed.

        This command locks the ValveModule managed by this Interface into the desired state.

        Args:
            state: The desired state of the valve. True means the valve is open; False means the valve is closed.
        """
        command = OneOffModuleCommand(
            module_type=self._module_type,
            module_id=self._module_id,
            return_code=np.uint8(0),
            command=np.uint8(2 if state else 3),
            noblock=np.bool(False),
        )
        self._input_queue.put(command)  # type: ignore

    def calibrate(self) -> None:
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
        """
        command = OneOffModuleCommand(
            module_type=self._module_type,
            module_id=self._module_id,
            return_code=np.uint8(0),
            command=np.uint8(4),
            noblock=np.bool(False),
        )
        self._input_queue.put(command)  # type: ignore

    def get_duration_from_volume(self, target_volume: float) -> np.uint32:
        """Converts the desired fluid volume in microliters to the valve pulse duration in microseconds that ValveModule
        will use to deliver that fluid volume.

        Use this method to convert the desired fluid volume into the pulse_duration value that can be submitted to the
        ValveModule via the set_parameters() class method.

        Args:
            target_volume: Desired fluid volume in microliters.

        Raises:
            ValueError: If the desired fluid volume is too small to be reliably dispensed by the valve, based on its
                calibration data.

        Returns:
            The microsecond pulse duration that would be used to deliver the specified volume.
        """
        # Determines the minimum valid pulse duration. We hardcode this at 10 ms as this is the lower calibration
        # boundary
        min_pulse_duration = 10.0  # microseconds
        min_dispensed_volume = self._scale_coefficient * np.power(min_pulse_duration, self._nonlinearity_exponent)

        if target_volume < min_dispensed_volume:
            message = (
                f"The requested volume {target_volume} uL is too small to be reliably dispensed by the ValveModule "
                f"{self._module_id}. Specifically, the smallest volume of fluid the valve can reliably dispense is "
                f"{min_dispensed_volume} uL."
            )
            console.error(message=message, error=ValueError)

        # Inverts the power-law calibration to get the pulse duration.
        pulse_duration = (target_volume / self._scale_coefficient) ** (1.0 / self._nonlinearity_exponent)

        return np.uint32(np.round(pulse_duration))

    @property
    def mqtt_topic(self) -> str:
        """Returns the MQTT topic monitored by the module to receive reward commands from Unity."""
        return self._reward_topic

    @property
    def scale_coefficient(self) -> np.float64:
        """Returns the scaling coefficient (A) from the power‐law calibration.

        In the calibration model, fluid_volume = A * (pulse_duration)^B, this coefficient
        converts pulse duration (in microseconds) into the appropriate fluid volume (in microliters)
        when used together with the nonlinearity exponent.
        """
        return self._scale_coefficient

    @property
    def nonlinearity_exponent(self) -> np.float64:
        """Returns the nonlinearity exponent (B) from the power‐law calibration.

        In the calibration model, fluid_volume = A * (pulse_duration)^B, this exponent indicates
        the degree of nonlinearity in how the dispensed volume scales with the valve’s pulse duration.
        For example, an exponent of 1 would indicate a linear relationship.
        """
        return self._nonlinearity_exponent

    @property
    def calibration_covariance(self) -> np.ndarray:
        """mReturns the 2x2 covariance matrix associated with the power‐law calibration fit.

        The covariance matrix contains the estimated variances of the calibration parameters
        on its diagonal (i.e., variance of the scale coefficient and the nonlinearity exponent)
        and the covariances between these parameters in its off-diagonal elements.

        This information can be used to assess the uncertainty in the calibration.

        Returns:
            A NumPy array (2x2) representing the covariance matrix.
        """
        return self._calibration_cov

    @property
    def delivered_volume(self) -> float:
        """Returns the total volume of water, in microliters, delivered by the valve during the current runtime."""
        return self._reward_tracker.read_data(index=1, convert_output=True)

    @property
    def reward_tracker(self) -> SharedMemoryArray:
        """Returns the SharedMemoryArray that stores the current valve state and the total volume of water delivered
        during the current runtime.

        The valve state is stored under index 0, while the total delivered volume is stored under index 1. Both values
        are stored as a float64 datatype. The valve state is 1 when the valve is open and 0 otherwise. The total
        delivered volume is given in microliters.
        """
        return self._reward_tracker

    def parse_logged_data(self) -> tuple[NDArray[np.uint64], NDArray[np.float64]]:
        """Extracts and prepares the data acquired by the module during runtime for further analysis.

        Returns:
            A tuple with two elements. The first element is a numpy array that stores the timestamps, as microseconds
            elapsed since UTC epoch onset. The second element is a numpy array that stores the total (aggregated) volume
            of water in microliters received by the animal since the beginning of the experiment, at each timestamp.
        """
        # Reads the data logged during runtime as a dictionary of dictionaries.
        log_data: dict[Any, list[dict[str, Any]]] = self.extract_logged_data()

        # Here, we only look for event-codes 52 (Valve Open) and event-codes 53 (Valve Closed).

        # The way this module is implemented guarantees there is at least one code 53 message, but there may be no code
        # 52 messages.
        open_data = log_data.get(np.uint8(52), [])
        closed_data = log_data[np.uint8(53)]

        # If there were no valve open events, no water was dispensed. In this case, uses the first code 53 timestamp
        # to report zero-volume reward and ends the runtime early.
        if not open_data:
            return np.array([closed_data[0]["timestamp"]], dtype=np.uint64), np.array([0], dtype=np.float64)

        # Precreates the storage numpy arrays for both message types. Timestamps use uint64 datatype. Although valve
        # trigger values are boolean, we translate them into the total volume of water, in microliters, dispensed to the
        # animal at each time-point and store that value as a float64.
        total_length = len(open_data) + len(closed_data)
        timestamps: NDArray[np.uint64] = np.empty(total_length, dtype=np.uint64)
        volume: NDArray[np.float64] = np.empty(total_length, dtype=np.float64)

        # The water is dispensed gradually while the valve stays open. Therefore, the full reward volume is dispensed
        # when the valve goes from open to closed. Based on calibration data, we have a conversion factor to translate
        # the time the valve remains open into the fluid volume dispensed to the animal, which we use to convert each
        # Open/Close cycle duration into the dispensed volume.

        # Extracts Open (Code 52) trigger codes. Statically assigns the value '1' to denote Open signals.
        n_on = len(open_data)
        timestamps[:n_on] = [value["timestamp"] for value in open_data]
        volume[:n_on] = np.uint8(1)  # All code 52 signals are Open (High)

        # Extracts Closed (Code 53) trigger codes.
        timestamps[n_on:] = [value["timestamp"] for value in closed_data]
        volume[n_on:] = np.uint8(0)  # All code 53 signals are Closed (Low)

        # Sorts both arrays based on timestamps.
        sort_indices = np.argsort(timestamps)
        timestamps = timestamps[sort_indices]
        volume = volume[sort_indices]

        # Find falling edges (valve closing events)
        falling_edges = np.where((volume[:-1] == 1) & (volume[1:] == 0))[0] + 1

        # Samples the timestamp array to only include timestamps for the falling edges. That is, the timestamps for
        # when the valve has fully delivered the requested volume of water.
        reward_timestamps = timestamps[falling_edges]

        # Calculates pulse durations in microseconds for each open-close cycle. Since the original timestamp array
        # contains alternating HIGH / LOW edges, falling edge - 1 corresponds to the rising edge of that very same
        # pulse.
        pulse_durations: NDArray[np.float64] = (timestamps[falling_edges] - timestamps[falling_edges - 1]).astype(
            np.float64
        )

        # Converts the time the Valve stayed open into the dispensed water volume, in microliters.
        # noinspection PyTypeChecker
        volumes: NDArray[np.float64] = np.round(
            np.cumsum(self._scale_coefficient * np.power(pulse_durations, self._nonlinearity_exponent)),
            decimals=8,
        )

        # Returns processed data to caller.
        return reward_timestamps, volumes


class LickInterface(ModuleInterface):
    """Interfaces with LickModule instances running on Ataraxis MicroControllers.

    LickModule allows interfacing with conductive lick sensors used in the Sun Lab to detect mouse interaction with
    water dispensing tubes. The sensor works by sending a small direct current through the mouse, which is picked up by
    the sensor connected to the metal lick tube. When the mouse completes the circuit by making the contact with the
    tube, the sensor determines whether the resultant voltage matches the threshold expected for a tongue contact and,
    if so, notifies the PC about the contact.

    Notes:
        The sensor is calibrated to work with very small currents that are not detectable by the animal, so it does not
        interfere with behavior during experiments. The sensor will, however, interfere with electrophysiological
        recordings.

        The resolution of the sensor is high enough to distinguish licks from paw touches. By default, the
        microcontroller is configured in a way that will likely send both licks and non-lick interactions to the PC.
        Use the lick_threshold argument to provide a more exclusive lick threshold.

        The interface automatically sends significant lick triggers to Unity via the "LickPort/" MQTT topic. This only
        includes the 'onset' triggers, the interface does not report voltage level reductions (associated with the end
        of the tongue-to-tube contact).

    Args:
        lick_threshold: The threshold voltage, in raw analog units recorded by a 12-bit ADC, for detecting the tongue
            contact. Note, 12-bit ADC only supports values between 0 and 4095, so setting the threshold above 4095 will
            result in no licks being reported to Unity.
        debug: A boolean flag that configures the interface to dump certain data received from the microcontroller into
            the terminal. This is used during debugging and system calibration and should be disabled for most runtimes.

    Attributes:
        _sensor_topic: Stores the output MQTT topic.
        _lick_threshold: The threshold voltage for detecting a tongue contact.
        _volt_per_adc_unit: The conversion factor to translate the raw analog values recorded by the 12-bit ADC into
            voltage in Volts.
        _communication: Stores the communication class used to send data to Unity over MQTT.
        _debug: Stores the debug flag.
        _lick_tracker: Stores the SharedMemoryArray that stores the current lick detection status and the ADC value
            associated with the status.
    """

    def __init__(self, lick_threshold: int = 1000, debug: bool = False) -> None:
        data_codes: set[np.uint8] = {np.uint8(51)}  # kChanged
        self._debug: bool = debug

        # Initializes the subclassed ModuleInterface using the input instance data. Type data is hardcoded.
        super().__init__(
            module_type=np.uint8(4),
            module_id=np.uint8(1),
            mqtt_communication=True,
            data_codes=data_codes,
            mqtt_command_topics=None,
            error_codes=None,
        )

        self._sensor_topic: str = "LickPort/"
        self._lick_threshold: np.uint16 = np.uint16(lick_threshold)

        # Statically computes the voltage resolution of each analog step, assuming a 3.3V ADC with 12-bit resolution.
        self._volt_per_adc_unit: np.float64 = np.round(a=np.float64(3.3 / (2**12)), decimals=8)

        # The communication class used to send data to Unity over MQTT. Initializes to a placeholder due to pickling
        # issues
        self._communication: MQTTCommunication | None = None

        # Precreates a shared memory array used to track and share the current lick sensor status.
        self._lick_tracker: SharedMemoryArray = SharedMemoryArray.create_array(
            name=f"{self._module_type}_{self._module_id}_lick_tracker",
            prototype=np.empty(shape=2, dtype=np.uint16),
            exist_ok=True,
        )

    def __del__(self) -> None:
        """Ensures the lick_tracker is properly cleaned up when the class is garbage-collected."""
        self._lick_tracker.disconnect()
        self._lick_tracker.destroy()

    def initialize_remote_assets(self) -> None:
        """Initializes the MQTTCommunication class, connects to the MQTT broker, and connects to the SharedMemoryArray
        used to communicate lick status to other processes.
        """
        # MQTT Client is used to send lick data to Unity over MQTT
        self._communication = MQTTCommunication()
        self._communication.connect()
        self._lick_tracker.connect()

    def terminate_remote_assets(self) -> None:
        """Destroys the MQTTCommunication class and disconnects from the lick-tracker SharedMemoryArray."""
        self._communication.disconnect()
        self._lick_tracker.disconnect()  # Does not destroy the array to support start / stop cycling.

    def process_received_data(self, message: ModuleData | ModuleState) -> None:
        """Processes incoming data.

        Lick data (code 51) comes in as a change in the voltage level detected by the sensor pin. This value is then
        evaluated against the _lick_threshold and if the value exceeds the threshold, a binary lick trigger is sent to
        Unity via MQTT. Additionally, the method sends both rising and falling lick detection triggers and their ADC
        values to the central process so that the data can be used for closed-loop lick-valve control.

        Notes:
            If the class runs in debug mode, this method sends all received lick sensor voltages to the
            terminal via console. Make sure the console is enabled before calling this method.
        """

        # Currently, only code 51 messages will be passed to this method. From each, extracts the detected voltage
        # level.
        detected_voltage: np.uint16 = message.data_object  # type: ignore

        # If the class is initialized in debug mode, prints each received voltage level to the terminal.
        if self._debug:
            console.echo(f"Lick ADC signal: {detected_voltage}")

        # If the voltage level exceeds the lick threshold, reports it to Unity via MQTT. Threshold is inclusive.
        if detected_voltage >= self._lick_threshold:
            # If the sensor detects a significantly high voltage, sends an empty message to the sensor MQTT topic,
            # which acts as a binary lick trigger.
            self._communication.send_data(topic=self._sensor_topic, payload=None)

            # Updates the tracker array with new data
            self._lick_tracker.write_data(index=(0, 2), data=np.array([1, detected_voltage], dtype=np.uint16))
        else:
            # Updates the tracker array with new data
            self._lick_tracker.write_data(index=(0, 2), data=np.array([0, detected_voltage], dtype=np.uint16))

    def parse_mqtt_command(self, topic: str, payload: bytes | bytearray) -> None:
        """Not used."""
        return

    def set_parameters(
        self,
        signal_threshold: np.uint16 = np.uint16(200),
        delta_threshold: np.uint16 = np.uint16(180),
        averaging_pool_size: np.uint8 = np.uint8(30),
    ) -> None:
        """Changes the PC-addressable runtime parameters of the LickModule instance.

        Use this method to package and apply new PC-addressable parameters to the LickModule instance managed by this
        Interface class.

        Notes:
            All threshold parameters are inclusive! If you need help determining appropriate threshold levels for
            specific targeted voltages, use the get_adc_units_from_volts() method of the interface instance.

        Args:
            signal_threshold: The minimum voltage level, in raw analog units of 12-bit Analog-to-Digital-Converter
                (ADC), that needs to be reported to the PC. Setting this threshold to a number above zero allows
                high-pass filtering the incoming signals. Note, Signals below the threshold will be pulled to 0.
            delta_threshold: The minimum value by which the signal has to change, relative to the previous check, for
                the change to be reported to the PC. Note, if the change is 0, the signal will not be reported to the
                PC, regardless of this parameter value.
            averaging_pool_size: The number of analog pin readouts to average together when checking pin state. This
                is used to smooth the recorded values to avoid communication line noise. Teensy microcontrollers have
                built-in analog pin averaging, but we disable it by default and use this averaging method instead. It is
                recommended to set this value between 15 and 30 readouts.
        """
        message = ModuleParameters(
            module_type=self._module_type,
            module_id=self._module_id,
            return_code=np.uint8(0),  # Generally, return code is only helpful for debugging.
            parameter_data=(signal_threshold, delta_threshold, averaging_pool_size),
        )
        self._input_queue.put(message)  # type: ignore

    def check_state(self, repetition_delay: np.uint32 = np.uint32(0)) -> None:
        """Returns the voltage signal detected by the analog pin monitored by the LickModule.

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
        """
        if repetition_delay == 0:
            command = OneOffModuleCommand(
                module_type=self._module_type,
                module_id=self._module_id,
                return_code=np.uint8(0),
                command=np.uint8(1),
                noblock=np.bool(False),
            )

        else:
            command = RepeatedModuleCommand(
                module_type=self._module_type,
                module_id=self._module_id,
                return_code=np.uint8(0),
                command=np.uint8(1),
                noblock=np.bool(False),
                cycle_delay=repetition_delay,
            )
        self._input_queue.put(command)  # type: ignore

    def get_adc_units_from_volts(self, voltage: float) -> np.uint16:
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
        return np.uint16(np.round(voltage / self._volt_per_adc_unit))

    @property
    def mqtt_topic(self) -> str:
        """Returns the MQTT topic used to transfer lick events from the interface to Unity."""
        return self._sensor_topic

    @property
    def volts_per_adc_unit(self) -> np.float64:
        """Returns the conversion factor to translate the raw analog values recorded by the 12-bit ADC into voltage in
        Volts.
        """
        return self._volt_per_adc_unit

    @property
    def lick_tracker(self) -> SharedMemoryArray:
        """Returns the SharedMemoryArray that stores the current lick status and the corresponding ADC value detected
        by the sensor.

        The lick status is stored under index 0, while the ADC readout is stored under index 1. Both values are stored
        as an uint16 datatype.

        The lick status is 1 when the sensor detects a tongue contact and 0 otherwise. The ADC value is between 0,
        representing no voltage flowing through the sensor (no tongue to complete the sensor circuit) and 4095,
        representing maximum system voltage of 3.3 V flowing through the sensor.
        """
        return self._lick_tracker

    @property
    def lick_status(self) -> bool:
        """Returns True if the sensor is currently detecting a lick and False otherwise."""
        return bool(self._lick_tracker.read_data(index=0))

    def parse_logged_data(self) -> tuple[NDArray[np.uint64], NDArray[np.uint8]]:
        """Extracts and prepares the data acquired by the module during runtime for further analysis.

        Notes:
            The extraction automatically filters out non-lick events by applying the class lick-threshold value. The
            time-difference between consecutive ON and OFF event edges corresponds to the time, in microseconds, the
            tongue maintained contact with the lick tube. This may include both the time the tongue physically
            touched the tube and the time there was a conductive fluid bridge between the tongue and the lick tube.

            In addition to filtering out non-lick events, the code also converts multiple consecutive above-threshold or
            below-threshold readouts into LOW and HIGH epochs. Each HIGH epoch denotes the duration, for that particular
            lick, that the tongue maintained contact with the sensor. Each LOW epoch denotes the duration between licks
            that the tongue was not making contact with the sensor.

        Returns:
            A tuple with two elements. The first element is a numpy array that stores the timestamps, as microseconds
            elapsed since UTC epoch onset. The second element is a numpy array that stores binary levels denoting the
            lick detection as 1 (Detected, tongue makes contact with the lick tube) and 0 (Not detected, tongue is not
            making contact with the tube).
        """
        # Reads the data logged during runtime as a dictionary of dictionaries.
        log_data: dict[Any, list[dict[str, Any]]] = self.extract_logged_data()

        # LickModule only sends messages with code 51 (Voltage level changed). Therefore, this extraction pipeline has
        # to apply the threshold filter, similar to how the real-time processing method.

        # Unlike the other parsing methods, this one will always work as expected since it only deals with one code and
        # that code is guaranteed to be received for each runtime.

        # Precreates the storage numpy arrays for both message types. Timestamps use uint64 datatype. Lick sensor
        # voltage levels come in as uint16, but we later replace them with binary uint8 1 and 0 values.
        voltage_data = log_data[np.uint8(51)]
        total_length = len(voltage_data)
        timestamps: NDArray[np.uint64] = np.empty(total_length, dtype=np.uint64)
        voltages: NDArray[np.uint16] = np.empty(total_length, dtype=np.uint16)

        # Extract timestamps and voltage levels
        timestamps[:] = [value["timestamp"] for value in voltage_data]
        voltages[:] = [value["data"] for value in voltage_data]

        # Converts voltage levels to binary lick states based on the class threshold. Note, the threshold is inclusive.
        licks = np.where(voltages >= self._lick_threshold, np.uint8(1), np.uint8(0))

        # Sorts all arrays by timestamp. This is technically not needed as the extracted values are already sorted by
        # timestamp, but this is still done for additional safety.
        sort_indices = np.argsort(timestamps)
        timestamps = timestamps[sort_indices]
        licks = licks[sort_indices]

        # Finds indices where lick state changes (either 0->1 or 1->0)
        state_changes = np.where(licks[:-1] != licks[1:])[0] + 1

        # Extracts the state values and corresponding timestamps for each change
        state_stamps = timestamps[state_changes]
        states = licks[state_changes]

        # The transformation above removes the initial lick state (0). Re-adds the initial timestamp and state to
        # the output array
        timestamps = np.concatenate(([timestamps[0]], state_stamps))
        states = np.concatenate(([licks[0]], states))

        # Returns timestamps and states at transition points
        return timestamps, states


class TorqueInterface(ModuleInterface):
    """Interfaces with TorqueModule instances running on Ataraxis MicroControllers.

    TorqueModule interfaces with a differential torque sensor. The sensor uses differential coding in the millivolt
    range to communicate torque in the CW and the CCW direction. To convert and amplify the output of the torque sensor,
    it is wired to an AD620 microvolt amplifier instrument that converts the output signal into a single positive
    vector and amplifies its strength to Volts range.

    The TorqueModule further refines the sensor data by ensuring that CCW and CW torque signals behave identically.
    Specifically, it adjusts the signal to scale from 0 to baseline proportionally to the detected torque, regardless
    of torque direction.

    Notes:
        This interface receives torque as a positive uint16_t value from zero to at most 2046 raw analog units of 3.3v
        12-bit ADC converter. The direction of the torque is reported by the event-code of the received message.

    Args:
        baseline_voltage: The voltage level, in raw analog units measured by 3.3v ADC at 12-bit resolution after the
            AD620 amplifier, that corresponds to no (0) torque readout. Usually, for a 3.3v ADC, this would be around
            2046 (the midpoint, ~1.65 V).
        maximum_voltage: The voltage level, in raw analog units measured by 3.3v ADC at 12-bit resolution after the
            AD620 amplifier, that corresponds to the absolute maximum torque detectable by the sensor. The best way
            to get this value is to measure the positive voltage level after applying the maximum CW (positive) torque.
            At most, this value can be 4095 (~3.3 V).
        sensor_capacity: The maximum torque detectable by the sensor, in grams centimeter (g cm).
        object_diameter: The diameter of the rotating object connected to the torque sensor, in centimeters. This is
            used to calculate the force at the edge of the object associated with the measured torque at the sensor.
        debug: A boolean flag that configures the interface to dump certain data received from the microcontroller into
            the terminal. This is used during debugging and system calibration and should be disabled for most runtimes.

    Attributes:
        _newton_per_gram_centimeter: Stores the hardcoded conversion factor from gram centimeter to Newton centimeter.
        _capacity_in_newtons_cm: The maximum torque detectable by the sensor in Newtons centimeter.
        _torque_per_adc_unit: The conversion factor to translate raw analog 3.3v 12-bit ADC values to torque in Newtons
            centimeter.
        _force_per_adc_unit: The conversion factor to translate raw analog 3.3v 12-bit ADC values to force in Newtons.
        _debug: Stores the debug flag.
    """

    def __init__(
        self,
        baseline_voltage: int = 2046,
        maximum_voltage: int = 2750,
        sensor_capacity: float = 720.0779,  # 10 oz in
        object_diameter: float = 15.0333,
        debug: bool = False,
    ) -> None:
        self._debug: bool = debug
        # data_codes = {np.uint8(51), np.uint8(52)}  # kCCWTorque, kCWTorque

        # If the interface runs in the debug mode, configures it to monitor and report detected torque values
        data_codes: set[np.uint8] | None = None
        if debug:
            data_codes = {np.uint8(51), np.uint8(52)}

        # Initializes the subclassed ModuleInterface using the input instance data. Type data is hardcoded.
        super().__init__(
            module_type=np.uint8(6),
            module_id=np.uint8(1),
            mqtt_communication=False,
            data_codes=data_codes,
            mqtt_command_topics=None,
            error_codes=None,
        )

        # Hardcodes the conversion factor used to translate torque in g cm to N cm
        self._newton_per_gram_centimeter: np.float64 = np.float64(0.00981)

        # Determines the capacity of the torque sensor in Newtons centimeter.
        self._capacity_in_newtons_cm: np.float64 = np.round(
            a=np.float64(sensor_capacity) * self._newton_per_gram_centimeter,
            decimals=8,
        )

        # Computes the conversion factor to translate the recorded raw analog readouts of the 3.3V 12-bit ADC to
        # torque in Newton centimeter. Rounds to 12 decimal places for consistency and to ensure
        # repeatability.
        self._torque_per_adc_unit: np.float64 = np.round(
            a=(self._capacity_in_newtons_cm / (maximum_voltage - baseline_voltage)),
            decimals=8,
        )

        # Also computes the conversion factor to translate the recorded raw analog readouts of the 3.3V 12-bit ADC to
        # force in Newtons.
        self._force_per_adc_unit: np.float64 = np.round(
            a=self._torque_per_adc_unit / (object_diameter / 2),
            decimals=8,
        )

    def initialize_remote_assets(self) -> None:
        """Not used."""
        pass

    def terminate_remote_assets(self) -> None:
        """Not used."""
        return

    def process_received_data(self, message: ModuleData | ModuleState) -> None:
        """If the class is initialized in debug mode, prints the received torque data to the terminal via console.

        In debug mode, this method parses incoming code 51 (CW torque) and code 52 (CCW torque) data and dumps it into
         the terminal via console. If the class is not initialized in debug mode, this method does nothing.

        Notes:
            Make sure the console is enabled before calling this method.
        """
        # The torque direction is encoded via the message event code. CW torque (code 52) is interpreted as negative
        # and CCW (code 51) as positive.
        sign = 1 if message.event == np.uint8(51) else -1

        # Translates the absolute torque into the CW / CCW vector and converts from raw ADC units to Newton centimeters
        # using the precomputed conversion factor. Uses float64 and rounds to 8 decimal places for consistency and
        # precision
        signed_torque = np.round(
            a=np.float64(message.data_object) * self._torque_per_adc_unit * sign,
            decimals=8,
        )

        # Since this method is only called in the debug mode, always prints the data to the console
        console.echo(message=f"Torque: {signed_torque} N cm, ADC: {np.int32(message.data_object) * sign}.")

    def parse_mqtt_command(self, topic: str, payload: bytes | bytearray) -> None:
        """Not used."""
        return

    def set_parameters(
        self,
        report_ccw: np.bool = np.bool(True),
        report_cw: np.bool = np.bool(True),
        signal_threshold: np.uint16 = np.uint16(100),
        delta_threshold: np.uint16 = np.uint16(70),
        averaging_pool_size: np.uint8 = np.uint8(10),
    ) -> None:
        """Changes the PC-addressable runtime parameters of the TorqueModule instance.

        Use this method to package and apply new PC-addressable parameters to the TorqueModule instance managed by this
        Interface class.

        Notes:
            All threshold parameters are inclusive! If you need help determining appropriate threshold levels for
            specific targeted torque levels, use the get_adc_units_from_torque() method of the interface instance.

        Args:
            report_ccw: Determines whether the sensor should report torque in the CounterClockwise (CCW) direction.
            report_cw: Determines whether the sensor should report torque in the Clockwise (CW) direction.
            signal_threshold: The minimum torque level, in raw analog units of 12-bit Analog-to-Digital-Converter
                (ADC), that needs to be reported to the PC. Setting this threshold to a number above zero allows
                high-pass filtering the incoming signals. Note, Signals below the threshold will be pulled to 0.
            delta_threshold: The minimum value by which the signal has to change, relative to the previous check, for
                the change to be reported to the PC. Note, if the change is 0, the signal will not be reported to the
                PC, regardless of this parameter value.
            averaging_pool_size: The number of analog pin readouts to average together when checking pin state. This
                is used to smooth the recorded values to avoid communication line noise. Teensy microcontrollers have
                built-in analog pin averaging, but we disable it by default and use this averaging method instead. It is
                recommended to set this value between 15 and 30 readouts.
        """
        message = ModuleParameters(
            module_type=self._module_type,
            module_id=self._module_id,
            return_code=np.uint8(0),  # Generally, return code is only helpful for debugging.
            parameter_data=(
                report_ccw,
                report_cw,
                signal_threshold,
                delta_threshold,
                averaging_pool_size,
            ),
        )
        self._input_queue.put(message)  # type: ignore

    def check_state(self, repetition_delay: np.uint32 = np.uint32(0)) -> None:
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
        """
        if repetition_delay == 0:
            command = OneOffModuleCommand(
                module_type=self._module_type,
                module_id=self._module_id,
                return_code=np.uint8(0),
                command=np.uint8(1),
                noblock=np.bool(False),
            )
        else:
            command = RepeatedModuleCommand(
                module_type=self._module_type,
                module_id=self._module_id,
                return_code=np.uint8(0),
                command=np.uint8(1),
                noblock=np.bool(False),
                cycle_delay=repetition_delay,
            )
        self._input_queue.put(command)  # type: ignore

    def get_adc_units_from_torque(self, target_torque: float) -> np.uint16:
        """Converts the input torque to raw analog units of 12-bit Analog-to-Digital-Converter (ADC).

        Use this method to determine the appropriate raw analog units for the threshold arguments of the
        set_parameters() method.

        Notes:
            This method assumes a 3.3V ADC with 12-bit resolution.

        Args:
            target_torque: The target torque in Newton centimeter, to convert to an ADC threshold.

        Returns:
            The raw analog units of 12-bit ADC for the input torque.
        """
        return np.uint16(np.round(target_torque / self._torque_per_adc_unit))

    @property
    def torque_per_adc_unit(self) -> np.float64:
        """Returns the conversion factor to translate the raw analog values recorded by the 12-bit ADC into torque in
        Newton centimeter.
        """
        return self._torque_per_adc_unit

    @property
    def force_per_adc_unit(self) -> np.float64:
        """Returns the conversion factor to translate the raw analog values recorded by the 12-bit ADC into force in
        Newtons.
        """
        return self._force_per_adc_unit

    def parse_logged_data(self) -> tuple[NDArray[np.uint64], NDArray[np.float64]]:
        """Extracts and prepares the data acquired by the module during runtime for further analysis.

        Notes:
            Despite this method trying to translate the detected torque into Newton centimeters, it may not be accurate.
            Partially, the accuracy of the translation depends on the calibration of the interface class, which is very
            hard with our current setup. The accuracy also depends on the used hardware, and currently our hardware is
            not very well suited for working with millivolt differential voltage levels used by the sensor to report
            torque. Therefore, currently, it is best to treat the torque data extracted from this module as a very rough
            estimate of how active the animal is at a given point in time.

        Returns:
            A tuple with two elements. The first element is a numpy array that stores the timestamps, as microseconds
            elapsed since UTC epoch onset. The second element is a numpy array that stores the torque applied by the
            animal to the running wheel at each timestamp in Newton centimeters.
        """
        # Reads the data logged during runtime as a dictionary of dictionaries.
        log_data: dict[Any, list[dict[str, Any]]] = self.extract_logged_data()

        # Here, we only look for event-codes 51 (CCW Torque) and event-codes 52 (CW Torque). CCW torque is interpreted
        # as torque in the positive direction, and CW torque is interpreted as torque in the negative direction.

        # Gets the data, defaulting to an empty list if the data is missing
        ccw_data = log_data.get(np.uint8(51), [])
        cw_data = log_data.get(np.uint8(52), [])

        # The way TorqueModule is implemented guarantees there is at least one CW code message with the displacement
        # of 0 that is received by the PC. In the worst case scenario, there will be no CCW codes and the parsing will
        # not work. To avoid that issue, we generate an artificial zero-code CCW value at the same timestamp + 1
        # microsecond as the original CW zero-code value. This does not affect the accuracy of our data, just makes the
        # code work for edge-cases.
        if not ccw_data:
            first_timestamp = cw_data[0]["timestamp"]
            ccw_data = [{"timestamp": first_timestamp + 1, "data": 0}]
        elif not cw_data:
            first_timestamp = ccw_data[0]["timestamp"]
            cw_data = [{"timestamp": first_timestamp + 1, "data": 0}]

        # Precreates the storage numpy arrays for both message types. Timestamps use uint64 datatype. Although torque
        # values are uint16, we translate them into the actual torque applied by the animal in Newton centimeters and
        # store them as float 64 values.
        total_length = len(ccw_data) + len(cw_data)
        timestamps: NDArray[np.uint64] = np.empty(total_length, dtype=np.uint64)
        torques: NDArray[np.float64] = np.empty(total_length, dtype=np.float64)

        # Processes CCW torques (Code 51). CCW torque is interpreted as positive torque
        n_ccw = len(ccw_data)
        timestamps[:n_ccw] = [value["timestamp"] for value in ccw_data]  # Extracts timestamps for each value
        # The values are initially using the uint16 type. This converts them to float64 and translates from raw ADC
        # units into Newton centimeters.
        torques[:n_ccw] = [
            np.round(np.float64(value["data"]) * self._torque_per_adc_unit, decimals=8) for value in ccw_data
        ]

        # Processes CW torques (Code 52). CW torque is interpreted as negative torque
        timestamps[n_ccw:] = [value["timestamp"] for value in cw_data]  # CW data just fills remaining space after CCW.
        torques[n_ccw:] = [
            np.round(-np.float64(value["data"]) * self._torque_per_adc_unit, decimals=8) for value in cw_data
        ]

        # Sorts both arrays based on timestamps.
        sort_indices = np.argsort(timestamps)
        timestamps = timestamps[sort_indices]
        torques = torques[sort_indices]

        return timestamps, torques


class ScreenInterface(ModuleInterface):
    """Interfaces with ScreenModule instances running on Ataraxis MicroControllers.

    ScreenModule is specifically designed to interface with the HDMI converter boards used in Sun lab's Virtual Reality
    setup. The ScreenModule communicates with the boards to toggle the screen displays on and off, without interfering
    with their setup on the host PC.

    Notes:
        Since the current VR setup uses three screens, this implementation of ScreenModule is designed to interface
        with all three screens at the same time. In the future, the module may be refactored to allow addressing
        individual screens.

        The physical wiring of the module also allows manual screen manipulation via the buttons on the control panel
        if the ScreenModule is not actively delivering a toggle pulse. However, changing the state of the screen
        manually is strongly discouraged, as it interferes with tracking the state of the screen via software.

    Args:
        initially_on: A boolean flag that communicates the initial state of the screen. This is used during log parsing
            to deduce the state of the screen after each toggle pulse and assumes the screens are only manipulated via
            this interface.
        debug: A boolean flag that configures the interface to dump certain data received from the microcontroller into
            the terminal. This is used during debugging and system calibration and should be disabled for most runtimes.

    Attributes:
        _initially_on: Stores the initial state of the screens.
        _debug: Stores the debug flag.
    """

    def __init__(self, initially_on: bool, debug: bool = False) -> None:
        error_codes: set[np.uint8] = {np.uint8(51)}  # kOutputLocked

        self._debug: bool = debug
        self._initially_on: bool = initially_on

        # kOn, kOff
        # data_codes = {np.uint8(52), np.uint8(53)}

        # If the interface runs in the debug mode, configures the interface to monitor relay On / Off codes.
        data_codes: set[np.uint8] | None = None
        if debug:
            data_codes = {np.uint8(52), np.uint8(53)}

        super().__init__(
            module_type=np.uint8(7),
            module_id=np.uint8(1),
            mqtt_communication=False,
            data_codes=data_codes,
            mqtt_command_topics=None,
            error_codes=error_codes,
        )

    def initialize_remote_assets(self) -> None:
        """Not used."""
        pass

    def terminate_remote_assets(self) -> None:
        """Not used."""
        pass

    def process_received_data(self, message: ModuleData | ModuleState) -> None:
        """If the class runs in the debug mode, dumps the received data into the terminal via console class.

        This method is only used in the debug mode to print Screen toggle signal HIGH (On) and LOW (Off) phases.

        Notes:
            This method uses the console to print the data to the terminal. Make sure it is enabled before calling this
            method.
        """
        if message.event == 52:
            console.echo(f"Screen toggle: HIGH")
        if message.event == 53:
            console.echo(f"Screen toggle: LOW")

    def parse_mqtt_command(self, topic: str, payload: bytes | bytearray) -> None:
        """Not used."""
        return

    def set_parameters(self, pulse_duration: np.uint32 = np.uint32(1000000)) -> None:
        """Changes the PC-addressable runtime parameters of the ScreenModule instance.

        Use this method to package and apply new PC-addressable parameters to the ScreenModule instance managed by
        this Interface class.

        Args:
            pulse_duration: The duration, in microseconds, of each emitted screen toggle pulse HIGH phase. This is
                equivalent to the duration of the control panel POWER button press. The main criterion for this
                parameter is to be long enough for the converter board to register the press.
        """
        message = ModuleParameters(
            module_type=self._module_type,
            module_id=self._module_id,
            return_code=np.uint8(0),
            parameter_data=(pulse_duration,),
        )
        self._input_queue.put(message)  # type: ignore

    def toggle(self) -> None:
        """Triggers the ScreenModule to briefly simulate pressing the POWER button of the scree control board.

        This command is used to turn the connected display on or off. The new state of the display depends on the
        current state of the display when the command is issued. Since the displays can also be controlled manually
        (via the physical control board buttons), the state of the display can also be changed outside this interface,
        although it is highly advised to NOT change screen states manually.

        Notes:
            It is highly recommended to use this command to manipulate display states, as it ensures that display state
            changes are logged for further data analysis.
        """
        command = OneOffModuleCommand(
            module_type=self._module_type,
            module_id=self._module_id,
            return_code=np.uint8(0),
            command=np.uint8(1),
            noblock=np.bool(False),
        )
        self._input_queue.put(command)  # type: ignore

    def parse_logged_data(self) -> tuple[NDArray[np.uint64], NDArray[np.uint8]]:
        """Extracts and prepares the data acquired by the module during runtime for further analysis.

        Notes:
            This extraction method works similar to the TTLModule method. This is intentional, as ScreenInterface is
            essentially a group of 3 TTLModules.

        Returns:
            A tuple with two elements. The first element is a numpy array that stores the timestamps, as microseconds
            elapsed since UTC epoch onset. The second element is a numpy array that stores the state of the screens
            (1 for ON, 0 for OFF) at each timestamp.
        """
        # Reads the data logged during runtime as a dictionary of dictionaries.
        log_data: dict[Any, list[dict[str, Any]]] = self.extract_logged_data()

        # Here, we only look for event-codes 52 (pulse ON) and event-codes 53 (pulse OFF).

        # The way the module is implemented guarantees there is at least one code 53 message. However, if screen state
        # is never toggled, there may be no code 52 messages.
        on_data = log_data.get(np.uint8(52), [])
        off_data = log_data[np.uint8(53)]

        # If there were no ON pulses, screens never changed state. In this case, shorts to returning the data for the
        # initial screen state using the initial Off timestamp. Otherwise, parses the data
        if not on_data:
            # Return first timestamp with initial state
            return np.array([off_data[0]["timestamp"]], dtype=np.uint64), np.array([self._initially_on], dtype=np.uint8)

        # Precreates the storage numpy arrays for both message types. Timestamps use uint64 datatype and the trigger
        # values are boolean. We use uint8 as it has the same memory footprint as a boolean and allows us to use integer
        # types across the entire dataset.
        total_length = len(on_data) + len(off_data)
        timestamps: NDArray[np.uint64] = np.empty(total_length, dtype=np.uint64)
        triggers: NDArray[np.uint8] = np.empty(total_length, dtype=np.uint8)

        # Extracts ON (Code 52) trigger codes. Statically assigns the value '1' to denote ON signals.
        n_on = len(on_data)
        timestamps[:n_on] = [value["timestamp"] for value in on_data]
        triggers[:n_on] = np.uint8(1)  # All code 52 signals are ON (High)

        # Extracts OFF (Code 53) trigger codes.
        timestamps[n_on:] = [value["timestamp"] for value in off_data]
        triggers[n_on:] = np.uint8(0)  # All code 53 signals are OFF (Low)

        # Sorts both arrays based on the timestamps, so that the data is in the chronological order.
        sort_indices = np.argsort(timestamps)
        timestamps = timestamps[sort_indices]
        triggers = triggers[sort_indices]

        # Finds rising edges (where the signal goes from 0 to 1). Then uses the indices for such events to extract the
        # timestamps associated with each rising edge, before returning them to the caller.
        rising_edges = np.where((triggers[:-1] == 0) & (triggers[1:] == 1))[0] + 1
        screen_timestamps = timestamps[rising_edges]

        # Adds the initial state of the screen using the first recorded timestamp. The module is configured to send the
        # initial state of the relay (Off) during Setup, so the first recorded timestamp will always be 0 and correspond
        # to the initial state of the screen.
        screen_timestamps = np.concatenate(([timestamps[0]], screen_timestamps))

        # Builds an array of screen states. Starts with the initial screen state and then flips the state for each
        # consecutive timestamp matching a rising edge of the toggle pulse.
        screen_states = np.zeros(len(screen_timestamps), dtype=np.uint8)
        screen_states[0] = self._initially_on
        for i in range(1, len(screen_states)):
            screen_states[i] = 1 - screen_states[i - 1]  # Flips between 0 and 1

        return screen_timestamps, screen_states
