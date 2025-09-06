"""This module provides the interfaces (ModuleInterface class implementations) for the hardware designed in the Sun lab.
These interfaces are designed to work with the hardware modules assembled and configured according to the instructions
from our microcontrollers' library: https://github.com/Sun-Lab-NBB/sl-micro-controllers."""

from typing import Any

import numpy as np
from numpy.typing import NDArray
from ataraxis_time import PrecisionTimer
from scipy.optimize import curve_fit
from ataraxis_base_utilities import console
from ataraxis_data_structures import SharedMemoryArray
from ataraxis_communication_interface import (
    ModuleData,
    ModuleState,
    ModuleInterface,
    ModuleParameters,
    OneOffModuleCommand,
    RepeatedModuleCommand,
)


class ValveInterface(ModuleInterface):
    """Interfaces with ValveModule instances running on Ataraxis MicroControllers.

    ValveModule allows interfacing with a solenoid valve to controllably dispense precise volumes of fluid. The module
    is designed to send digital signals that trigger Field-Effect-Transistor (FET) gated relay hardware to deliver
    voltage that opens or closes the controlled valve. The module can be used to either permanently open or close the
    valve or to cycle opening and closing in a way that ensures a specific amount of fluid passes through the
    valve.

    Args:
        valve_calibration_data: A tuple of tuples that contains the data required to map pulse duration to delivered
            fluid volume. Each sub-tuple should contain the integer that specifies the pulse duration in microseconds
            and a float that specifies the delivered fluid volume in microliters. If this data is not known at class
            initialization, use a placeholder calibration tuple and use the calibration() class method to collect this
            data using the ValveModule.
        debug: A boolean flag that configures the interface to dump certain data received from the microcontroller into
            the terminal. This is used during debugging and system calibration and should be disabled for most runtimes.

    Notes:
        This class is explicitly designed to work with the valves whose calibration curve is most closely approximated
        using a power law equation. Modify the source code of the class to use a different calibration procedure when
        using a valve whose calibration curve deviates from this assumption.

    Attributes:
        _scale_coefficient: Stores the power law scale coefficient derived from the calibration data.
        _nonlinearity_exponent: The intercept of the valve calibration curve. This is used to account for the fact that
            some valves may have a minimum open time or dispensed fluid volume, which is captured by the intercept.
            This improves the precision of fluid-volume-to-valve-open-time conversions.
        _debug: Stores the debug flag.
        _valve_tracker: Stores the SharedMemoryArray that tracks the total volume of water dispensed by the valve
            during runtime.
        _previous_state: Tracks the previous valve state as Open (True) or Closed (False). This is used to accurately
            track delivered water volumes each time the valve opens and closes.
        _cycle_timer: A PrecisionTimer instance initialized in the Communication process to track how long the valve
            stays open during cycling. This is used together with the _previous_state to determine the volume of water
            delivered by the valve during runtime.
    """

    def __init__(
        self,
        module_id: np.uint8,
        valve_calibration_data: tuple[tuple[int | float, int | float], ...],
        debug: bool = False,
    ) -> None:
        error_codes: set[np.uint8] = {np.uint8(51)}  # kOutputLocked
        # kOpen, kClosed, kCalibrated, kToneOn, kToneOff, kTonePinNotSet
        # data_codes = {np.uint8(52), np.uint8(53), np.uint8(54), np.uint8(55), np.uint8(56), np.uint8(57)}
        data_codes = {np.uint8(52), np.uint8(53), np.uint8(54)}  # kOpen, kClosed, kCalibrated

        self._debug: bool = debug

        # If the interface runs in the debug mode, expands the list of processed data codes to include all codes used
        # by the valve module.
        if debug:
            data_codes = {np.uint8(52), np.uint8(53), np.uint8(54)}

        super().__init__(
            module_type=np.uint8(5),
            module_id=module_id,
            mqtt_communication=False,
            data_codes=data_codes,
            mqtt_command_topics=None,
            error_codes=error_codes,
        )

        # Extracts pulse durations and fluid volumes into separate arrays
        pulse_durations: NDArray[np.float64] = np.array([x[0] for x in valve_calibration_data], dtype=np.float64)
        fluid_volumes: NDArray[np.float64] = np.array([x[1] for x in valve_calibration_data], dtype=np.float64)

        # Defines the power-law model. Our calibration data suggests that the Valve performs in a non-linear fashion
        # and is better calibrated using the power law, rather than a linear fit
        def power_law_model(pulse_duration: Any, a: Any, b: Any, /) -> Any:
            return a * np.power(pulse_duration, b)

        # Fits the power-law model to the input calibration data and saves the fit parameters and covariance matrix to
        # class attributes
        # noinspection PyTupleAssignmentBalance
        params, _ = curve_fit(f=power_law_model, xdata=pulse_durations, ydata=fluid_volumes)
        scale_coefficient, nonlinearity_exponent = params
        self._scale_coefficient: np.float64 = np.round(a=np.float64(scale_coefficient), decimals=8)
        self._nonlinearity_exponent: np.float64 = np.round(a=np.float64(nonlinearity_exponent), decimals=8)

        # Precreates a shared memory array used to track and share valve state data. Index 0 tracks the total amount of
        # water dispensed by the valve during runtime.
        self._valve_tracker: SharedMemoryArray = SharedMemoryArray.create_array(
            name=f"{self._module_type}_{self._module_id}_valve_tracker",
            prototype=np.zeros(shape=1, dtype=np.float64),
            exist_ok=True,
        )
        self._previous_state: bool = False
        self._cycle_timer: PrecisionTimer | None = None

    def __del__(self) -> None:
        """Ensures the reward_tracker is properly cleaned up when the class is garbage-collected."""
        self._valve_tracker.disconnect()
        self._valve_tracker.destroy()

    def initialize_remote_assets(self) -> None:
        """Connects to the reward tracker SharedMemoryArray and initializes the cycle PrecisionTimer from the
        Communication process."""
        self._valve_tracker.connect()
        self._cycle_timer = PrecisionTimer("us")

    def terminate_remote_assets(self) -> None:
        """Disconnects from the reward tracker SharedMemoryArray."""
        self._valve_tracker.disconnect()

    def process_received_data(self, message: ModuleData | ModuleState) -> None:
        """Processes incoming data sent by the module to the PC."""
        if message.event == 52:
            if self._debug:
                console.echo(f"Valve Opened")

            # Resets the cycle timer each time the valve transitions to open state.
            if not self._previous_state:
                self._previous_state = True
                self._cycle_timer.reset()  # type: ignore

        elif message.event == 53:
            if self._debug:
                console.echo(f"Valve Closed")

            # Each time the valve transitions to closed state, records the period of time the valve was open and uses it
            # to estimate the volume of fluid delivered through the valve.
            if self._previous_state:
                self._previous_state = False
                open_duration = self._cycle_timer.elapsed  # type: ignore

                # Accumulates delivered water volumes into the tracker.
                delivered_volume = np.float64(
                    self._scale_coefficient * np.power(open_duration, self._nonlinearity_exponent)
                )
                previous_volume = np.float64(self._valve_tracker.read_data(index=0, convert_output=False))
                new_volume = previous_volume + delivered_volume
                # noinspection PyTypeChecker
                self._valve_tracker.write_data(index=0, data=new_volume)
        elif message.event == 54:
            console.echo(f"Valve Calibration: Complete")

    def parse_mqtt_command(self, topic: str, payload: bytes | bytearray) -> None:
        """Not used."""
        return

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
            calibration_count: The number of times to pulse the valve during calibration. A number between 100 and 200
                is enough for most use cases.
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
            noblock: Determines whether the command should block the microcontroller while the valve is kept open.
                Blocking ensures precise pulse duration and dispensed fluid volume. Non-blocking allows the
                microcontroller to perform other operations while waiting, increasing its throughput.
        """
        command: OneOffModuleCommand | RepeatedModuleCommand
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
        """Converts the desired fluid volume in microliters to the valve pulse duration in microseconds.

        Use this method to convert the desired fluid volume into the pulse_duration value that can be submitted to the
        ValveModule via the set_parameters() class method.

        Args:
            target_volume: Desired fluid volume in microliters.

        Raises:
            ValueError: If the desired fluid volume is too small to be reliably dispensed by the valve, based on its
                calibration data.

        Returns:
            The microsecond pulse duration that would allow the valve deliver the specified volume of fluid.
        """
        # Determines the minimum valid pulse duration. This is hardcoded at 10 ms as this is the lower calibration
        # boundary.
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
    def scale_coefficient(self) -> np.float64:
        """Returns the scaling coefficient (A) derived during the power‐law calibration.

        In the calibration model, fluid_volume = A * (pulse_duration)^B, this coefficient
        converts pulse duration (in microseconds) into the appropriate fluid volume (in microliters)
        when used together with the nonlinearity exponent.
        """
        return self._scale_coefficient

    @property
    def nonlinearity_exponent(self) -> np.float64:
        """Returns the nonlinearity exponent (B) derived during the power‐law calibration.

        In the calibration model, fluid_volume = A * (pulse_duration)^B, this exponent indicates
        the degree of nonlinearity in how the dispensed volume scales with the valve’s pulse duration.
        For example, an exponent of 1 would indicate a linear relationship.
        """
        return self._nonlinearity_exponent

    @property
    def delivered_volume(self) -> np.float64:
        """Returns the total volume of water, in microliters, delivered by the valve during this runtime."""
        return self._valve_tracker.read_data(index=0, convert_output=False)  # type: ignore


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
        _debug: Stores the debug flag.
        _lick_tracker: Stores the SharedMemoryArray that stores the current lick detection status and the total number
            of licks detected since class initialization.
        _previous_readout_zero: Stores a boolean indicator of whether the previous voltage readout was a 0-value.
    """

    def __init__(self, module_id: np.uint8, lick_threshold: int = 1000, debug: bool = False) -> None:
        data_codes: set[np.uint8] = {np.uint8(51)}  # kChanged
        self._debug: bool = debug

        # Initializes the subclassed ModuleInterface using the input instance data. Type data is hardcoded.
        super().__init__(
            module_type=np.uint8(4),
            module_id=module_id,
            mqtt_communication=False,
            data_codes=data_codes,
            mqtt_command_topics=None,
            error_codes=None,
        )

        self._sensor_topic: str = "LickPort/"
        self._lick_threshold: np.uint16 = np.uint16(lick_threshold)

        # Statically computes the voltage resolution of each analog step, assuming a 3.3V ADC with 12-bit resolution.
        self._volt_per_adc_unit: np.float64 = np.round(a=np.float64(3.3 / (2**12)), decimals=8)

        # Precreates a shared memory array used to track and share the total number of licks recorded by the sensor
        # since class initialization.
        self._lick_tracker: SharedMemoryArray = SharedMemoryArray.create_array(
            name=f"{self._module_type}_{self._module_id}_lick_tracker",
            prototype=np.zeros(shape=1, dtype=np.uint64),
            exist_ok=True,
        )

        # Precreates storage variables used to prevent excessive lick reporting
        self._previous_readout_zero: bool = False

    def __del__(self) -> None:
        """Ensures the lick_tracker is properly cleaned up when the class is garbage-collected."""
        self._lick_tracker.disconnect()
        self._lick_tracker.destroy()

    def initialize_remote_assets(self) -> None:
        """Connects to the SharedMemoryArray used to communicate lick status to other processes."""
        self._lick_tracker.connect()

    def terminate_remote_assets(self) -> None:
        """Disconnects from the lick-tracker SharedMemoryArray."""
        self._lick_tracker.disconnect()  # Does not destroy the array to support start / stop cycling.

    def process_received_data(self, message: ModuleData | ModuleState) -> None:
        """Processes incoming data.

        Lick data (code 51) comes in as a change in the voltage level detected by the sensor pin. This value is then
        evaluated against the _lick_threshold, and if the value exceeds the threshold, a binary lick trigger is sent to
        Unity via MQTT. Additionally, the method increments the total lick count stored in the _lick_tracker each time
        an above-threshold voltage readout is received from the module.

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

        # Since the sensor is pulled to 0 to indicate lack of tongue contact, a zero-readout necessarily means no
        # lick. Sets zero-tracker to 1 to indicate that a zero-state has been encountered
        if detected_voltage == 0:
            self._previous_readout_zero = True
            return

        # If the voltage level exceeds the lick threshold and this is the first time the threshold is exceeded since
        # the last zero-value, reports it to Unity via MQTT. Threshold is inclusive. This exploits the fact that every
        # pair of licks has to be separated by a zero-value (lack of tongue contact). So, to properly report the licks,
        # only does it once per encountering a zero-value.
        if detected_voltage >= self._lick_threshold and self._previous_readout_zero:
            # Increments the lick count and updates the tracker array with new data
            count = self._lick_tracker.read_data(index=0, convert_output=False)
            count += 1
            self._lick_tracker.write_data(index=0, data=count)

            # This disables further reports until the sensor sends a zero-value again
            self._previous_readout_zero = False

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
        command: OneOffModuleCommand | RepeatedModuleCommand
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
    def lick_count(self) -> np.uint64:
        """Returns the total number of licks detected by the module since runtime onset."""
        return self._lick_tracker.read_data(index=0, convert_output=False)  # type: ignore

    @property
    def lick_threshold(self) -> np.uint16:
        """Returns the voltage threshold, in raw ADC units of a 12-bit Analog-to-Digital voltage converter that is
        interpreted as the mouse licking the sensor."""
        return self._lick_threshold
