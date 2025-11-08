"""This module provides the API for interfacing with hardware modules managed by a Teensy 4.0 microcontroller."""

from enum import IntEnum
from typing import TYPE_CHECKING, Any

import numpy as np
from ataraxis_time import PrecisionTimer
from scipy.optimize import curve_fit
from ataraxis_base_utilities import LogLevel, console
from ataraxis_data_structures import DataLogger, SharedMemoryArray
from ataraxis_communication_interface import (
    ModuleData,
    ModuleState,
    ModuleInterface,
    MicroControllerInterface,
)

# Prevents typing-related imports from being imported at runtime
if TYPE_CHECKING:
    from numpy.typing import NDArray

# Defines constants used in this module
_ZERO_LONG = np.uint32(0)
_ZERO_BYTE = np.uint8(0)
_BOOL_FALSE = np.bool(False)
_BOOL_TRUE = np.bool(True)
_FIVE_MICROLITERS = np.float64(5)

# Microcontroller parameters
_CONTROLLED_ID = np.uint8(111)
_CONTROLLER_PORT = "COM4"
_CONTROLLER_BUFFER_SIZE = 8192
_CONTROLLER_BAUDRATE = 115200
_CONTROLLER_KEEPALIVE_INTERVAL = 500

# Valve module calibration parameters
# The delay between calibration pulses in us. Should never be below 200000.
_VALVE_CALIBRAZTION_COUNT = np.uint16(200)  # The of calibration pulses to use at each calibration level.

# Maps right valve pulse durations in microseconds to the corresponding dispensed volume of fluid in microliters.
_RIGHT_VALVE_CALIBRATION_DATA = (
    (15000, 1.10),  # 15 ms dispenses 1.10 uL of fluid.
    (30000, 3.00),
    (45000, 6.25),
    (60000, 10.90),
)

# Same as above, but for the left valve
_LEFT_VALVE_CALIBRATION_DATA = (
    (15000, 0.86),
    (30000, 2.11),
    (45000, 4.24),
    (60000, 6.40),
)

# Lick module calibration parameters
# In 12-bit ADC units. Signals below this threshold are treated as noise and pulled to 0 (no signal) level.
_LICK_SIGNAL_THRESHOLD = np.uint16(500)

# In 12-bit ADC units. The level for classifying a sensor activation event as a lick. Any sensor-reported value above
# this threshold is considered a lick
_LICK_DETECTION_THRESHOLD = np.uint16(1000)

# In 12-bit ADC units. The minimum difference between two consecutive sensor readouts for the new readout to be
# reported to the PC. This ensures that the PC is only informed about significant voltage changes that usually
# correspond to major event transitions (no touch / touch / lick).
_LICK_DELTA_THRESHOLD = np.uint16(180)

# The number of analog pin readouts to average into the final sensor value. Larger values produce smoother data, but
# introduce detection latency. On Teensy controllers, the val=ue listed here is multiplied by 4 (e.g. averaging pool of
# 4 actually means 4 * 4 = 16 readouts).
_LICK_AVERAGING_POOL = np.uint8(2)

# The number of microseconds to delay between polling (checking) the lick sensor. A value of 1000 means 1 ms, which
# gives a polling rate of ~1000 HZ.
_LICK_POLLING_DELAY = np.uint32(1000)


class ModuleTypeCodes(IntEnum):
    """Stores the module type (family) codes used by the hardware modules supported by this library version."""

    VALVE_MODULE = 1
    LICK_MODULE = 2
    ANALOG_MODULE = 3


class _ValveStateCodes(IntEnum):
    """Stores the message state codes used by the ValveModule instances that require online processing at runtime."""

    VALVE_OPEN = 51
    VALVE_CLOSED = 52
    VALVE_CALIBRATED = 53


class _LickStateCodes(IntEnum):
    """Stores the message state codes used by the LickModule instances that require online processing at runtime."""

    VOLTAGE_READOUT_CHANGED = 51


class ValveInterface(ModuleInterface):
    """Interfaces with ValveModule instances running on Ataraxis MicroControllers.

    ValveModule instances control a solenoid valve to dispense precise volumes of fluid.

    Notes:
        This class is explicitly designed to work with the valves whose calibration curve is most closely approximated
        using a power law equation.

    Args:
        module_id: The unique identifier of the hardware module instance managed by this interface.
        valve_calibration_data: A tuple of tuples that contains the data required to map pulse duration to delivered
            fluid volume. Each sub-tuple should contain the integer that specifies the pulse duration in microseconds
            and a float that specifies the delivered fluid volume in microliters.
        debug: A boolean flag that configures the interface to dump certain data received from the microcontroller into
            the terminal. This is used during debugging and system calibration and should be disabled for most runtimes.

    Attributes:
        _scale_coefficient: Stores the power law scale coefficient derived from the calibration data.
        _nonlinearity_exponent: The intercept of the valve calibration curve. This is used to account for the fact that
            some valves may have a minimum open time or dispensed fluid volume, which is captured by the intercept.
            This improves the precision of fluid-volume-to-valve-open-time conversions.
        _debug: Stores the debug flag.
        _valve_tracker: Stores the SharedMemoryArray that tracks the total volume of fluid dispensed by the valve
            during runtime.
        _previous_state: Tracks the previous valve state as Open (True) or Closed (False). This is used to accurately
            track delivered fluid volumes each time the valve opens and closes.
        _cycle_timer: A PrecisionTimer instance initialized in the Communication process to track how long the valve
            stays open during cycling. This is used together with the _previous_state to determine the volume of fluid
            delivered by the valve during runtime.
        _previous_volume: Tracks the volume of fluid, in microliters, dispensed during the last dispense_volume()
            method runtime.
    """

    def __init__(
        self,
        module_id: np.uint8,
        valve_calibration_data: tuple[tuple[int | float, int | float], ...],
        *,
        debug: bool = False,
    ) -> None:
        error_codes = None
        # kOpen, kClosed, kCalibrated
        data_codes = {np.uint8(51), np.uint8(52), np.uint8(53)}  # kOpen, kClosed, kCalibrated

        self._debug: bool = debug

        # If the interface runs in the debug mode, expands the list of processed data codes to include all codes used
        # by the valve module.
        if debug:
            data_codes = {np.uint8(51), np.uint8(52), np.uint8(53)}

        super().__init__(
            module_type=np.uint8(ModuleTypeCodes.VALVE_MODULE),
            module_id=module_id,
            data_codes=data_codes,
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
        # fluid dispensed by the valve during runtime.
        self._valve_tracker: SharedMemoryArray = SharedMemoryArray.create_array(
            name=f"{self._module_type}_{self._module_id}_valve_tracker",
            prototype=np.zeros(shape=1, dtype=np.float64),
            exists_ok=True,
        )
        self._previous_state: bool = False
        self._cycle_timer: PrecisionTimer | None = None
        self._previous_volume: np.float64 = np.float64(0.0)

    def __del__(self) -> None:
        """Ensures the reward tracker is properly cleaned up when the class is garbage-collected."""
        self._valve_tracker.disconnect()
        self._valve_tracker.destroy()

    def initialize_remote_assets(self) -> None:
        """Connects to the reward tracker SharedMemoryArray and initializes the cycle PrecisionTimer from the
        Communication process.
        """
        self._valve_tracker.connect()
        self._cycle_timer = PrecisionTimer("us")

    def terminate_remote_assets(self) -> None:
        """Disconnects from the reward tracker SharedMemoryArray."""
        self._valve_tracker.disconnect()

    def process_received_data(self, message: ModuleData | ModuleState) -> None:
        """Processes incoming data sent by the module to the PC."""

        if message.event == _ValveStateCodes.VALVE_OPEN:
            if self._debug:
                console.echo("Valve Opened")

            # Resets the cycle timer each time the valve transitions to open state.
            if not self._previous_state:
                self._previous_state = True
                self._cycle_timer.reset()  # type: ignore[union-attr]

        elif message.event == _ValveStateCodes.VALVE_CLOSED:
            if self._debug:
                console.echo("Valve Closed")

            # Each time the valve transitions to closed state, records the period of time the valve was open and uses it
            # to estimate the volume of fluid delivered through the valve.
            if self._previous_state:
                self._previous_state = False
                open_duration = self._cycle_timer.elapsed  # type: ignore[union-attr]

                # Accumulates delivered fluid volumes into the tracker.
                delivered_volume = np.float64(
                    self._scale_coefficient * np.power(open_duration, self._nonlinearity_exponent)
                )
                previous_volume = np.float64(self._valve_tracker[0])
                new_volume = previous_volume + delivered_volume
                # noinspection PyTypeChecker
                self._valve_tracker[0] = new_volume
        elif message.event == _ValveStateCodes.VALVE_CALIBRATED:
            console.echo("Valve Calibration: Complete")

    def dispense_volume(self, volume: np.float64 = _FIVE_MICROLITERS, noblock: np.bool = _BOOL_TRUE) -> None:
        """Delivers teh requested volume of fluid through the valve.

        Args:
            volume: The volume of fluid to dispense, in microliters.
            noblock: Determines whether the command should block the microcontroller while the valve is kept open.
        """
        # If necessary, reconfigures the valve to deliver the requested volume of fluid
        if volume != self._previous_volume:
            # The minimum valid pulse duration is hardcoded at 15 ms as this is the lower boundary used during
            # calibration.
            min_pulse_duration = 15000  # microseconds
            min_dispensed_volume = self._scale_coefficient * np.power(min_pulse_duration, self._nonlinearity_exponent)

            if volume < min_dispensed_volume:
                message = (
                    f"The requested volume {volume} uL is too small to be reliably dispensed by the ValveModule "
                    f"{self._module_id}. Specifically, the smallest volume of fluid the valve can reliably dispense is "
                    f"{min_dispensed_volume} uL."
                )
                console.error(message=message, error=ValueError)

            # Inverts the power-law calibration to get the pulse duration.
            pulse_duration = (volume / self._scale_coefficient) ** (1.0 / self._nonlinearity_exponent)
            pulse_duration_us = np.uint32(np.ceil(pulse_duration))

            # Updates the runtime configuration of the valve to deliver the requested volume of fluid.
            self.send_parameters(parameter_data=(pulse_duration_us, _VALVE_CALIBRAZTION_COUNT))

        # Instructs the valve to execute the command
        self.send_command(command=np.uint8(1), noblock=noblock, repetition_delay=_ZERO_LONG)

    def toggle(self, state: bool) -> None:
        """Permanently opens or closes the valve.

        Args:
            state: The desired state of the valve. True means the valve is open; False means the valve is closed.
        """
        self.send_command(command=np.uint8(2 if state else 3), noblock=_BOOL_FALSE, repetition_delay=_ZERO_LONG)

    def calibrate(self, pulse_duration: np.uint32) -> None:
        """Opens and closes the valve 200 times, keeping the valve open for the requested number of microseconds during
        each cycle.

        This command is used to build the calibration map (curve) of the valve that matches each tested 'pulse_duration'
        to the volume of fluid dispensed during the time the valve is open.

        Notes:
            When activated, this command blocks in-place until the calibration cycle is completed. Currently, there
            is no way to interrupt the command, and it may take a long period of time (minutes) to complete.
        """
        self.send_parameters(parameter_data=(pulse_duration, _VALVE_CALIBRAZTION_COUNT))
        self.send_command(command=np.uint8(4), noblock=_BOOL_FALSE, repetition_delay=_ZERO_LONG)

    @property
    def scale_coefficient(self) -> np.float64:
        """Returns the scaling coefficient (A) derived during the power-law calibration.

        In the calibration model, fluid_volume = A * (pulse_duration)^B, this coefficient
        converts pulse duration (in microseconds) into the appropriate fluid volume (in microliters)
        when used together with the nonlinearity exponent.
        """
        return self._scale_coefficient

    @property
    def nonlinearity_exponent(self) -> np.float64:
        """Returns the nonlinearity exponent (B) derived during the power-law calibration.

        In the calibration model, fluid_volume = A * (pulse_duration)^B, this exponent indicates
        the degree of nonlinearity in how the dispensed volume scales with the valve`s pulse duration.
        For example, an exponent of 1 would indicate a linear relationship.
        """
        return self._nonlinearity_exponent

    @property
    def dispensed_volume(self) -> np.float64:
        """Returns the total volume of fluid, in microliters, delivered by the valve during the current runtime."""
        return self._valve_tracker[0]  # type: ignore[no-any-return]


class LickInterface(ModuleInterface):
    """Interfaces with LickModule instances running on Ataraxis MicroControllers.

    LickModule monitor conductive lick sensors to detect animal interactions with fluid dispensing tubes (lick-ports).

    Args:
        module_id: The unique identifier for the LickModule instance.
        debug: A boolean flag that configures the interface to dump certain data received from the microcontroller into
            the terminal. This is used during debugging and system calibration and should be disabled for most runtimes.

    Attributes:
        _lick_threshold: Stores the threshold voltage to use for detecting a tongue contact.
        _volt_per_adc_unit: Stores the conversion factor to translate the raw analog values recorded by the 12-bit ADC
            into voltage in Volts.
        _debug: Stores the debug flag.
        _lick_tracker: Stores the SharedMemoryArray that stores the total number of licks detected since class
            initialization.
        _previous_readout_zero: Stores a boolean indicator of whether the previous voltage readout was a 0-value.
        _once: Ensures that the sensor detection configuration is applied exactly once per instance life cycle.
    """

    def __init__(self, module_id: np.uint8, *, debug: bool = False) -> None:
        data_codes: set[np.uint8] = {np.uint8(51)}  # kChanged
        self._debug: bool = debug

        # Initializes the subclassed ModuleInterface using the input instance data. Type data is hardcoded.
        super().__init__(
            module_type=np.uint8(ModuleTypeCodes.LICK_MODULE),
            module_id=module_id,
            data_codes=data_codes,
            error_codes=None,
        )

        self._lick_threshold: np.uint16 = _LICK_DETECTION_THRESHOLD

        # Statically computes the voltage resolution of each analog step, assuming a 3.3V ADC with 12-bit resolution.
        self._volt_per_adc_unit: np.float64 = np.round(a=np.float64(3.3 / (2**12)), decimals=8)

        # Precreates a shared memory array used to track and share the total number of licks recorded by the sensor
        # since class initialization.
        self._lick_tracker: SharedMemoryArray = SharedMemoryArray.create_array(
            name=f"{self._module_type}_{self._module_id}_lick_tracker",
            prototype=np.zeros(shape=1, dtype=np.uint64),
            exists_ok=True,
        )

        # Precreates storage variables used to prevent excessive lick reporting
        self._previous_readout_zero: bool = False

        self._once: bool = True

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
        """Processes incoming data sent by the module to the PC."""
        # Currently, only code 51 messages are passed to this method. From each, extracts the detected voltage level.
        detected_voltage: np.uint16 = message.data_object  # type: ignore[union-attr, assignment]

        # If the class is initialized in debug mode, prints each received voltage level to the terminal.
        if self._debug:
            console.echo(f"Lick ADC signal: {detected_voltage}")

        # Since the sensor is pulled to 0 to indicate lack of tongue contact, a zero-readout necessarily means no
        # lick. Sets zero-tracker to 1 to indicate that a zero-state has been encountered
        if detected_voltage == 0:
            self._previous_readout_zero = True
            return

        # If the voltage level exceeds the lick threshold and this is the first time the threshold is exceeded since
        # the last zero-value, reports it to other processes. The Threshold is inclusive. This exploits the fact that
        # every pair of licks has to be separated by a zero-value (lack of tongue contact). So, to properly report the
        # licks, only does it once per encountering a zero-value.
        if detected_voltage >= self._lick_threshold and self._previous_readout_zero:
            # Increments the lick count and updates the tracker array with new data
            count = self._lick_tracker[0]
            count += 1
            self._lick_tracker[0] = count

            # This disables further reports until the sensor sends a zero-value again
            self._previous_readout_zero = False

    def check_state(self, repetition_delay: np.uint32 = _LICK_POLLING_DELAY) -> None:
        """Checks and reports the voltage level detected by the lick sensor to the PC.

        Args:
            repetition_delay: The time, in microseconds, to delay before repeating the command. When set to 0, the
                command only runs once.
        """
        # Applies sensor configuration parameters the first time the method is called
        if self._once:
            self.send_parameters(parameter_data=(_LICK_SIGNAL_THRESHOLD, _LICK_DELTA_THRESHOLD, _LICK_AVERAGING_POOL))
            self._once = False
        self.send_command(command=np.uint8(1), noblock=_BOOL_FALSE, repetition_delay=repetition_delay)

    def get_adc_units_from_volts(self, voltage: float) -> np.uint16:
        """Converts the input voltage to raw analog units of 12-bit Analog-to-Digital-Converter (ADC).

        Notes:
            This method assumes a 3.3V ADC with 12-bit resolution.

        Args:
            voltage: The voltage to convert to raw analog units, in Volts.

        Returns:
            The raw analog units of 12-bit ADC for the input voltage.
        """
        return np.uint16(np.round(voltage / self._volt_per_adc_unit))

    @property
    def volts_per_adc_unit(self) -> np.float64:
        """Returns the conversion factor to translate the raw analog values recorded by the 12-bit ADC into voltage in
        Volts.
        """
        return self._volt_per_adc_unit

    @property
    def lick_count(self) -> np.uint64:
        """Returns the total number of licks detected by the module since runtime onset."""
        return self._lick_tracker[0]  # type: ignore[no-any-return]

    @property
    def lick_threshold(self) -> np.uint16:
        """Returns the voltage threshold, in raw ADC units of a 12-bit Analog-to-Digital voltage converter that is
        interpreted as the animal licking the sensor.
        """
        return self._lick_threshold


class AnalogInterface(ModuleInterface):
    """Interfaces with AnalogModule instances running on Ataraxis MicroControllers.
       

    Args:
        module_id: The unique identifier for the AnalogModule instance.

    Attributes:
        _volt_per_adc_unit: Stores the conversion factor to translate the raw analog values recorded by the 12-bit ADC
            into voltage in Volts."""

    def __init__(self, module_id: np.uint8, debug: bool = False) -> None:
        data_codes: set[np.uint8] = {np.uint8(51)}  # kNonZero
        self._debug: bool = debug

        # Initializes the subclassed ModuleInterface using the input instance data. Type data is hardcoded.
        super().__init__(
            module_type=np.uint8(ModuleTypeCodes.ANALOG_MODULE),  # ModuleTypeCodes.ANALOG_MODULE
            module_id=module_id,
            data_codes=data_codes,
            error_codes=None,
        )

        self._volt_per_adc_unit: np.float64 = np.round(a=np.float64(3.3 / (2**12)), decimals=8)

        # We don't need any trackers for now, just need to initialize one for PC to detect the module
        self._analog_tracker: SharedMemoryArray = SharedMemoryArray.create_array(
            name=f"{self._module_type}_{self._module_id}_analog_tracker",
            prototype=np.zeros(shape=1, dtype=np.uint64),
            exists_ok=True,
        )

    def __del__(self) -> None:
        """Ensures the lick_tracker is properly cleaned up when the class is garbage-collected."""
        self._analog_tracker.disconnect()
        self._analog_tracker.destroy()

    def initialize_remote_assets(self) -> None:
        """Connects to the SharedMemoryArray used to communicate lick status to other processes."""
        self._analog_tracker.connect()

    def terminate_remote_assets(self) -> None:
        """Disconnects from the lick-tracker SharedMemoryArray."""
        self._analog_tracker.disconnect()  # Does not destroy the array to support start / stop cycling.

    def process_received_data(self, message: ModuleData | ModuleState) -> None:
        """Processes incoming data sent by the module to the PC."""
        # Currently, only code 51 messages are passed to this method. From each, extracts the detected voltage level.
        detected_voltage: np.uint16 = message.data_object  # type: ignore[union-attr, assignment]

        # If the class is initialized in debug mode, prints each received voltage level to the terminal.
        if self._debug:
            console.echo(f"Analog ADC signal: {detected_voltage}")



class AMCInterface:
    """Interfaces with all Ataraxis Micro Controller (AMC) interfaces used to acquire non-video behavior data.

    This class interfaces with the single AMC device used to both record the behavior data and interface with the
    interactive components of the experiment environment.

    Notes:
        Calling the initializer does not start the underlying processes. Use the start() method before issuing other
        commands to properly initialize all remote processes.

    Args:
        data_logger: The initialized DataLogger instance used to log the data generated by the managed microcontrollers.
            For most runtimes, this argument is resolved by the _MesoscopeExperiment or _BehaviorTraining classes that
            initialize this class.

    Attributes:
        _started: Tracks whether the VR system and experiment runtime are currently running.
        _controller: The main interface for the Ataraxis Micro Controller (AMC) device managing the hardware modules.

    """

    def __init__(self, data_logger: DataLogger) -> None:
        # Initializes the start state tracker first
        self._started: bool = False

        # Module interfaces:
        self.left_valve = ValveInterface(
            module_id=np.uint8(1),
            valve_calibration_data=_LEFT_VALVE_CALIBRATION_DATA,
            debug=True
        )

        self.right_valve = ValveInterface(
            module_id=np.uint8(2),
            valve_calibration_data=_RIGHT_VALVE_CALIBRATION_DATA,
            debug=False,
        )

        self.left_lick_sensor = LickInterface(
            module_id=np.uint8(1),
            debug=False,
        )

        self.right_lick_sensor = LickInterface(
            module_id=np.uint8(2),
            debug=False,
        )

        self.analog_input = AnalogInterface(
            module_id=np.uint8(1),
            debug=False,
        )

        self.module_interfaces = (self.left_valve, self.right_valve, self.left_lick_sensor, self.right_lick_sensor, self.analog_input)

        # Main interface:
        self._controller: MicroControllerInterface = MicroControllerInterface(
            controller_id=_CONTROLLED_ID,
            buffer_size=_CONTROLLER_BUFFER_SIZE,
            port=_CONTROLLER_PORT,
            data_logger=data_logger,
            module_interfaces=self.module_interfaces,
            baudrate=_CONTROLLER_BAUDRATE,
            keepalive_interval=_CONTROLLER_KEEPALIVE_INTERVAL,
        )

    def __del__(self) -> None:
        """Releases all reserved resources before the object is garbage-collected."""
        self.stop()

    def start(self) -> None:
        """Starts the microcontroller communication process and verifies the connected microcontroller's configuration.

        This method sets up the necessary assets that enable microcontroller-PC communication. Calling this method is
        a prerequisite for using all other microcontroller and module interface methods.
        """
        # Prevents executing this method if the MicroControllers are already running.
        if self._started:
            return

        message = "Initializing the Ataraxis Micro Controller (AMC) Interface..."
        console.echo(message=message, level=LogLevel.INFO)

        # Starts all microcontroller communication process
        self._controller.start()

        # The setup procedure is complete.
        self._started = True

        message = "Ataraxis Micro Controller (AMC) Interface: Initialized."
        console.echo(message=message, level=LogLevel.SUCCESS)

    def stop(self) -> None:
        """Stops the microcontroller communication process and releases the resources reserved by the process during
        runtime.
        """
        # Prevents stopping an already stopped VR process.
        if not self._started:
            return

        message = "Terminating the Ataraxis Micro Controller (AMC) Interface..."
        console.echo(message=message, level=LogLevel.INFO)

        # Resets the _started tracker
        self._started = False

        # Stops the microcontroller interface. This directly shuts down and resets all managed hardware modules.
        self._controller.stop()

        message = "Ataraxis Micro Controller (AMC) Interface: Terminated."
        console.echo(message=message, level=LogLevel.SUCCESS)

    def connect_to_smh(self) -> None:
        """Establishes connections to SharedMemoryArray for all modules. This function needs to be called after start()."""
        for module in self.module_interfaces:
            module.initialize_remote_assets()

    def disconnect_to_smh(self) -> None:
        """Disconnects from SharedMemoryArray for all modules. This function needs to be called before stop()."""
        for module in self.module_interfaces:
            module.terminate_remote_assets()

    @property
    def controller_id(self) -> int:
        """Returns the unique identifier code of the microcontroller."""
        return int(_CONTROLLED_ID)

