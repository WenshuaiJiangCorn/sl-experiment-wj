from enum import IntEnum

import numpy as np
from _typeshed import Incomplete
from numpy.typing import NDArray as NDArray
from ataraxis_time import PrecisionTimer
from ataraxis_data_structures import DataLogger, SharedMemoryArray
from ataraxis_communication_interface import (
    ModuleData as ModuleData,
    ModuleState as ModuleState,
    ModuleInterface,
    MicroControllerInterface,
)

_ZERO_LONG: Incomplete
_ZERO_BYTE: Incomplete
_BOOL_FALSE: Incomplete
_BOOL_TRUE: Incomplete
_FIVE_MICROLITERS: Incomplete
_CONTROLLED_ID: Incomplete
_CONTROLLER_PORT: str
_CONTROLLER_BUFFER_SIZE: int
_CONTROLLER_BAUDRATE: int
_CONTROLLER_KEEPALIVE_INTERVAL: int
_VALVE_CALIBRATION_DELAY: Incomplete
_VALVE_CALIBRAZTION_COUNT: Incomplete
_RIGHT_VALVE_CALIBRATION_DATA: Incomplete
_LEFT_VALVE_CALIBRATION_DATA: Incomplete
_LICK_SIGNAL_THRESHOLD: Incomplete
_LICK_DETECTION_THRESHOLD: Incomplete
_LICK_DELTA_THRESHOLD: Incomplete
_LICK_AVERAGING_POOL: Incomplete
_LICK_POLLING_DELAY: Incomplete

class ModuleTypeCodes(IntEnum):
    VALVE_MODULE = 101
    LICK_MODULE = 102

class _ValveStateCodes(IntEnum):
    VALVE_OPEN = 52
    VALVE_CLOSED = 53
    VALVE_CALIBRATED = 54

class _LickStateCodes(IntEnum):
    VOLTAGE_READOUT_CHANGED = 51

class ValveInterface(ModuleInterface):
    _debug: bool
    _scale_coefficient: np.float64
    _nonlinearity_exponent: np.float64
    _valve_tracker: SharedMemoryArray
    _previous_state: bool
    _cycle_timer: PrecisionTimer | None
    _previous_volume: np.float64
    def __init__(
        self,
        module_id: np.uint8,
        valve_calibration_data: tuple[tuple[int | float, int | float], ...],
        *,
        debug: bool = False,
    ) -> None: ...
    def __del__(self) -> None: ...
    def initialize_remote_assets(self) -> None: ...
    def terminate_remote_assets(self) -> None: ...
    def process_received_data(self, message: ModuleData | ModuleState) -> None: ...
    def dispense_volume(self, volume: np.float64 = ..., noblock: np.bool = ...) -> None: ...
    def toggle(self, state: bool) -> None: ...
    def calibrate(self, pulse_duration: np.uint32) -> None: ...
    @property
    def scale_coefficient(self) -> np.float64: ...
    @property
    def nonlinearity_exponent(self) -> np.float64: ...
    @property
    def dispensed_volume(self) -> np.float64: ...

class LickInterface(ModuleInterface):
    _debug: bool
    _lick_threshold: np.uint16
    _volt_per_adc_unit: np.float64
    _lick_tracker: SharedMemoryArray
    _previous_readout_zero: bool
    _once: bool
    def __init__(self, module_id: np.uint8, *, debug: bool = False) -> None: ...
    def __del__(self) -> None: ...
    def initialize_remote_assets(self) -> None: ...
    def terminate_remote_assets(self) -> None: ...
    def process_received_data(self, message: ModuleData | ModuleState) -> None: ...
    def check_state(self, repetition_delay: np.uint32 = ...) -> None: ...
    def get_adc_units_from_volts(self, voltage: float) -> np.uint16: ...
    @property
    def volts_per_adc_unit(self) -> np.float64: ...
    @property
    def lick_count(self) -> np.uint64: ...
    @property
    def lick_threshold(self) -> np.uint16: ...

class AMCInterface:
    _started: bool
    left_valve: Incomplete
    right_valve: Incomplete
    left_lick_sensor: Incomplete
    right_lick_sensor: Incomplete
    _controller: MicroControllerInterface
    def __init__(self, data_logger: DataLogger) -> None: ...
    def __del__(self) -> None: ...
    def start(self) -> None: ...
    def stop(self) -> None: ...
    @property
    def controller_id(self) -> int: ...
