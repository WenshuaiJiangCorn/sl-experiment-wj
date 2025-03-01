"""This module provides the main MesoscopeExperiment and BehavioralTraining classes that abstract working with Sun
lab's Mesoscope-VR system and SessionData class that abstracts working with acquired experimental data."""

import os
import warnings
from pathlib import Path
import tempfile

import numpy as np
from numpy.typing import NDArray
from numpy.lib.npyio import NpzFile
import copy

from .module_interfaces import (
    TTLInterface,
    LickInterface,
    BreakInterface,
    ValveInterface,
    ScreenInterface,
    TorqueInterface,
    EncoderInterface,
)
from ataraxis_base_utilities import console, ensure_directory_exists, LogLevel
from ataraxis_data_structures import DataLogger, LogPackage, YamlConfig, SharedMemoryArray
from ataraxis_time.time_helpers import get_timestamp
from ataraxis_communication_interface import MicroControllerInterface, MQTTCommunication
from ataraxis_video_system import (
    VideoSystem,
    CameraBackends,
    VideoFormats,
    VideoCodecs,
    GPUEncoderPresets,
    InputPixelFormats,
    OutputPixelFormats,
)
from ataraxis_time import PrecisionTimer

from .zaber_bindings import ZaberConnection, ZaberAxis
from .transfer_tools import transfer_directory
from .packaging_tools import calculate_directory_checksum
from .data_preprocessing import interpolate_data, process_mesoscope_directory
from .visualizers import BehaviorVisualizer
from concurrent.futures import ProcessPoolExecutor, as_completed
import shutil
from dataclasses import dataclass
import polars as pl
from typing import Any
from tqdm import tqdm
from pynput import keyboard


@dataclass()
class _ZaberPositions(YamlConfig):
    """This class is used to save and restore Zaber motor positions between sessions by saving them as .yaml file.

    The class is specifically designed to store, save, and load the positions of the LickPort and HeadBar motors
    (axes). It is used to both store Zaber motor positions for each session for future analysis and to (optionally)
    restore the same Zaber motor positions across consecutive experimental sessions for the same project and animal
    combination.

    Notes:
        This class is designed to be used internally by other classes from this library. Do not instantiate or load
        this class from .yaml files manually. Do not modify the data stored inside the .yaml file unless you know what
        you are doing.

        All positions are saved using native motor units. All class fields initialize to default placeholders that are
        likely NOT safe to apply to the VR system. Do not apply the positions loaded from the file unless you are
        certain they are safe to use.

        Exercise caution when working with Zaber motors. The motors are powerful enough to damage the surrounding
        equipment and manipulated objects.
    """

    headbar_z: int = 0
    """The absolute position, in native motor units, of the HeadBar z-axis motor."""
    headbar_pitch: int = 0
    """The absolute position, in native motor units, of the HeadBar pitch-axis motor."""
    headbar_roll: int = 0
    """The absolute position, in native motor units, of the HeadBar roll-axis motor."""
    lickport_z: int = 0
    """The absolute position, in native motor units, of the LickPort z-axis motor."""
    lickport_x: int = 0
    """The absolute position, in native motor units, of the LickPort x-axis motor."""
    lickport_y: int = 0
    """The absolute position, in native motor units, of the LickPort y-axis motor."""


@dataclass()
class _LickTrainingDescriptor(YamlConfig):
    """This class is used to save the description information specific to lick training sessions as a .yaml file.

    Notes:
        Primarily, descriptors are used to store session-specific information that may be challenging or impossible
        to extract from the logged data in a format easily accessible by experimenters. This allows experimenters to
        quickly check the session description data file and reference the information stored inside the file.

        Do not instantiate or use this class manually. It is designed to be written by the runtime management class
        during its stop() method runtime.
    """

    session_type: str = "lick_training"
    """
    The type of the session. Currently, the following options are supported: "lick_training", "run_training", and 
    "mesoscope_experiment". This field is hardcoded and should not be modified.
    """
    dispensed_water_volume_ul: float = 0.0
    """Stores the total water volume, in microliters, dispensed during runtime."""
    average_reward_delay_s: int = 12
    """Stores the center-point for the reward delay distribution, in seconds."""
    maximum_deviation_from_average_s: int = 6
    """Stores the deviation value, in seconds, used to determine the upper and lower bounds for the reward delay 
    distribution."""
    maximum_water_volume_ml: float = 1.0
    """Stores the maximum volume of water the system is allowed to dispensed during training."""
    maximum_training_time_m: int = 40
    """Stores the maximum time, in minutes, the system is allowed to run the training for."""
    experimenter_notes: str = "Replace this with your notes."
    """This field is not set during runtime. It is expected that each experimenter will replace this field with their 
    notes made during runtime."""


@dataclass()
class _RunTrainingDescriptor(YamlConfig):
    session_type: str = "run_training"
    """
    The type of the session. Currently, the following options are supported: "lick_training", "run_training", and 
    "mesoscope_experiment". This field is hardcoded and should not be modified.
    """
    dispensed_water_volume_ul: float = 0.0
    """Stores the total water volume, in microliters, dispensed during runtime."""
    maximum_training_time_m: int = 40
    """Stores the maximum time, in minutes, the system is allowed to run the training for."""
    experimenter_notes: str = "Replace this with your notes."
    """This field is not set during runtime. It is expected that each experimenter will replace this field with their 
    notes made during runtime."""


@dataclass()
class _MesoscopeExperimentDescriptor(YamlConfig):
    session_type: str = "mesoscope_experiment"
    """
    The type of the session. Currently, the following options are supported: "lick_training", "run_training", and 
    "mesoscope_experiment". This field is hardcoded and should not be modified.
    """
    dispensed_water_volume_ul: float = 0.0
    """Stores the total water volume, in microliters, dispensed during runtime."""
    experimenter_notes: str = "Replace this with your notes."
    """This field is not set during runtime. It is expected that each experimenter will replace this field with their 
    notes made during runtime."""


class _HeadBar:
    """Interfaces with Zaber motors that control the position of the HeadBar manipulator arm.

    This class abstracts working with Zaber motors that move the HeadBar in Z, Pitch, and Roll axes. It is used
    by the major runtime classes, such as MesoscopeExperiment, to interface with HeadBar motors. The class is designed
    to transition the HeadBar between a small set of predefined states and should not be used directly by the user.

    Notes:
        This class does not contain the guards that notify users about risks associated with moving the motors. Do not
        use any methods from this class unless you know what you are doing. It is very easy to damage the motors, the
        mesoscope, or harm the animal.

        To fine-tune the position of any HeadBar motors in real time, use the main Zaber interface from the VRPC.

        Unless you know that the motors are homed and not parked, always call the prepare_motors() method before
        calling any other methods. Otherwise, Zaber controllers will likely ignore the issued commands.

    Args:
        headbar_port: The USB port used by the HeadBar Zaber motor controllers (devices).
        zaber_positions_path: The path to the zaber_positions.yaml file that stores the motor positions saved during
            previous runtime.

    Attributes:
        _headbar: Stores the Connection class instance that manages the USB connection to a daisy-chain of Zaber
            devices (controllers) that allow repositioning the headbar holder.
        _headbar_z: The ZaberAxis class instance for the HeadBar z-axis motor.
        _headbar_pitch: The ZaberAxis class instance for the HeadBar pitch-axis motor.
        _headbar_roll: The ZaberAxis class instance for the HeadBar roll-axis motor.
        _previous_positions: An instance of _ZaberPositions class that stores the positions of HeadBar motors during a
           previous runtime. If this data is not available, this attribute is set to None to indicate there are no
           previous positions to use.
    """

    def __init__(self, headbar_port: str, zaber_positions_path: Path) -> None:
        # HeadBar controller (zaber). This is an assembly of 3 zaber controllers (devices) that allow moving the
        # headbar attached to the mouse in Z, Roll, and Pitch dimensions. Note, this assumes that the chaining order of
        # individual zaber devices is fixed and is always Z-Pitch-Roll.
        self._headbar: ZaberConnection = ZaberConnection(port=headbar_port)
        self._headbar.connect()
        self._headbar_z: ZaberAxis = self._headbar.get_device(0).axis
        self._headbar_pitch: ZaberAxis = self._headbar.get_device(1).axis
        self._headbar_roll: ZaberAxis = self._headbar.get_device(2).axis

        # If the previous positions path points to an existing .yaml file, loads the data from the file into
        # _ZaberPositions instance. Otherwise, sets the previous_positions attribute to None to indicate there are no
        # previous positions.
        self._previous_positions: None | _ZaberPositions = None
        if zaber_positions_path.exists():
            self._previous_positions = _ZaberPositions.from_yaml(zaber_positions_path)

    def restore_position(self, wait_until_idle: bool = True) -> None:
        """Restores the HeadBar motor positions to the states recorded at the end of the previous runtime.

        For most runtimes, this method is used to restore the HeadBar to the state used during a previous experiment or
        training session for each animal. Since all animals are slightly different, the optimal HeadBar positions will
        vary slightly for each animal.

        Notes:
            If previous positions are not available, the method falls back to moving the HeadBar motors to the general
            'mounting' positions saved in the non-volatile memory of each motor controller. These positions are designed
            to work for most animals and provide an initial HeadBar position for the animal to be mounted into the VR
            rig.

            When used together with the LickPort class, this method should always be called before the similar method
            from the LickPort class.

            This method moves all HeadBar axes in-parallel to optimize runtime speed.

        Args:
            wait_until_idle: Determines whether to block in-place until all motors finish moving or to return without
                waiting for the motors to stop moving. This is primarily used to move multiple motor groups at the same
                time.
        """
        # If the positions are not available, warns the user and sets the motors to the 'generic' mount position.
        if self._previous_positions is None:
            message = (
                "No previous positions found when attempting to restore HeadBar to the previous runtime state. Setting "
                "the HeadBar motors to the default animal mounting positions loaded from motor controller non-volatile "
                "memory."
            )
            console.echo(message=message, level=LogLevel.ERROR)
            self._headbar_z.move(amount=self._headbar_z.mount_position, absolute=True, native=True)
            self._headbar_pitch.move(amount=self._headbar_pitch.mount_position, absolute=True, native=True)
            self._headbar_roll.move(amount=self._headbar_roll.mount_position, absolute=True, native=True)
        else:
            # Otherwise, restores Zaber positions.
            self._headbar_z.move(amount=self._previous_positions.headbar_z, absolute=True, native=True)
            self._headbar_pitch.move(amount=self._previous_positions.headbar_pitch, absolute=True, native=True)
            self._headbar_roll.move(amount=self._previous_positions.headbar_roll, absolute=True, native=True)

        # If requested, waits for the motors to finish moving before returning to caller. Otherwise, returns
        # without waiting for the motors to stop moving. The latter case is used to issue commands to multiple motor
        # groups at the same time.
        if wait_until_idle:
            self.wait_until_idle()

    def prepare_motors(self, wait_until_idle: bool = True) -> None:
        """Unparks and homes all HeadBar motors.

        This method should be used at the beginning of each runtime (experiment, training, etc.) to ensure all HeadBar
        motors can be moved (are not parked) and have a stable point of reference. The motors are left at their
        respective homing positions at the end of this method's runtime, and it is assumed that a different class
        method is called after this method to set the motors into the desired position.

        Notes:
            This method moves all HeadBar axes in-parallel to optimize runtime speed.

        Args:
            wait_until_idle: Determines whether to block in-place until all motors finish moving or to return without
                waiting for the motors to stop moving. This is primarily used to move multiple motor groups at the same
                time.
        """

        # Unparks all motors.
        self._headbar_z.unpark()
        self._headbar_pitch.unpark()
        self._headbar_roll.unpark()

        # Homes all motors in-parallel.
        self._headbar_z.home()
        self._headbar_pitch.home()
        self._headbar_roll.home()

        # If requested, waits for the motors to finish moving before returning to caller. Otherwise, returns
        # without waiting for the motors to stop moving. The latter case is used to issue commands to multiple motor
        # groups at the same time.
        if wait_until_idle:
            self.wait_until_idle()

    def park_position(self, wait_until_idle: bool = True) -> None:
        """Moves all HeadBar motors to their parking positions and parks (locks) them preventing future movements.

        This method should be used at the end of each runtime (experiment, training, etc.) to ensure all HeadBar motors
        are positioned in a way that guarantees that they can be homed during the next runtime.

        Notes:
            The motors are moved to the parking positions stored in the non-volatile memory of each motor controller. If
            this class is used together with the LickPort class, this method should always be called before the similar
            method from the LickPort class.

            This method moves all HeadBar axes in-parallel to optimize runtime speed.

        Args:
            wait_until_idle: Determines whether to block in-place until all motors finish moving or to return without
                waiting for the motors to stop moving. This is primarily used to move multiple motor groups at the same
                time.
        """

        # Moves all HeadBar motors to their parking positions
        self._headbar_z.move(amount=self._headbar_z.park_position, absolute=True, native=True)
        self._headbar_pitch.move(amount=self._headbar_pitch.park_position, absolute=True, native=True)
        self._headbar_roll.move(amount=self._headbar_roll.park_position, absolute=True, native=True)

        # If requested, waits for the motors to finish moving before returning to caller. Otherwise, returns
        # without waiting for the motors to stop moving. The latter case is used to issue commands to multiple motor
        # groups at the same time.
        if wait_until_idle:
            self.wait_until_idle()

    def calibrate_position(self, wait_until_idle: bool = True) -> None:
        """Moves all HeadBar motors to the water valve calibration position.

        This position is stored in the non-volatile memory of each motor controller. This position is used during the
        water valve calibration to provide experimenters with easier access to the LickPort tube.

        Notes:
            This method moves all HeadBar axes in-parallel to optimize runtime speed.

        Args:
            wait_until_idle: Determines whether to block in-place until all motors finish moving or to return without
                waiting for the motors to stop moving. This is primarily used to move multiple motor groups at the same
                time.
        """
        # Moves all HeadBar motors to their calibration positions
        self._headbar_z.move(amount=self._headbar_z.valve_position, absolute=True, native=True)
        self._headbar_pitch.move(amount=self._headbar_pitch.valve_position, absolute=True, native=True)
        self._headbar_roll.move(amount=self._headbar_roll.valve_position, absolute=True, native=True)

        # If requested, waits for the motors to finish moving before returning to caller. Otherwise, returns
        # without waiting for the motors to stop moving. The latter case is used to issue commands to multiple motor
        # groups at the same time.
        if wait_until_idle:
            self.wait_until_idle()

    def mount_position(self, wait_until_idle: bool = True) -> None:
        """Moves all HeadBar motors to the animal mounting position.

        This position is stored in the non-volatile memory of each motor controller. This position is used when the
        animal is mounted into the VR rig to provide the experimenter with easy access to the HeadBar holder.

        Notes:
            This method moves all HeadBar axes in-parallel to optimize runtime speed.

        Args:
            wait_until_idle: Determines whether to block in-place until all motors finish moving or to return without
                waiting for the motors to stop moving. This is primarily used to move multiple motor groups at the same
                time.
        """
        # Moves all motors to their mounting positions
        self._headbar_z.move(amount=self._headbar_z.mount_position, absolute=True, native=True)
        self._headbar_pitch.move(amount=self._headbar_pitch.mount_position, absolute=True, native=True)
        self._headbar_roll.move(amount=self._headbar_roll.mount_position, absolute=True, native=True)

        # If requested, waits for the motors to finish moving before returning to caller. Otherwise, returns
        # without waiting for the motors to stop moving. The latter case is used to issue commands to multiple motor
        # groups at the same time.
        if wait_until_idle:
            self.wait_until_idle()

    def get_positions(self) -> tuple[int, int, int]:
        """Returns the current position of all HeadBar motors in native motor units.

        The positions are returned in the order of : Z, Pitch, and Roll. These positions can be saves as a
        zaber_positions.yaml file to be used during the following runtimes.
        """
        return (
            int(self._headbar_z.get_position(native=True)),
            int(self._headbar_pitch.get_position(native=True)),
            int(self._headbar_roll.get_position(native=True)),
        )

    def wait_until_idle(self) -> None:
        """This method blocks in-place while at least one motor in the managed motor group is moving.

        Primarily, this method is used to issue commands to multiple motor groups and then block until all motors in
        all groups finish moving. This optimizes the overall time taken to move the motors.
        """
        # Waits for the motors to finish moving.
        while self._headbar_z.is_busy or self._headbar_pitch.is_busy or self._headbar_roll.is_busy:
            pass

    def disconnect(self) -> None:
        """Disconnects from the access port of the motor group.

        This method should be called after the motors are parked (moved to their final parking position) to release
        the connection resources. If this method is not called, the runtime will NOT be able to terminate.

        Notes:
            Calling this method will execute the motor parking sequence, which involves moving the motors to their
            parking position. Make sure there are no animals mounted on the rig and that the mesoscope objective is
            removed from the rig before executing this command.
        """
        message = f"HeadBar motor connection: Terminated"
        console.echo(message, LogLevel.SUCCESS)
        self._headbar.disconnect()


class _LickPort:
    """Interfaces with Zaber motors that control the position of the LickPort manipulator arm.

    This class abstracts working with Zaber motors that move the LickPort in Z, X, and Y axes. It is used
    by the major runtime classes, such as MesoscopeExperiment, to interface with LickPort motors. The class is designed
    to transition the LickPort between a small set of predefined states and should not be used directly by the user.

    Notes:
        This class does not contain the guards that notify users about risks associated with moving the motors. Do not
        use any methods from this class unless you know what you are doing. It is very easy to damage the motors, the
        mesoscope, or harm the animal.

        To fine-tune the position of any LickPort motors in real time, use the main Zaber interface from the VRPC.

        Unless you know that the motors are homed and not parked, always call the prepare_motors() method before
        calling any other methods. Otherwise, Zaber controllers will likely ignore the issued commands.

    Args:
        lickport_port: The USB port used by the LickPort Zaber motor controllers (devices).
        zaber_positions_path: The path to the zaber_positions.yaml file that stores the motor positions saved during
            previous runtime.

    Attributes:
        _lickport: Stores the Connection class instance that manages the USB connection to a daisy-chain of Zaber
            devices (controllers) that allow repositioning the lick tube.
        _lickport_z: Stores the Axis (motor) class that controls the position of the lickport along the Z axis.
        _lickport_x: Stores the Axis (motor) class that controls the position of the lickport along the X axis.
        _lickport_y: Stores the Axis (motor) class that controls the position of the lickport along the Y axis.
        _previous_positions: An instance of _ZaberPositions class that stores the positions of LickPort motors during a
           previous runtime. If this data is not available, this attribute is set to None to indicate there are no
           previous positions to use.
    """

    def __init__(self, lickport_port: str, zaber_positions_path: Path) -> None:
        # Lickport controller (zaber). This is an assembly of 3 zaber controllers (devices) that allow moving the
        # lick tube in Z, X, and Y dimensions. Note, this assumes that the chaining order of individual zaber devices is
        # fixed and is always Z-X-Y.
        self._lickport: ZaberConnection = ZaberConnection(port=lickport_port)
        self._lickport.connect()
        self._lickport_z: ZaberAxis = self._lickport.get_device(0).axis
        self._lickport_x: ZaberAxis = self._lickport.get_device(1).axis
        self._lickport_y: ZaberAxis = self._lickport.get_device(2).axis

        # If the previous positions path points to an existing .yaml file, loads the data from the file into
        # _ZaberPositions instance. Otherwise, sets the previous_positions attribute to None to indicate there are no
        # previous positions.
        self._previous_positions: None | _ZaberPositions = None
        if zaber_positions_path.exists():
            self._previous_positions = _ZaberPositions.from_yaml(zaber_positions_path)

    def restore_position(self, wait_until_idle: bool = True) -> None:
        """Restores the LickPort motor positions to the states recorded at the end of the previous runtime.

        For most runtimes, this method is used to restore the LickPort to the state used during a previous experiment or
        training session for each animal. Since all animals are slightly different, the optimal LickPort positions will
        vary slightly for each animal.

        Notes:
            If previous positions are not available, the method falls back to moving the LickPort motors to the general
            'parking' positions saved in the non-volatile memory of each motor controller. Note, this is in contrast to
            the HeadBar, which falls back to using the 'mounting' positions. The mounting position for the LickPort
            aligns it to the top left corner of the running wheel, to provide experimenter with easier access to the
            HeadBar. The parking position, on the other hand, positions the lick tube roughly next to the animal's
            head.

            When used together with the HeadBar class, this method should always be called after the similar method
            from the HeadBar class.

            This method moves all LickPort axes in-parallel to optimize runtime speed.

        Args:
            wait_until_idle: Determines whether to block in-place until all motors finish moving or to return without
                waiting for the motors to stop moving. This is primarily used to move multiple motor groups at the same
                time.
        """
        # If the positions are not available, warns the user and sets the motors to the 'generic' mount position.
        if self._previous_positions is None:
            message = (
                "No previous positions found when attempting to restore LickPort to the previous runtime state. "
                "Setting the LickPort motors to the default parking positions loaded from motor controller "
                "non-volatile memory."
            )
            console.echo(message=message, level=LogLevel.ERROR)
            self._lickport_z.move(amount=self._lickport_z.park_position, absolute=True, native=True)
            self._lickport_x.move(amount=self._lickport_x.park_position, absolute=True, native=True)
            self._lickport_y.move(amount=self._lickport_y.park_position, absolute=True, native=True)
        else:
            # Otherwise, restores Zaber positions.
            self._lickport_z.move(amount=self._previous_positions.lickport_z, absolute=True, native=True)
            self._lickport_x.move(amount=self._previous_positions.lickport_x, absolute=True, native=True)
            self._lickport_y.move(amount=self._previous_positions.lickport_y, absolute=True, native=True)

        # If requested, waits for the motors to finish moving before returning to caller. Otherwise, returns
        # without waiting for the motors to stop moving. The latter case is used to issue commands to multiple motor
        # groups at the same time.
        if wait_until_idle:
            self.wait_until_idle()

    def prepare_motors(self, wait_until_idle: bool = True) -> None:
        """Unparks and homes all LickPort motors.

        This method should be used at the beginning of each runtime (experiment, training, etc.) to ensure all LickPort
        motors can be moved (are not parked) and have a stable point of reference. The motors are left at their
        respective homing positions at the end of this method's runtime, and it is assumed that a different class
        method is called after this method to set the motors into the desired position.

        Notes:
            This method moves all LickPort axes in-parallel to optimize runtime speed.

        Args:
            wait_until_idle: Determines whether to block in-place until all motors finish moving or to return without
                waiting for the motors to stop moving. This is primarily used to move multiple motor groups at the same
                time.
        """

        # Unparks all motors.
        self._lickport_z.unpark()
        self._lickport_x.unpark()
        self._lickport_y.unpark()

        # Homes all motors in-parallel.
        self._lickport_z.home()
        self._lickport_x.home()
        self._lickport_y.home()

        # If requested, waits for the motors to finish moving before returning to caller. Otherwise, returns
        # without waiting for the motors to stop moving. The latter case is used to issue commands to multiple motor
        # groups at the same time.
        if wait_until_idle:
            self.wait_until_idle()

    def park_position(self, wait_until_idle: bool = True) -> None:
        """Moves all LickPort motors to their parking positions and parks (locks) them preventing future movements.

        This method should be used at the end of each runtime (experiment, training, etc.) to ensure all LickPort motors
        are positioned in a way that guarantees that they can be homed during the next runtime.

        Notes:
            The motors are moved to the parking positions stored in the non-volatile memory of each motor controller. If
            this class is used together with the HeadBar class, this method should always be called after the similar
            method from the HeadBar class.

            This method moves all LickPort axes in-parallel to optimize runtime speed.

        Args:
            wait_until_idle: Determines whether to block in-place until all motors finish moving or to return without
                waiting for the motors to stop moving. This is primarily used to move multiple motor groups at the same
                time.
        """

        # Moves all motors to their parking positions
        self._lickport_z.move(amount=self._lickport_z.park_position, absolute=True, native=True)
        self._lickport_x.move(amount=self._lickport_x.park_position, absolute=True, native=True)
        self._lickport_y.move(amount=self._lickport_y.park_position, absolute=True, native=True)

        # If requested, waits for the motors to finish moving before returning to caller. Otherwise, returns
        # without waiting for the motors to stop moving. The latter case is used to issue commands to multiple motor
        # groups at the same time.
        if wait_until_idle:
            self.wait_until_idle()

    def calibrate_position(self, wait_until_idle: bool = True) -> None:
        """Moves all LickPort motors to the water valve calibration position.

        This position is stored in the non-volatile memory of each motor controller. This position is used during the
        water valve calibration to provide experimenters with easier access to the LickPort tube.

        Notes:
            This method moves all LickPort axes in-parallel to optimize runtime speed.

        Args:
            wait_until_idle: Determines whether to block in-place until all motors finish moving or to return without
                waiting for the motors to stop moving. This is primarily used to move multiple motor groups at the same
                time.
        """
        # Moves all motors to their calibration positions
        self._lickport_z.move(amount=self._lickport_z.valve_position, absolute=True, native=True)
        self._lickport_x.move(amount=self._lickport_x.valve_position, absolute=True, native=True)
        self._lickport_y.move(amount=self._lickport_y.valve_position, absolute=True, native=True)

        # If requested, waits for the motors to finish moving before returning to caller. Otherwise, returns
        # without waiting for the motors to stop moving. The latter case is used to issue commands to multiple motor
        # groups at the same time.
        if wait_until_idle:
            self.wait_until_idle()

    def mount_position(self, wait_until_idle: bool = True) -> None:
        """Moves all LickPort motors to the animal mounting position.

        This position is stored in the non-volatile memory of each motor controller. This position is used when the
        animal is mounted into the VR rig to provide the experimenter with easy access to the HeadBar holder.

        Notes:
            This method moves all LickPort axes in-parallel to optimize runtime speed.

        Args:
            wait_until_idle: Determines whether to block in-place until all motors finish moving or to return without
                waiting for the motors to stop moving. This is primarily used to move multiple motor groups at the same
                time.
        """
        # Moves all motors to their mounting positions
        self._lickport_z.move(amount=self._lickport_z.mount_position, absolute=True, native=True)
        self._lickport_x.move(amount=self._lickport_x.mount_position, absolute=True, native=True)
        self._lickport_y.move(amount=self._lickport_y.mount_position, absolute=True, native=True)

        # If requested, waits for the motors to finish moving before returning to caller. Otherwise, returns
        # without waiting for the motors to stop moving. The latter case is used to issue commands to multiple motor
        # groups at the same time.
        if wait_until_idle:
            self.wait_until_idle()

    def get_positions(self) -> tuple[int, int, int]:
        """Returns the current position of all LickPort motors in native motor units.

        The positions are returned in the order of : Z, X, and Y. These positions can be saves as a zaber_positions.yaml
        file to be used during the following runtimes.
        """
        return (
            int(self._lickport_z.get_position(native=True)),
            int(self._lickport_x.get_position(native=True)),
            int(self._lickport_y.get_position(native=True)),
        )

    def wait_until_idle(self) -> None:
        """This method blocks in-place while at least one motor in the managed motor group is moving.

        Primarily, this method is used to issue commands to multiple motor groups and then block until all motors in
        all groups finish moving. This optimizes the overall time taken to move the motors.
        """
        # Waits for the motors to finish moving.
        while self._lickport_z.is_busy or self._lickport_x.is_busy or self._lickport_y.is_busy:
            pass

    def disconnect(self) -> None:
        """Disconnects from the access port of the motor group.

        This method should be called after the motors are parked (moved to their final parking position) to release
        the connection resources. If this method is not called, the runtime will NOT be able to terminate.

        Notes:
            Calling this method will execute the motor parking sequence, which involves moving the motors to their
            parking position. Make sure there are no animals mounted on the rig and that the mesoscope objective is
            removed from the rig before executing this command.
        """
        message = f"LickPort motor connection: Terminated"
        console.echo(message, LogLevel.SUCCESS)
        self._lickport.disconnect()


class _MicroControllerInterfaces:
    """Interfaces with all Ataraxis Micro Controller (AMC) devices that control and record non-video behavioral data
    from the Mesoscope-VR system.

    This class interfaces with the three AMC controllers used during various runtimes: Actor, Sensor, and Encoder. The
    class exposes methods to send commands to the hardware modules managed by these microcontrollers. In turn, these
    modules control specific components of the Mesoscope-Vr system, such as rotary encoders, solenoid valves, and
    conductive lick sensors.

    Notes:
        This class is primarily intended to be used internally by the MesoscopeExperiment and BehavioralTraining
        classes. Our valve calibration CLI uses this class directly to calibrate the water valve, but this is a unique
        use scenario. Do not initialize this class directly unless you know what you are doing.

        This class is calibrated and statically configured for the Mesoscope-VR system used in the Sun lab. Source code
        refactoring will likely be necessary to adapt the class to other runtime conditions.

        Calling the initializer does not start the underlying processes. Use the start() method before issuing other
        commands to properly initialize all remote processes. This design is intentional and is used during experiment
        and training runtimes to parallelize data preprocessing and starting the next animal's session.

    Args:
        data_logger: The initialized DataLogger instance used to log the data generated by the managed microcontrollers.
            For most runtimes, this argument is resolved by the MesoscopeExperiment or BehavioralTraining classes that
            initialize this class.
        screens_on: Determines whether the VR screens are ON when this class is initialized. Since there is no way of
            getting this information via hardware, the initial screen state has to be supplied as an argument. The class
            will manage and track the state after initialization.
        actor_port: The USB port used by the Actor Microcontroller.
        sensor_port: The USB port used by the Sensor Microcontroller.
        encoder_port: The USB port used by the Encoder Microcontroller.
        valve_calibration_data: A tuple of tuples, with each inner tuple storing a pair of values. The first value is
            the duration, in microseconds, the valve was open. The second value is the volume of dispensed water, in
            microliters. This data is used by the ValveInterface to calculate pulse times necessary to deliver requested
            volumes of water.
        debug: Determines whether to run the managed interfaces in debug mode. Generally, this mode should be disabled
            for most runtimes. It is used during the initial system calibration to interactively debug and adjust the
            hardware module and interface configurations.

    Attributes:
        _started: Tracks whether the VR system and experiment runtime are currently running.
        _previous_volume: Tracks the volume of water dispensed during previous deliver_reward() calls.
        _screen_state: Tracks the current VR screen state.
        _mesoscope_start: The interface that starts mesoscope frame acquisition via TTL pulse.
        _mesoscope_stop: The interface that stops mesoscope frame acquisition via TTL pulse.
        _break: The interface that controls the electromagnetic break attached to the running wheel.
        _reward: The interface that controls the solenoid water valve that delivers water to the animal.
        _screens: The interface that controls the power state of the VR display screens.
        _actor: The main interface for the 'Actor' Ataraxis Micro Controller (AMC) device.
        _mesoscope_frame: The interface that monitors frame acquisition timestamp signals sent by the mesoscope.
        _lick: The interface that monitors animal's interactions with the lick sensor (detects licks).
        _torque: The interface that monitors the torque applied by the animal to the running wheel.
        _sensor: The main interface for the 'Sensor' Ataraxis Micro Controller (AMC) device.
        _wheel_encoder: The interface that monitors the rotation of the running wheel and converts it into the distance
            traveled by the animal.
        _encoder: The main interface for the 'Encoder' Ataraxis Micro Controller (AMC) device.

    Raises:
        TypeError: If the provided valve_calibration_data argument is not a tuple or does not contain valid elements.
    """

    def __init__(
        self,
        data_logger: DataLogger,
        screens_on: bool = False,
        actor_port: str = "/dev/ttyACM0",
        sensor_port: str = "/dev/ttyACM1",
        encoder_port: str = "/dev/ttyACM2",
        valve_calibration_data: tuple[tuple[int | float, int | float], ...] = (
            (15000, 1.8556),
            (30000, 3.4844),
            (45000, 7.1846),
            (60000, 10.0854),
        ),
        debug: bool = False,
    ) -> None:
        # Initializes the start state tracker first
        self._started: bool = False

        self._previous_volume: float = 0.0
        self._screen_state: bool = screens_on  # Tracks the current screen state

        # Verifies water valve calibration data.
        if not isinstance(valve_calibration_data, tuple) or not all(
            isinstance(item, tuple)
            and len(item) == 2
            and isinstance(item[0], (int, float))
            and isinstance(item[1], (int, float))
            for item in valve_calibration_data
        ):
            message = (
                f"Unable to initialize the MicroControllerInterfaces class. Expected a tuple of 2-element tuples with "
                f"integer or float values for 'valve_calibration_data' argument, but instead encountered "
                f"{valve_calibration_data} of type {type(valve_calibration_data).__name__} with at least one "
                f"incompatible element."
            )
            console.error(message=message, error=TypeError)

        # ACTOR. Actor AMC controls the hardware that needs to be triggered by PC at irregular intervals. Most of such
        # hardware is designed to produce some form of an output: deliver water reward, engage wheel breaks, issue a
        # TTL trigger, etc.

        # Module interfaces:
        self._mesoscope_start: TTLInterface = TTLInterface(module_id=np.uint8(1), debug=debug)
        self._mesoscope_stop: TTLInterface = TTLInterface(module_id=np.uint8(2), debug=debug)
        self._break = BreakInterface(
            minimum_break_strength=43.2047,  # 0.6 in oz
            maximum_break_strength=1152.1246,  # 16 in oz
            object_diameter=15.0333,  # 15 cm diameter + 0.0333 to account for the wrap
            debug=debug,
        )
        self._reward = ValveInterface(valve_calibration_data=valve_calibration_data, debug=debug)
        self._screens = ScreenInterface(initially_on=screens_on, debug=debug)

        # Main interface:
        self._actor: MicroControllerInterface = MicroControllerInterface(
            controller_id=np.uint8(101),
            microcontroller_serial_buffer_size=8192,
            microcontroller_usb_port=actor_port,
            data_logger=data_logger,
            module_interfaces=(self._mesoscope_start, self._mesoscope_stop, self._break, self._reward, self._screens),
        )

        # SENSOR. Sensor AMC controls the hardware that collects data at regular intervals. This includes lick sensors,
        # torque sensors, and input TTL recorders. Critically, all managed hardware does not rely on hardware interrupt
        # logic to maintain the necessary precision.

        # Module interfaces:
        # Mesoscope frame timestamp recorder. THe class is configured to report detected pulses during runtime to
        # support checking whether mesoscope start trigger correctly starts the frame acquisition process.
        self._mesoscope_frame: TTLInterface = TTLInterface(module_id=np.uint8(1), debug=debug)
        self._lick: LickInterface = LickInterface(lick_threshold=650, debug=debug)  # Lick sensor
        self._torque: TorqueInterface = TorqueInterface(
            baseline_voltage=2046,  # ~1.65 V
            maximum_voltage=2750,  # This was determined experimentally and matches the torque that overcomes break
            sensor_capacity=720.0779,  # 10 in oz
            object_diameter=15.0333,  # 15 cm diameter + 0.0333 to account for the wrap
            debug=debug,
        )

        # Main interface:
        self._sensor: MicroControllerInterface = MicroControllerInterface(
            controller_id=np.uint8(152),
            microcontroller_serial_buffer_size=8192,
            microcontroller_usb_port=sensor_port,
            data_logger=data_logger,
            module_interfaces=(self._mesoscope_frame, self._lick, self._torque),
        )

        # ENCODER. Encoder AMC is specifically designed to interface with a rotary encoder connected to the running
        # wheel. The encoder uses hardware interrupt logic to maintain high precision and, therefore, it is isolated
        # to a separate microcontroller to ensure adequate throughput.

        # Module interfaces:
        self._wheel_encoder: EncoderInterface = EncoderInterface(
            encoder_ppr=8192, object_diameter=15.0333, cm_per_unity_unit=10.0, debug=debug
        )

        # Main interface:
        self._encoder: MicroControllerInterface = MicroControllerInterface(
            controller_id=np.uint8(203),
            microcontroller_serial_buffer_size=8192,
            microcontroller_usb_port=encoder_port,
            data_logger=data_logger,
            module_interfaces=(self._wheel_encoder,),
        )

    def start(self) -> None:
        """Starts MicroController communication processes and configures all hardware modules to use predetermined
        runtime parameters.

        This method sets up the necessary assets that enable MicroController-PC communication. Until this method is
        called, all other class methods will not function correctly.

        Notes:
            After calling this method, most hardware modules will be initialized to an idle state. The only exception to
            this rule is the wheel break, which initializes to the 'engaged' state. Use other class methods to
            switch individual hardware modules into the desired state.

            Since most modules initialize to an idle state, they will not be generating data. Therefore, it is safe
            to call this method before enabling the DataLogger class. However, it is strongly advised to enable the
            DataLogger as soon as possible to avoid data piling up in the buffer.

            This method uses Console to notify the user about the initialization progress, but it does not enable the
            Console class itself. Make sure the console is enabled before calling this method.
        """

        # Prevents executing this method if the MicroControllers are already running.
        if self._started:
            return

        message = "Initializing Ataraxis Micro Controller (AMC) Interfaces..."
        console.echo(message=message, level=LogLevel.INFO)

        # Starts all microcontroller interfaces
        self._actor.start()
        self._actor.unlock_controller()  # Only Actor outputs data, so no need to unlock other controllers.
        self._sensor.start()
        self._encoder.start()

        # Configures the encoder to only report forward motion (CW) if the motion exceeds ~ 1 mm of distance.
        self._wheel_encoder.set_parameters(report_cw=False, report_ccw=True, delta_threshold=15)

        # Configures mesoscope start and stop triggers to use 10 ms pulses
        self._mesoscope_start.set_parameters(pulse_duration=np.uint32(10000))
        self._mesoscope_stop.set_parameters(pulse_duration=np.uint32(10000))

        # Configures screen trigger to use 500 ms pulses
        self._screens.set_parameters(pulse_duration=np.uint32(500000))

        # Configures the water valve to deliver ~ 5 uL of water. Also configures the valve calibration method to run the
        # 'reference' calibration for 5 uL rewards used to verify the valve calibration before every experiment.
        self._reward.set_parameters(
            pulse_duration=np.uint32(35590), calibration_delay=np.uint32(200000), calibration_count=np.uint16(200)
        )

        # Configures the lick sensor to filter out dry touches and only report significant changes in detected voltage
        # (used as a proxy for detecting licks).
        self._lick.set_parameters(
            signal_threshold=np.uint16(400), delta_threshold=np.uint16(400), averaging_pool_size=np.uint8(10)
        )

        # Configures the torque sensor to filter out noise and sub-threshold 'slack' torque signals.
        self._torque.set_parameters(
            report_ccw=True,
            report_cw=True,
            signal_threshold=np.uint16(100),
            delta_threshold=np.uint16(70),
            averaging_pool_size=np.uint8(5),
        )

        # The setup procedure is complete.
        self._started = True

        message = "Ataraxis Micro Controller (AMC) Interfaces: Initialized."
        console.echo(message=message, level=LogLevel.SUCCESS)

    def stop(self) -> None:
        """Stops all MicroController communication processes and releases all resources.

        This method needs to be called at the end of each runtime to release the resources reserved by the start()
        method. Until the stop() method is called, the DataLogger instance may receive data from running
        MicroControllers, so calling this method also guarantees no MicroController data will be lost if the DataLogger
        process is terminated.
        """

        # Prevents stopping an already stopped VR process.
        if not self._started:
            return

        message = "Terminating Ataraxis Micro Controller (AMC) Interfaces..."
        console.echo(message=message, level=LogLevel.INFO)

        # Resets the _started tracker
        self._started = False

        # Stops all microcontroller interfaces. This directly shuts down and resets all managed hardware modules.
        self._actor.stop()
        self._sensor.stop()
        self._encoder.stop()

        message = "Ataraxis Micro Controller (AMC) Interfaces: Terminated."
        console.echo(message=message, level=LogLevel.SUCCESS)

    def enable_encoder_monitoring(self) -> None:
        """Enables wheel encoder monitoring at 2 kHz rate.

        This means that, at most, the Encoder will send the data to the PC at the 2 kHz rate. The Encoder collects data
        at the native rate supported by the microcontroller hardware, which likely exceeds the reporting rate.
        """
        self._wheel_encoder.reset_pulse_count()
        self._wheel_encoder.check_state(repetition_delay=np.uint32(500))

    def disable_encoder_monitoring(self) -> None:
        """Stops monitoring the wheel encoder."""
        self._wheel_encoder.reset_command_queue()

    def start_mesoscope(self) -> None:
        """Sends the acquisition start TTL pulse to the mesoscope."""
        self._mesoscope_start.send_pulse()

    def stop_mesoscope(self) -> None:
        """Sends the acquisition stop TTL pulse to the mesoscope."""
        self._mesoscope_stop.send_pulse()

    def enable_break(self) -> None:
        """Engages the wheel break at maximum strength, preventing the animal from running on the wheel."""
        self._break.toggle(state=True)

    def disable_break(self) -> None:
        """Disengages the wheel break, enabling the animal to run on the wheel."""
        self._break.toggle(state=False)

    def enable_vr_screens(self) -> None:
        """Sets the VR screens to be ON."""
        if not self._screen_state:  # If screens are OFF
            self._screens.toggle()  # Sets them ON
            self._screen_state = True

    def disable_vr_screens(self) -> None:
        """Sets the VR screens to be OFF."""
        if self._screen_state:  # If screens are ON
            self._screens.toggle()  # Sets them OFF
            self._screen_state = False

    def enable_mesoscope_frame_monitoring(self) -> None:
        """Enables monitoring the TTL pulses sent by the mesoscope to communicate when it is scanning a frame at
        ~ 1 kHZ rate.

        The mesoscope sends the HIGH phase of the TTL pulse while it is scanning the frame, which produces a pulse of
        ~100ms. This is followed by ~5ms LOW phase during which the Galvos are executing the flyback procedure. This
        command checks the state of the TTL pin at the 1 kHZ rate, which is enough to accurately report both phases.
        """
        self._mesoscope_frame.check_state(repetition_delay=np.uint32(1000))

    def disable_mesoscope_frame_monitoring(self) -> None:
        """Stops monitoring the TTL pulses sent by the mesoscope to communicate when it is scanning a frame."""
        self._mesoscope_frame.reset_command_queue()

    def enable_lick_monitoring(self) -> None:
        """Enables monitoring the state of the conductive lick sensor at ~ 1 kHZ rate.

        The lick sensor measures the voltage across the lick sensor and reports surges in voltage to the PC as a
        reliable proxy for tongue-to-sensor contact. Most lick events span at least 100 ms of time and, therefore, the
        rate of 1 kHZ is adequate for resolving all expected single-lick events.
        """
        self._lick.check_state(repetition_delay=np.uint32(1000))

    def disable_lick_monitoring(self) -> None:
        """Stops monitoring the conductive lick sensor."""
        self._lick.reset_command_queue()

    def enable_torque_monitoring(self) -> None:
        """Enables monitoring the torque sensor at ~ 1 kHZ rate.

        The torque sensor detects CW and CCW torques applied by the animal to the wheel. Currently, we do not have a
        way of reliably calibrating the sensor, so detected torque magnitudes are only approximate. However, the sensor
        reliably distinguishes large torques from small torques and accurately tracks animal motion activity when the
        wheel break is engaged.
        """
        self._torque.check_state(repetition_delay=np.uint32(1000))

    def disable_torque_monitoring(self) -> None:
        """Stops monitoring the torque sensor."""
        self._torque.reset_command_queue()

    def open_valve(self) -> None:
        """Opens the water reward solenoid valve.

        This method is primarily used to prime the water line with water before the first experiment or training session
        of the day.
        """
        self._reward.toggle(state=True)

    def close_valve(self) -> None:
        """Closes the water reward solenoid valve."""
        self._reward.toggle(state=False)

    def deliver_reward(self, volume: float = 5.0) -> None:
        """Pulses the water reward solenoid valve for the duration of time necessary to deliver the provided volume of
        water.

        This method assumes that the valve has been calibrated before calling this method. It uses the calibration data
        provided at class instantiation to determine the period of time the valve should be kept open to deliver the
        requested volume of water.

        Args:
            volume: The volume of water to deliver, in microliters.
        """

        # This ensures that the valve settings are only updated if the new volume does not match the previous volume.
        # This minimizes unnecessary updates to the valve settings.
        if volume != self._previous_volume:
            # Note, calibration parameters are not used by the command below, but we explicitly set them here for
            # consistency
            self._reward.set_parameters(
                pulse_duration=self._reward.get_duration_from_volume(volume),
                calibration_delay=np.uint32(200000),
                calibration_count=np.uint16(200),
            )
        self._reward.send_pulse()

    def reference_valve(self) -> None:
        """Runs the reference valve calibration procedure.

        Reference calibration is functionally similar to the calibrate_valve() method runtime. It is, however, optimized
        to deliver the overall volume of water recognizable for the human eye looking at the syringe holding the water
        (water 'tank' used in our system). Additionally, this uses the 5 uL volume as the reference volume, which
        matches the volume we use during experiments and training sessions.

        The reference calibration HAS to be run with the water line being primed, deaerated, and the holding ('tank')
        syringe filled exactly to the 5 mL mark. This procedure is designed to dispense 5 uL of water 200 times, which
        should overall dispense ~ 1 ml of water.

        Notes:
            Use one of the conical tubes stored next to the Mesoscope cage to collect the dispensed water. It is highly
            encouraged to use both the visual confirmation (looking at the syringe water level drop) and the weight
            confirmation (weighing the water dispensed into the collection tube). This provides the most accurate
            referencing result.

            If the referencing procedure fails to deliver 5 +- 0.5 uL of water measured with either method, the valve
            needs to be recalibrated using the calibrate_valve() method. Also, if valve referencing result stability
            over multiple days fluctuates significantly, it is advised to recalibrate the valve using the
            calibrate_valve() method.
        """
        self._reward.set_parameters(
            pulse_duration=np.uint32(self._reward.get_duration_from_volume(target_volume=5.0)),
            calibration_delay=np.uint32(200000),
            calibration_count=np.uint16(200),
        )  # 5 ul x 200 times
        self._reward.calibrate()

    def calibrate_valve(self, pulse_duration: int = 15) -> None:
        """Cycles solenoid valve opening and closing 500 times to determine the amount of water dispensed by the input
        pulse_duration.

        The valve is kept open for the specified number of milliseconds. Between pulses, the valve is kept closed for
        200 ms. Due to our valve design, keeping the valve closed for less than 200 ms generates a large pressure
        at the third (Normally Open) port, which puts unnecessary strain on the port plug.

        During runtime, the valve will be pulsed 500 times to provide a large sample size. During calibration, the water
        should be collected in a pre-weighted conical tube. After the calibration is over, the tube with dispensed water
        has to be weighted to determine the dispensed volume by weight.

        Notes:
            The calibration should be run with the following durations: 15 ms, 30 ms, 45 ms, and 60 ms. During testing,
            we found that these values roughly cover the range from 2 uL to 10 uL, which is enough to cover most
            training and experiment runtimes.

            Make sure that the water line is primed, deaerated, and the holding ('tank') syringe filled exactly to the
            5 mL mark at the beginning of each calibration cycle. Depending on the calibrated pulse_duration, you may
            need to refill the syringe during the calibration runtime. The calibration durations mentioned above should
            not need manual tank refills.

        Args:
            pulse_duration: The duration, in milliseconds, the valve is kept open at each calibration cycle
        """
        pulse_us = pulse_duration * 1000  # Convert milliseconds to microseconds
        self._reward.set_parameters(
            pulse_duration=np.uint32(pulse_us), calibration_delay=np.uint32(200000), calibration_count=np.uint16(500)
        )
        self._reward.calibrate()

    def get_mesoscope_frame_data(self) -> NDArray[np.uint64]:
        """Returns the array that stores the data extracted from the compressed Mesoscope Frame TTLModule log file.

        The array stores the timestamps for each frame acquired by the mesoscope. The timestamps mark the beginning of
        each frame scanning cycle and are returned as the number of microseconds since the UTC epoch onset.

        Notes:
            Do not call this method before the DataLogger has compressed all logged data.
        """
        return self._mesoscope_frame.parse_logged_data()

    def get_encoder_data(self) -> tuple[NDArray[np.uint64], NDArray[np.float64]]:
        """Returns two arrays that jointly store the data extracted from the compressed EncoderModule log file.

        The first array stores the timestamps as the number of microseconds since the UTC epoch onset. The second array
        stores the cumulative distance, in centimeters, the animal has traveled since the onset of the session, at each
        timestamp.

        Notes:
            Do not call this method before the DataLogger has compressed all logged data.
        """
        return self._wheel_encoder.parse_logged_data()

    def get_torque_data(self) -> tuple[NDArray[np.uint64], NDArray[np.float64]]:
        """Returns two arrays that jointly store the data extracted from the compressed TorqueModule log file.

        The first array stores the timestamps as the number of microseconds since the UTC epoch onset. The second array
        stores the torque value, in Newton centimeters, applied by the animal to the wheel at each timestamp. Positive
        values denote CCW torques and negative values denote CW torques.

        Notes:
            Do not call this method before the DataLogger has compressed all logged data.
        """
        return self._torque.parse_logged_data()

    def get_lick_data(self) -> tuple[NDArray[np.uint64], NDArray[np.uint8]]:
        """Returns two arrays that jointly store the data extracted from the compressed LickModule log file.

        The first array stores the timestamps as the number of microseconds since the UTC epoch onset. The second array
        stores the binary lick detection state (1 for tongue touching the sensor, 0 for tongue not touching the sensor)
        at each timestamp.

        Notes:
            Do not call this method before the DataLogger has compressed all logged data.
        """
        return self._lick.parse_logged_data()

    def get_screen_data(self) -> tuple[NDArray[np.uint64], NDArray[np.uint8]]:
        """Returns two arrays that jointly store the data extracted from the compressed ScreenModule log file.

        The first array stores the timestamps as the number of microseconds since the UTC epoch onset. The second array
        stores the binary screen state (1 for screens being enabled, 0 for screens being disabled) at each timestamp.

        Notes:
            Do not call this method before the DataLogger has compressed all logged data.
        """
        return self._screens.parse_logged_data()

    def get_break_data(self) -> tuple[NDArray[np.uint64], NDArray[np.float64]]:
        """Returns two arrays that jointly store the data extracted from the compressed BreakModule log file.

        The first array stores the timestamps as the number of microseconds since the UTC epoch onset. The second array
        stores the torque applied by the break to the wheel, in Newton centimeters, at each timestamp.

        Notes:
            Do not call this method before the DataLogger has compressed all logged data.
        """
        return self._break.parse_logged_data()

    def get_valve_data(self) -> tuple[NDArray[np.uint64], NDArray[np.float64]]:
        """Returns two arrays that jointly store the data extracted from the compressed ValveModule log file.

        The first array stores the timestamps as the number of microseconds since the UTC epoch onset. The second array
        stores the cumulative volume of water, in microliters, delivered at each timestamp relative to experiment onset.

        Notes:
            Do not call this method before the DataLogger has compressed all logged data.
        """
        return self._reward.parse_logged_data()

    @property
    def frame_acquisition_status(self) -> bool:
        """Returns true if the mesoscope is currently scanning (acquiring) a frame."""
        return self._mesoscope_frame.pulse_status

    @property
    def total_delivered_volume(self) -> float:
        """Returns the total volume of water, in microliters, dispensed by the valve since runtime onset."""
        return self._reward.delivered_volume

    @property
    def speed_tracker(self) -> SharedMemoryArray:
        """Returns the SharedMemoryArray used to communicate the average speed of the animal during runtime.

        This array should be passed to a Visualizer class so that it can sample the shared data to generate real-time
        running speed plots. It is also used by the run training logic to evaluate animal's performance during training.
        """
        return self._wheel_encoder.speed_tracker

    @property
    def lick_tracker(self) -> SharedMemoryArray:
        """Returns the SharedMemoryArray used to communicate the lick sensor status.

        This array should be passed to a Visualizer class so that it can sample the shared data to generate real-time
        lick detection plots.
        """
        return self._lick.lick_tracker

    @property
    def valve_tracker(self) -> SharedMemoryArray:
        """Returns the SharedMemoryArray used to communicate the water reward valve state.

        This array should be passed to a Visualizer class so that it can sample the shared data to generate real-time
        reward delivery plots.
        """
        return self._reward.reward_tracker


class _VideoSystems:
    """Interfaces with all cameras managed by Ataraxis Video System (AVS) classes that acquire and save camera frames
    as .mp4 video files.

    This class interfaces with the three AVS cameras used during various runtimes to record animal behavior: the face
    camera and the body cameras (the left camera and the right camera). The face camera is a high-grade scientific
    camera that records the animal's face and pupil. The left and right cameras are lower-end security cameras recording
    the animal's body from the left and right sides.

    Notes:
        This class is primarily intended to be used internally by the MesoscopeExperiment and BehavioralTraining
        classes. Do not initialize this class directly unless you know what you are doing.

        This class is calibrated and statically configured for the Mesoscope-VR system used in the Sun lab. Source code
        refactoring will likely be necessary to adapt the class to other runtime conditions.

        Calling the initializer does not start the underlying processes. Call the appropriate start() method to start
        acquiring and displaying face and body camera frames (there is a separate method for these two groups). Call
        the appropriate save() method to start saving the acquired frames to video files. Note that there is a single
        'global' stop() method that works for all cameras at the same time.

        The class is designed to be 'lock-in'. Once a camera is enabled, the only way to disable frame acquisition is to
        call the main stop() method. Similarly, once frame saving is started, there is no way to disable it without
        stopping the whole class. This is an intentional design decision optimized to the specific class use-pattern in
        our lab.

    Args:
        data_logger: The initialized DataLogger instance used to log the data generated by the managed cameras. For most
            runtimes, this argument is resolved by the MesoscopeExperiment or BehavioralTraining classes that
            initialize this class.
        output_directory: The path to the directory where to output the generated .mp4 video files. Each managed camera
            generates a separate video file saved in the provided directory. For most runtimes, this argument is
            resolved by the MesoscopeExperiment or BehavioralTraining classes that initialize this class.
        face_camera_index: The index of the face camera in the list of all available Harvester-managed cameras.
        left_camera_index: The index of the left camera in the list of all available OpenCV-managed cameras.
        right_camera_index: The index of the right camera in the list of all available OpenCV-managed cameras.
        harvesters_cti_path: The path to the GeniCam CTI file used to connect to Harvesters-managed cameras.

    Attributes:
        _face_camera_started: Tracks whether the face camera frame acquisition is running.
        _body_cameras_started: Tracks whether the body cameras frame acquisition is running.
        _face-camera: The interface that captures and saves the frames acquired by the 9MP scientific camera aimed at
            the animal's face and eye from the left side (via a hot mirror).
        _left_camera: The interface that captures and saves the frames acquired by the 1080P security camera aimed on
            the left side of the animal and the right and center VR screens.
        _right_camera: The interface that captures and saves the frames acquired by the 1080P security camera aimed on
            the right side of the animal and the left VR screen.
    """

    def __init__(
        self,
        data_logger: DataLogger,
        output_directory: Path,
        face_camera_index: int = 0,
        left_camera_index: int = 0,
        right_camera_index: int = 2,
        harvesters_cti_path: Path = Path("/opt/mvIMPACT_Acquire/lib/x86_64/mvGenTLProducer.cti"),
    ) -> None:
        # Creates the _started flags first to avoid leaks if the initialization method fails.
        self._face_camera_started: bool = False
        self._body_cameras_started: bool = False

        # FACE CAMERA. This is the high-grade scientific camera aimed at the animal's face using the hot-mirror. It is
        # a 10-gigabit 9MP camera with a red long-pass filter and has to be interfaced through the GeniCam API. Since
        # the VRPC has a 4090 with 2 hardware acceleration chips, we are using the GPU to save all of our frame data.
        self._face_camera: VideoSystem = VideoSystem(
            system_id=np.uint8(51),
            data_logger=data_logger,
            output_directory=output_directory,
            harvesters_cti_path=harvesters_cti_path,
        )
        # The acquisition parameters (framerate, frame dimensions, crop offsets, etc.) are set via the SVCapture64
        # software and written to non-volatile device memory. Generally, all projects in the lab should be using the
        # same parameters.
        self._face_camera.add_camera(
            save_frames=True,
            camera_index=face_camera_index,
            camera_backend=CameraBackends.HARVESTERS,
            output_frames=False,
            display_frames=True,
            display_frame_rate=25,
        )
        self._face_camera.add_video_saver(
            hardware_encoding=True,
            video_format=VideoFormats.MP4,
            video_codec=VideoCodecs.H265,
            preset=GPUEncoderPresets.MEDIUM,
            input_pixel_format=InputPixelFormats.MONOCHROME,
            output_pixel_format=OutputPixelFormats.YUV444,
            quantization_parameter=15,
        )

        # LEFT CAMERA. A 1080P security camera that is mounted on the left side from the mouse's perspective
        # (viewing the left side of the mouse and the right screen). This camera is interfaced with through the OpenCV
        # backend.
        self._left_camera: VideoSystem = VideoSystem(
            system_id=np.uint8(62), data_logger=data_logger, output_directory=output_directory
        )

        # DO NOT try to force the acquisition rate. If it is not 30 (default), the video will not save.
        self._left_camera.add_camera(
            save_frames=True,
            camera_index=left_camera_index,
            camera_backend=CameraBackends.OPENCV,
            output_frames=False,
            display_frames=True,
            display_frame_rate=25,
            color=False,
        )
        self._left_camera.add_video_saver(
            hardware_encoding=True,
            video_format=VideoFormats.MP4,
            video_codec=VideoCodecs.H265,
            preset=GPUEncoderPresets.FASTEST,
            input_pixel_format=InputPixelFormats.MONOCHROME,
            output_pixel_format=OutputPixelFormats.YUV420,
            quantization_parameter=30,
        )

        # RIGHT CAMERA. Same as the left camera, but mounted on the right side from the mouse's perspective.
        self._right_camera: VideoSystem = VideoSystem(
            system_id=np.uint8(73), data_logger=data_logger, output_directory=output_directory
        )
        # Same as above, DO NOT force acquisition rate
        self._right_camera.add_camera(
            save_frames=True,
            camera_index=right_camera_index,  # The only difference between left and right cameras.
            camera_backend=CameraBackends.OPENCV,
            output_frames=False,
            display_frames=True,
            display_frame_rate=25,
            color=False,
        )
        self._right_camera.add_video_saver(
            hardware_encoding=True,
            video_format=VideoFormats.MP4,
            video_codec=VideoCodecs.H265,
            preset=GPUEncoderPresets.FASTEST,
            input_pixel_format=InputPixelFormats.MONOCHROME,
            output_pixel_format=OutputPixelFormats.YUV420,
            quantization_parameter=30,
        )

    def start_face_camera(self) -> None:
        """Starts face camera frame acquisition.

        This method sets up both the frame acquisition (producer) process and the frame saver (consumer) process.
        However, the consumer process will not save any frames until save_face_camera_frames() method is called.
        """

        # Prevents executing this method if the face camera is already running
        if self._face_camera_started:
            return

        message = "Initializing face camera frame acquisition..."
        console.echo(message=message, level=LogLevel.INFO)

        # Starts frame acquisition. Note, this does NOT start frame saving.
        self._face_camera.start()
        self._face_camera_started = True

        message = "Face camera frame acquisition: Started."
        console.echo(message=message, level=LogLevel.SUCCESS)

    def start_body_cameras(self) -> None:
        """Starts left and right (body) camera frame acquisition.

        This method sets up both the frame acquisition (producer) process and the frame saver (consumer) process for
        both cameras. However, the consumer processes will not save any frames until save_body_camera_frames() method is
        called.
        """

        # Prevents executing this method if the body cameras are already running
        if self._body_cameras_started:
            return

        message = "Initializing body cameras (left and right) frame acquisition..."
        console.echo(message=message, level=LogLevel.INFO)

        # Starts frame acquisition. Note, this does NOT start frame saving.
        self._left_camera.start()
        self._right_camera.start()
        self._body_cameras_started = True

        message = "Body cameras frame acquisition: Started."
        console.echo(message=message, level=LogLevel.SUCCESS)

    def save_face_camera_frames(self) -> None:
        """Starts saving the frames acquired by the face camera as a video file."""

        # Starts frame saving process
        self._face_camera.start_frame_saving()

        message = "Face camera frame saving: Started."
        console.echo(message=message, level=LogLevel.SUCCESS)

    def save_body_camera_frames(self) -> None:
        """Starts saving the frames acquired by the left and right body cameras as a video file."""

        # Starts frame saving process
        self._left_camera.start_frame_saving()
        self._right_camera.start_frame_saving()

        message = "Body camera frame saving: Started."
        console.echo(message=message, level=LogLevel.SUCCESS)

    def stop(self) -> None:
        """Stops saving all camera frames and terminates the managed VideoSystems.

        This method needs to be called at the end of each runtime to release the resources reserved by the start()
        methods. Until the stop() method is called, the DataLogger instance may receive data from running
        VideoSystems, so calling this method also guarantees no VideoSystem data will be lost if the DataLogger
        process is terminated. Similarly, this guarantees the integrity of the generated video files.
        """

        # Prevents executing this method if no cameras are running.
        if not self._face_camera_started and not self._body_cameras_started:
            return

        message = "Terminating Ataraxis Video System (AVS) Interfaces..."
        console.echo(message=message, level=LogLevel.INFO)

        # Instructs all cameras to stop saving frames
        self._face_camera.stop_frame_saving()
        self._left_camera.stop_frame_saving()
        self._right_camera.stop_frame_saving()

        message = "Camera frame saving: Stopped."
        console.echo(message=message, level=LogLevel.SUCCESS)

        # Stops all cameras
        self._face_camera.stop()
        self._left_camera.stop()
        self._right_camera.stop()

        message = "Video Systems: Terminated."
        console.echo(message=message, level=LogLevel.SUCCESS)

    def process_log_data(self, output_file: Path) -> None:
        """Extracts the timestamps for each saved camera frame from the compressed log file of each VideoSystem and
        saves them as a .parquet file.

        Primarily, this method is used by the experimental runtime classes to generate the
        'camera_frame_timestamps.parquet' file that is later used by our video data processing pipeline. The output file
        is compressed with zstd at the maximum level (22) to optimize transmission over the network and long-term
        storage.

        Notes:
            In contrast to other binding classes, this class directly dumps the processed data into the output file.
            Since video data is processed the same way regardless of the runtime that produced it, this method can be
            used to execute the full preprocessing pipeline, including outputting the data. Other biding classes
            require runtime-specific processing, which is handled by the major runtime classes, so they instead return
            data stored in memory as NumPy arrays or Python variables.

        Args:
            output_file: The path to the output .parquet file. Usually, this is resolved by the SessionData class.
        """
        # Extracts the timestamps for the frames saved by each camera.
        face_stamps = self._face_camera.extract_logged_data()
        left_stamps = self._left_camera.extract_logged_data()
        right_stamps = self._right_camera.extract_logged_data()

        # Finds the maximum array length. This should be the length of the face_camera array, as it acquires data at
        # 60 fps vs. body cameras that use 30 fps.
        max_len = max(len(face_stamps), len(left_stamps), len(right_stamps))

        # Pads each array with zeros to match max length. Zero is not a valid timestamp for this data, as it points to
        # the UTC epoch onset which was in 1980. Therefore, seeing a zero timestamp automatically suggests it is not
        # valid.
        face_stamps = np.pad(face_stamps, (0, max_len - len(face_stamps)), constant_values=0)
        left_stamps = np.pad(left_stamps, (0, max_len - len(left_stamps)), constant_values=0)
        right_stamps = np.pad(right_stamps, (0, max_len - len(right_stamps)), constant_values=0)

        # Creates a Polars DataFrame with explicit uint64 type to store extracted timestamps
        df = pl.DataFrame(
            {
                "face_camera_timestamps_μs": pl.Series(face_stamps, dtype=pl.UInt64),
                "left_camera_timestamps_μs": pl.Series(left_stamps, dtype=pl.UInt64),
                "right_camera_timestamps_μs": pl.Series(right_stamps, dtype=pl.UInt64),
            }
        )

        # Saves as parquet with LZ4 compression
        df.write_parquet(file=output_file, compression="zstd", compression_level=22, use_pyarrow=True, statistics=True)


class KeyboardListener:
    """Monitors the keyboard input for various runtime control signals and changes internal flags to communicate
    detected signals.

    This class is used during all training runtimes to allow the user to manually control some aspects of the
    Mesoscope-VR system and runtime. For example, it is used to abort the training runtime early and manually deliver
    rewards via the lick-tube.

    This class looks for the following key combinations to set the following flags:
        - ESC + 'q': Immediately aborts the training runtime.
        - ESC + 'r': Delivers 5 uL of water via the LickTube.

    Notes:
        While our training logic functions automatically make use of this class, it is NOT explicitly part of the
        MesoscopeExperiment class runtime. We highly encourage incorporating this class into all experiment runtimes to
        provide similar APi as done by our training runtimes.

        This monitor may pick up keyboard strokes directed at other applications during runtime. While our unique key
        combination is likely to not be used elsewhere, exercise caution when using other applications alongside the
        runtime code.

    Attributes:
        _exit_flag: Tracks whether the instance has detected the runtime abort key sequence press.
        _reward_flag: Tracks whether the instance has detected the reward delivery key sequence press.
        _currently_pressed: Stores the keys that are currently being pressed.
        _listener: The Listener instance used to monitor keyboard strokes.

    """

    def __init__(self):
        self._exit_flag = False
        self._reward_flag = False
        self._currently_pressed = set()

        # Set up listeners for both press and release
        self._listener = keyboard.Listener(on_press=self._on_press, on_release=self._on_release)
        self._listener.daemon = True
        self._listener.start()

    def _on_press(self, key):
        """Adds newly pressed keys to the storage set and determines whether the pressed key combination matches the
        shutdown combination.

        This method is used as the 'on_press' callback for the Listener instance.
        """
        # Updates the set with current data
        self._currently_pressed.add(str(key))

        # Checks if ESC and 'q' are in the currently pressed set and, if so, flips the exit_flag.
        if "Key.esc" in self._currently_pressed and "'q'" in self._currently_pressed:
            self._exit_flag = True

        # Checks if ESC and 'r' are in the currently pressed set and, if so, flips the _reward_flag.
        if "Key.esc" in self._currently_pressed and "'r'" in self._currently_pressed:
            self._reward_flag = True

    def _on_release(self, key):
        """Removes no longer pressed keys from the storage set.

        This method is used as the 'on_release' callback for the Listener instance.
        """
        # Removes no longer pressed keys from the set
        key_str = str(key)
        if key_str in self._currently_pressed:
            self._currently_pressed.remove(key_str)

    @property
    def exit_signal(self):
        """Returns True if the listener has detected the runtime abort keys combination (ESC + q) being pressed.

        This indicates that the user has requested the runtime to gracefully abort.
        """
        return self._exit_flag

    @property
    def reward_signal(self):
        """Returns True if the listener has detected the water reward delivery keys combination (ESC + r) being
        pressed.

        This indicates that the user has requested the system to deliver 5uL water reward.

        Notes:
            Each time this property is accessed, the flag is reset to 0.
        """
        signal = copy.copy(self._reward_flag)
        self._reward_flag = False  # FLips the flag to False
        return signal


class SessionData:
    """Provides methods for managing the data acquired during one experiment or training session.

    This class functions as the central hub for collecting the data from all local PCs involved in the data acquisition
    process and pushing it to the NAS and the BioHPC server. Its primary purpose is to maintain the session data
    structure across all supported destinations and to efficiently and safely move the data to these destinations with
    minimal redundancy and footprint. Additionally, this class generates the paths used by all other classes from
    this library to determine where to load and saved various data during runtime.

    As part of its initialization, the class generates the session directory for the input animal and project
    combination. Session directories use the current UTC timestamp, down to microseconds, as the directory name. This
    ensures that each session name is unique and preserves the overall session order.

    Notes:
        Do not call methods from this class directly. This class is intended to be used through the MesoscopeExperiment
        and BehavioralTraining classes. The only reason the class is defined as public is to support reconfiguring major
        source / destination paths (NAS, BioHPC, etc.).

        It is expected that the server, nas, and mesoscope data directories are mounted on the host-machine via the
        SMB or equivalent protocol. All manipulations with these destinations are carried out with the assumption that
        the OS has full access to these directories and filesystems.

        This class is specifically designed for working with raw data from a single animal participating in a single
        experimental project session. Processed data is managed by the processing library methods and classes.

        This class generates an xxHash-128 checksum stored inside the ax_checksum.txt file at the root of each
        experimental session 'raw_data' directory. The checksum verifies the data of each file and the paths to each
        file relative to the 'raw_data' root directory.

    Args:
        project_name: The name of the project managed by the class.
        animal_name: The name of the animal managed by the class.
        generate_mesoscope_paths: Determines whether the managed session uses ScanImage (mesoscope) PC. Training
            sessions that do not use the Mesoscope do not need to resolve paths to mesoscope data folders and storage
            directories.
        local_root_directory: The path to the root directory where all projects are stored on the host-machine. Usually,
            this is the 'Experiments' folder on the 16TB volume of the VRPC machine.
        server_root_directory: The path to the root directory where all projects are stored on the BioHPC server
            machine. Usually, this is the 'storage/SunExperiments' (RAID 6) volume of the BioHPC server.
        nas_root_directory: The path to the root directory where all projects are stored on the Synology NAS. Usually,
            this is the non-compressed 'raw_data' directory on one of the slow (RAID 6) volumes, such as Volume 1.
        mesoscope_data_directory: The path to the directory where the mesoscope saves acquired frame data. Usually, this
            directory is on the fast Data volume (NVME) and is cleared of data after each acquisition runtime, such as
            the /mesodata/mesoscope_frames directory. Note, this class will delete and recreate the directory during its
            runtime, so it is highly advised to make sure it does not contain any important non-session-related data.

    Attributes:
        _local: The path to the host-machine directory for the managed project, animal and session combination.
            This path points to the 'raw_data' subdirectory that stores all acquired and preprocessed session data.
        _server: The path to the BioHPC server directory for the managed project, animal, and session
            combination.
        _nas: The path to the Synology NAS directory for the managed project, animal, and session combination.
        _mesoscope: The path to the root ScanImage PC (mesoscope) data directory. This directory is shared by
            all projects, animals, and sessions.
        _persistent: The path to the host-machine directory used to retain persistent data from previous session(s) of
            the managed project and animal combination. For example, this directory is used to persist Zaber positions
            and mesoscope motion estimator files when the original session directory is moved to nas and server.
        _project_name: Stores the name of the project whose data is managed by the class.
        _animal_name: Stores the name of the animal whose data is managed by the class.
        _session_name: Stores the name of the session directory whose data is managed by the class.
    """

    def __init__(
        self,
        project_name: str,
        animal_name: str,
        generate_mesoscope_paths: bool = True,
        local_root_directory: Path = Path("/media/Data/Experiments"),
        server_root_directory: Path = Path("/media/cbsuwsun/storage/sun_data"),
        nas_root_directory: Path = Path("/home/cybermouse/nas/rawdata"),
        mesoscope_data_directory: Path = Path("/home/cybermouse/scanimage/mesodata"),
    ) -> None:
        # Computes the project + animal directory paths for the local machine (VRPC)
        self._local: Path = local_root_directory.joinpath(project_name, animal_name)

        # Mesoscope is configured to use the same directories for all projects and animals
        self._mesoscope: Path = mesoscope_data_directory

        # Generates a separate directory to store persistent data. This has to be done early as _local is later
        # overwritten with the path to the raw_data directory of the created session.
        self._persistent: Path = self._local.joinpath("persistent_data")

        # Records animal and project names to attributes. Session name is resolved below
        self._project_name: str = project_name
        self._animal_name: str = animal_name

        # Acquires the UTC timestamp to use as the session name
        session_name = get_timestamp(time_separator="-")

        # Constructs the session directory path and generates the directory
        raw_session_path = self._local.joinpath(session_name)

        # Handles potential session name conflicts. While this is extremely unlikely, it is not impossible for
        # such conflicts to occur.
        counter = 0
        while raw_session_path.exists():
            counter += 1
            new_session_name = f"{session_name}_{counter}"
            raw_session_path = self._local.joinpath(new_session_name)

        # If a conflict is detected and resolved, warns the user about the resolved conflict.
        if counter > 0:
            message = (
                f"Session name conflict occurred for animal '{self._animal_name}' of project '{self._project_name}' "
                f"when adding the new session with timestamp {session_name}. The session with identical name "
                f"already exists. The newly created session directory uses a '_{counter}' postfix to distinguish "
                f"itself from the already existing session directory."
            )
            warnings.warn(message=message)

        # Saves the session name to class attribute
        self._session_name: str = raw_session_path.stem

        # Modifies the local path to point to the raw_data directory under the session directory. The 'raw_data'
        # directory acts as the root for all raw and preprocessed data generated during session runtime.
        self._local = self._local.joinpath(self._session_name, "raw_data")

        # Modifies nas and server paths to point to the session directory. The "raw_data" folder will be created during
        # data movement method runtime, so the folder is not precreated for destinations.
        self._server: Path = server_root_directory.joinpath(self._project_name, self._animal_name, self._session_name)
        self._nas: Path = nas_root_directory.joinpath(self._project_name, self._animal_name, self._session_name)

        # Ensures that root paths exist for all destinations and sources. Note
        ensure_directory_exists(self._local)
        ensure_directory_exists(self._nas)
        ensure_directory_exists(self._server)
        if not generate_mesoscope_paths:
            ensure_directory_exists(self._mesoscope)
        ensure_directory_exists(self._persistent)

    @property
    def raw_data_path(self) -> Path:
        """Returns the path to the 'raw_data' directory of the managed session.

        The raw_data is the root directory for aggregating all acquired and preprocessed data. This path is primarily
        used by MesoscopeExperiment class to determine where to save captured videos, data logs, and other acquired data
        formats. After runtime, SessionData pulls all mesoscope frames into this directory.
        """
        return self._local

    @property
    def mesoscope_frames_path(self) -> Path:
        """Returns the path to the mesoscope_frames directory of the managed session.

        This path is used during mesoscope data preprocessing to store compressed (preprocessed) mesoscope frames.
        """
        directory_path = self._local.joinpath("mesoscope_frames")

        # Since this is a directory, we need to ensure it exists before this path is returned to caller
        ensure_directory_exists(directory_path)
        return directory_path

    @property
    def ops_path(self) -> Path:
        """Returns the path to the ops.json file of the managed session.

        This path is used to save the ops.json generated from the mesoscope TIFF metadata during preprocessing. This is
        a configuration file used by the suite2p during mesoscope data registration.
        """
        return self.mesoscope_frames_path.joinpath("ops.json")

    @property
    def frame_invariant_metadata_path(self) -> Path:
        """Returns the path to the frame_invariant_metadata.json file of the managed session.

        This path is used to save the metadata shared by all frames in all TIFF stacks acquired by the mesoscope.
        Currently, this data is not used during processing.
        """
        return self.mesoscope_frames_path.joinpath("frame_invariant_metadata.json")

    @property
    def frame_variant_metadata_path(self) -> Path:
        """Returns the path to the frame_variant_metadata.npz file of the managed session.

        This path is used to save the metadata unique for each frame in all TIFF stacks acquired by the mesoscope.
        Currently, this data is not used during processing.

        Notes:
            Unlike frame-invariant metadata, this file is stored as a compressed NumPy archive (NPZ) file to optimize
            storage space usage.
        """
        return self.mesoscope_frames_path.joinpath("frame_variant_metadata.npz")

    @property
    def camera_frames_path(self) -> Path:
        """Returns the path to the camera_frames directory of the managed session.

        This path is used during camera data preprocessing to store the videos and extracted frame timestamps for all
        video cameras used to record behavior.
        """
        directory_path = self._local.joinpath("camera_frames")

        # Since this is a directory, we need to ensure it exists before this path is returned to caller
        ensure_directory_exists(directory_path)
        return directory_path

    @property
    def zaber_positions_path(self) -> Path:
        """Returns the path to the zaber_positions.yaml file of the managed session.

        This path is used to save the positions for all Zaber motors of the HeadBar and LickPort controllers at the
        end of the experimental session. This allows restoring the motors to those positions during the following
        experimental session(s).
        """
        return self._local.joinpath("zaber_positions.yaml")

    @property
    def session_descriptor_path(self) -> Path:
        """Returns the path to the session_descriptor.yaml file of the managed session.

        This path is used to save important session information to be viewed by experimenters post-runtime and to use
        for further processing. This includes the type of the session (e.g. 'lick_training') and the total volume of
        water delivered during runtime (important for water restriction).
        """
        return self._local.joinpath("session_descriptor.yaml")

    @property
    def camera_timestamps_path(self) -> Path:
        """Returns the path to the camera_timestamps.parquet file of the managed session.

        This path is used to save the timestamps associated with each saved frame acquired by each camera during
        runtime. This data is later used to align the tracking data extracted from camera frames via DeepLabCut with
        other behavioral data stored in the main parquet dataset file.
        """
        return self.camera_frames_path.joinpath("camera_timestamps.parquet")

    @property
    def behavioral_data_path(self) -> Path:
        """Returns the path to the behavioral_data.parquet file of the managed session.

        This path is used to save the behavioral dataset assembled from the data logged during runtime by the central
        process and the AtaraxisMicroController modules. This dataset is assembled via Polars and stored as a
        parquet file. It contains all behavioral data other than video-tracking. For experiment runtimes, the data is
        aligned to the mesoscope frame acquisition timestamps. For training runtimes, the data is interpolated to
        include all time-points from all recorded sources.
        """
        return self._local.joinpath("behavioral_data.parquet")

    @property
    def previous_zaber_positions_path(self) -> Path:
        """Returns the path to the zaber_positions.yaml file of the previous session.

        The file is stored inside the 'persistent' directory of the project and animal combination. The file is saved to
        the persistent directory when the original session is moved to long-term storage. Loading the file allows
        reusing LickPort and HeadBar motor positions across sessions. The contents of this file are updated after each
        experimental or training session.
        """
        file_path = self._persistent.joinpath("zaber_positions.yaml")
        return file_path

    @property
    def persistent_motion_estimator_path(self) -> Path:
        """Returns the path to the MotionEstimator.me file for the managed animal and project combination stored on the
        ScanImagePC.

        This path is used during the first training session to save the 'reference' MotionEstimator.me file established
        during the initial mesoscope ROI selection to the ScanImagePC. The same reference file is used for all following
        sessions to correct for the natural motion of the brain relative to the cranial window.
        """
        return self._mesoscope.joinpath("persistent_data", self._project_name, self._animal_name, "MotionEstimator.me")

    def pull_mesoscope_data(
        self, num_threads: int = 28, remove_sources: bool = False, verify_transfer_integrity: bool = False
    ) -> None:
        """Pulls the frames acquired by the mesoscope from the ScanImage PC to the VRPC.

        This method should be called after the data acquisition runtime to aggregate all recorded data on the VRPC
        before running the preprocessing pipeline. The method expects that the mesoscope frames source directory
        contains only the frames acquired during the current session runtime, and the MotionEstimator.me and
        zstack.mat used for motion registration.

        Notes:
            This method is configured to parallelize data transfer and verification to optimize runtime speeds where
            possible.

            When the method is called for the first time for a particular project and animal combination, it also
            'persists' the MotionEstimator.me file before moving all mesoscope data to the VRPC. This creates the
            reference for all further motion estimation procedures carried out during future sessions.

        Args:
            num_threads: The number of parallel threads used for transferring the data from ScanImage (mesoscope) PC to
                the local machine. Depending on the connection speed between the PCs, it may be useful to set this
                number to the number of available CPU cores - 4.
            remove_sources: Determines whether to remove the transferred mesoscope frame data from the ScanImagePC.
                Generally, it is recommended to remove source data to keep ScanImagePC disk usage low.
            verify_transfer_integrity: Determines whether to verify the integrity of the transferred data. This is
                performed before source folder is removed from the ScanImagePC, if remove_sources is True.
        Raises:
            RuntimeError: If the mesoscope source directory does not contain motion estimator files or mesoscope frames.
        """
        # Resolves source and destination paths
        source = self._mesoscope.joinpath("mesoscope_frames")
        destination = self.raw_data_path  # The path to the raw_data subdirectory of the current session

        # Extracts the names of files stored in the source folder
        files = tuple([path for path in source.glob("*")])

        # Ensures the folder contains motion estimator data files
        if "MotionEstimator.me" not in files:
            message = (
                f"Unable to pull the mesoscope-acquired data from the ScanImage PC to the VRPC. The 'mesoscope_frames' "
                f"ScanImage PC directory does not contain the MotionEstimator.me file, which is required for further "
                f"frame data processing."
            )
            console.error(message=message, error=RuntimeError)
        if "zstack.mat" not in files:
            message = (
                f"Unable to pull the mesoscope-acquired data from the ScanImage PC to the VRPC. The 'mesoscope_frames' "
                f"ScanImage PC directory does not contain the zstack.mat file, which is required for further "
                f"frame data processing."
            )
            console.error(message=message, error=RuntimeError)

        # Prevents 'pulling' an empty folder. At a minimum, we expect 2 motion estimation files and one TIFF stack file
        if len(files) < 3:
            message = (
                f"Unable to pull the mesoscope-acquired data from the ScanImage PC to the VRPC. The 'mesoscope_frames' "
                f"ScanImage PC directory does not contain the minimum expected number of files (3). This indicates "
                f"that no frames were acquired during runtime or that the frames were saved at a different location."
            )
            console.error(message=message, error=RuntimeError)

        # If the processed project and animal combination does not have a reference MotionEstimator.me saved in the
        # persistent ScanImagePC directory, copies the MotionEstimator.me to the persistent directory. This ensures that
        # the first ever created MotionEstimator.me is saved as the reference MotionEstimator.me for further sessions.
        if not self.persistent_motion_estimator_path.exists():
            ensure_directory_exists(self.persistent_motion_estimator_path)
            shutil.copy2(src=source.joinpath("MotionEstimator.me"), dst=self.persistent_motion_estimator_path)

        # Generates the checksum for the source folder if transfer integrity verification is enabled.
        if verify_transfer_integrity:
            calculate_directory_checksum(directory=source, num_processes=None, save_checksum=True)

        # Transfers the mesoscope frames data from the ScanImage PC to the local machine.
        transfer_directory(
            source=source, destination=destination, num_threads=num_threads, verify_integrity=verify_transfer_integrity
        )

        # Removes the checksum file after the transfer is complete. The checksum will be recalculated for the whole
        # session directory during preprocessing, so there is no point in keeping the original mesoscope checksum file.
        if verify_transfer_integrity:
            destination.joinpath("ax_checksum.txt").unlink(missing_ok=True)

        # After the transfer completes successfully (including integrity verification), recreates the mesoscope_frames
        # folder to remove the transferred data from the ScanImage PC.
        if remove_sources:
            shutil.rmtree(source)
            ensure_directory_exists(source)

    def process_mesoscope_data(self) -> None:
        """Preprocesses the (pulled) mesoscope data.

        This is a wrapper around the process_mesoscope_directory() function. It compressed all mesoscope frames using
        LERC, extracts and save sframe-invariant and frame-variant metadata, and generates an ops.json file for future
        suite2p registration. This method also ensures that after processing all mesoscope data, including motion
        estimation files, are found under the mesoscope_frames directory.

        Notes:
            Additional processing for camera data is carried out by the stop() method of each experimental class. This
            is intentional, as not all classes use all cameras.
        """
        # Preprocesses the pulled mesoscope frames.
        process_mesoscope_directory(
            image_directory=self.raw_data_path,
            output_directory=self.mesoscope_frames_path,
            ops_path=self.ops_path,
            frame_invariant_metadata_path=self.frame_invariant_metadata_path,
            frame_variant_metadata_path=self.frame_variant_metadata_path,
            num_processes=28,
            remove_sources=True,
            verify_integrity=True,
        )

        # Cleans up some data inconsistencies. Moves motion estimator files to the mesoscope_frames directory generated
        # during mesoscope data preprocessing. This way, ALL mesoscope-related data is stored under mesoscope_frames.
        shutil.move(
            src=self.raw_data_path.joinpath("MotionEstimator.me"),
            dst=self.mesoscope_frames_path.joinpath("MotionEstimator.me"),
        )
        shutil.move(
            src=self.raw_data_path.joinpath("zstack.mat"),
            dst=self.mesoscope_frames_path.joinpath("zstack.mat"),
        )

    def push_data(
        self,
        parallel: bool = True,
        num_threads: int = 10,
        remove_sources: bool = False,
        verify_transfer_integrity: bool = False,
    ) -> None:
        """Copies the raw_data directory from the VRPC to the NAS and the BioHPC server.

        This method should be called after data acquisition and preprocessing to move the prepared data to the NAS and
        the server. This method generates the xxHash3-128 checksum for the source folder and, if configured, verifies
        that the transferred data produces the same checksum to ensure data integrity.

        Notes:
            This method is configured to run data transfer and checksum calculation in parallel where possible. It is
            advised to minimize the use of the host-machine while it is running this method, as most CPU resources will
            be consumed by the data transfer process.

            The method also replaces the persisted zaber_positions.yaml file with the file generated during the managed
            session runtime. This ensures that the persisted file is always up to date with the current zaber motor
            positions.

        Args:
            parallel: Determines whether to parallelize the data transfer. When enabled, the method will transfer the
                data to all destinations at the same time (in-parallel). Note, this argument does not affect the number
                of parallel threads used by each transfer process or the number of threads used to compute the
                xxHash3-128 checksum. This is determined by the 'num_threads' argument (see below).
            num_threads: Determines the number of threads used by each transfer process to copy the files and calculate
                the xxHash3-128 checksums. Since each process uses the same number of threads, it is highly
                advised to set this value so that num_threads * 2 (number of destinations) does not exceed the total
                number of CPU cores - 4.
            remove_sources: Determines whether to remove the raw_data directory from the VRPC once it has been copied
                to the NAS and Server. Depending on the overall load of the VRPC, we recommend keeping source data on
                the VRPC at least until the integrity of the transferred data is verified on the server.
            verify_transfer_integrity: Determines whether to verify the integrity of the transferred data. This is
                performed before source folder is removed from the VRPC, if remove_sources is True.
        """
        # Resolves source and destination paths
        source = self.raw_data_path

        # Destinations include short destination names used for progress reporting
        destinations = (
            (self._nas.joinpath("raw_data"), "NAS"),
            (self._server.joinpath("raw_data"), "Server"),
        )

        # Updates the zaber_positions.yaml file stored inside the persistent directory for the project+animal
        # combination with the zaber_positions.yaml file from the current session. This ensures that the zaber_positions
        # file is always set to the latest snapshot of zaber motor positions.
        if self.zaber_positions_path.exists():
            self.previous_zaber_positions_path.unlink(missing_ok=True)  # Removes the previous persisted file
            shutil.copy2(self.zaber_positions_path, self.previous_zaber_positions_path)  # Persists the current file

        # Resolves the destination paths based on the provided short destination names
        destinations = [(dest[0], dest[1]) for dest in destinations]

        # Computes the xxHash3-128 checksum for the source folder
        calculate_directory_checksum(directory=source, num_processes=None, save_checksum=True)

        # If the method is configured to transfer files in parallel, submits tasks to a ProcessPoolExecutor
        if parallel:
            with ProcessPoolExecutor(max_workers=len(destinations)) as executor:
                futures = {
                    executor.submit(
                        transfer_directory,
                        source=source,
                        destination=dest[0],
                        num_threads=num_threads,
                        verify_integrity=verify_transfer_integrity,
                    ): dest
                    for dest in destinations
                }
                for future in as_completed(futures):
                    # Propagates any exceptions from the transfers
                    future.result()

        # Otherwise, runs the transfers sequentially. Note, transferring individual files is still done in parallel, but
        # the transfer is performed for each destination sequentially.
        else:
            for destination in destinations:
                transfer_directory(
                    source=source,
                    destination=destination[0],
                    num_threads=num_threads,
                    verify_integrity=verify_transfer_integrity,
                )

        # After all transfers complete successfully, removes the source directory, if requested
        if remove_sources:
            shutil.rmtree(source)


class MesoscopeExperiment:
    """The base class for all mesoscope experiment runtimes.

    This class provides methods for conducting experiments using the Mesoscope-VR system. It abstracts most
    low-level interactions with the VR system and the mesoscope via a simple high-level state API.

    This class also provides methods for limited preprocessing of the collected data. The preprocessing methods are
    designed to be executed after each experiment runtime to prepare the data for long-term storage and transmission
    over the network. Preprocessing methods use multiprocessing to optimize runtime performance and assume that the
    VRPC is kept mostly idle during data preprocessing.

    Notes:
        Calling this initializer does not start the Mesoscope-VR components. Use the start() method before issuing other
        commands to properly initialize all remote processes. This class reserves up to 11 CPU cores during runtime.

        Use the 'axtl-ports' cli command to discover the USB ports used by Ataraxis Micro Controller (AMC) devices.

        Use the 'axvs-ids' cli command to discover the camera indices used by the Harvesters-managed and
        OpenCV-managed cameras.

        Use the 'sl-devices' cli command to discover the serial ports used by the Zaber motion controllers.

        This class statically reserves the id code '1' to label its log entries. Make sure no other Ataraxis class, such
        as MicroControllerInterface or VideoSystem, uses this id code.

    Args:
        session_data: An initialized SessionData instance. This instance is used to transfer the data between VRPC,
            ScanImagePC, BioHPC server, and the NAS during runtime. Each instance is initialized for the specific
            project, animal, and session combination for which the data is acquired.
        descriptor: A partially configured _LickTrainingDescriptor or _RunTrainingDescriptor instance. This instance is
            used to store session-specific information in a human-readable format.
        cue_length_map: A dictionary that maps each integer-code associated with a wall cue used in the Virtual Reality
            experiment environment to its length in real-world centimeters. Ity is used to map each VR cue to the
            distance the mouse needs to travel to fully traverse the wall cue region from start to end.
        screens_on: Communicates whether the VR screens are currently ON at the time of class initialization.
        experiment_state: The integer code that represents the initial state of the experiment. Experiment state codes
            are used to mark different stages of each experiment (such as rest_1, run_1, rest_2, etc...). During
            analysis, these codes can be used to segment experimental data into sections.
        actor_port: The USB port used by the actor Microcontroller.
        sensor_port: The USB port used by the sensor Microcontroller.
        encoder_port: The USB port used by the encoder Microcontroller.
        headbar_port: The USB port used by the headbar Zaber motor controllers (devices).
        lickport_port: The USB port used by the lickport Zaber motor controllers (devices).
        unity_ip: The IP address of the MQTT broker used to communicate with the Unity game engine.
        unity_port: The port number of the MQTT broker used to communicate with the Unity game engine.
        valve_calibration_data: A tuple of tuples, with each inner tuple storing a pair of values. The first value is
            the duration, in microseconds, the valve was open. The second value is the volume of dispensed water, in
            microliters.
        face_camera_index: The index of the face camera in the list of all available Harvester-managed cameras.
        left_camera_index: The index of the left camera in the list of all available OpenCV-managed cameras.
        right_camera_index: The index of the right camera in the list of all available OpenCV-managed cameras.
        harvesters_cti_path: The path to the GeniCam CTI file used to connect to Harvesters-managed cameras.

    Attributes:
        _started: Tracks whether the VR system and experiment runtime are currently running.
        _descriptor: Stores the session descriptor instance.
        _logger: A DataLogger instance that collects behavior log data from all sources: microcontrollers, video
            cameras, and the MesoscopeExperiment instance.
        _microcontrollers: Stores the _MicroControllerInterfaces instance that interfaces with all MicroController
            devices used during runtime.
        _cameras: Stores the _VideoSystems instance that interfaces with video systems (cameras) used during
            runtime.
        _headbar: Stores the _HeadBar class instance that interfaces with all HeadBar manipulator motors.
        _lickport: Stores the _LickPort class instance that interfaces with all LickPort manipulator motors.
        _vr_state: Stores the current state of the VR system. The MesoscopeExperiment updates this value whenever it is
            instructed to change the VR system state.
        _state_map: Maps the integer state-codes used to represent VR system states to human-readable string-names.
        _experiment_state: Stores the user-defined experiment state. Experiment states are defined by the user and
            are expected to be unique for each project and, potentially, experiment. Different experiment states can
            reuse the same VR state.
        _timestamp_timer: A PrecisionTimer instance used to timestamp log entries generated by the class instance.
        _source_id: Stores the unique identifier code for this class instance. The identifier is used to mark log
            entries made by this class instance and has to be unique across all sources that log data at the same time,
            such as MicroControllerInterfaces and VideoSystems.
        _cue_map: Stores the dictionary that maps integer-codes associated with each VR wall cue with
            its length in centimeters.
        _session_data: Stores the SessionData instance used to manage the acquired data.

    Raises:
        TypeError: If session_data or cue_length_map arguments have invalid types.
    """

    # Maps integer VR state codes to human-readable string-names.
    _state_map: dict[int, str] = {0: "Idle", 1: "Rest", 2: "Run"}

    def __init__(
        self,
        session_data: SessionData,
        descriptor: _MesoscopeExperimentDescriptor,
        cue_length_map: dict[int, float],
        screens_on: bool = False,
        experiment_state: int = 0,
        actor_port: str = "/dev/ttyACM0",
        sensor_port: str = "/dev/ttyACM1",
        encoder_port: str = "/dev/ttyACM2",
        headbar_port: str = "/dev/ttyUSB0",
        lickport_port: str = "/dev/ttyUSB1",
        unity_ip: str = "127.0.0.1",
        unity_port: int = 1883,
        valve_calibration_data: tuple[tuple[int | float, int | float], ...] = (
            (15000, 1.8556),
            (30000, 3.4844),
            (45000, 7.1846),
            (60000, 10.0854),
        ),
        face_camera_index: int = 0,
        left_camera_index: int = 0,
        right_camera_index: int = 2,
        harvesters_cti_path: Path = Path("/opt/mvIMPACT_Acquire/lib/x86_64/mvGenTLProducer.cti"),
    ) -> None:
        # Activates the console to display messages to the user if the console is disabled when the class is
        # instantiated.
        if not console.enabled:
            console.enable()

        # Creates the _started flag first to avoid leaks if the initialization method fails.
        self._started: bool = False
        self._descriptor: _MesoscopeExperimentDescriptor = descriptor

        # Input verification:
        if not isinstance(session_data, SessionData):
            message = (
                f"Unable to initialize the MesoscopeExperiment class. Expected a SessionData instance for "
                f"'session_data' argument, but instead encountered {session_data} of type "
                f"{type(session_data).__name__}."
            )
            console.error(message=message, error=TypeError)
        if not isinstance(cue_length_map, dict):
            message = (
                f"Unable to initialize the MesoscopeExperiment class. Expected a dictionary for 'cue_length_map' "
                f"argument, but instead encountered {cue_length_map} of type {type(cue_length_map).__name__}."
            )
            console.error(message=message, error=TypeError)

        # Defines other flags used during runtime:
        self._vr_state: int = 0  # Stores the current state of the VR system
        self._experiment_state: int = experiment_state  # Stores user-defined experiment state
        self._timestamp_timer: PrecisionTimer = PrecisionTimer("us")  # A timer used to timestamp log entries
        self._source_id: np.uint8 = np.uint8(1)  # Reserves source ID code 1 for this class

        # This dictionary is used to convert distance traveled by the animal into the corresponding sequence of
        # traversed cues (corridors).
        self._cue_map: dict[int, float] = cue_length_map

        # Saves the SessionData instance to class attribute so that it can be used from class methods. Since SessionData
        # resolves session directory structure at initialization, the instance is ready to resolve all paths used by
        # the experiment class instance.
        self._session_data: SessionData = session_data

        # Initializes the DataLogger instance used to log data from all microcontrollers, camera frame savers, and this
        # class instance.
        self._logger: DataLogger = DataLogger(
            output_directory=session_data.raw_data_path,
            instance_name="behavior",  # Creates behavior_log subfolder under raw_data
            sleep_timer=0,
            exist_ok=True,
            process_count=1,
            thread_count=10,
        )

        # Initializes the binding class for all MicroController Interfaces.
        self._microcontrollers: _MicroControllerInterfaces = _MicroControllerInterfaces(
            data_logger=self._logger,
            screens_on=screens_on,
            actor_port=actor_port,
            sensor_port=sensor_port,
            encoder_port=encoder_port,
            valve_calibration_data=valve_calibration_data,
            debug=False,
        )

        # Also instantiates an MQTTCommunication instance to directly communicate with Unity. Currently, this is used
        # exclusively to verify that the Unity is running and to collect the sequence of VR wall cues used by the task.
        monitored_topics = ("CueSequence/",)
        self._unity: MQTTCommunication = MQTTCommunication(
            ip=unity_ip, port=unity_port, monitored_topics=monitored_topics
        )

        # Initializes the binding class for all VideoSystems.
        self._cameras: _VideoSystems = _VideoSystems(
            data_logger=self._logger,
            output_directory=self._session_data.camera_frames_path,
            face_camera_index=face_camera_index,
            left_camera_index=left_camera_index,
            right_camera_index=right_camera_index,
            harvesters_cti_path=harvesters_cti_path,
        )

        # While we can connect to ports managed by ZaberLauncher, ZaberLauncher cannot connect to ports managed via
        # software. Therefore, we have to make sure ZaberLauncher is running before connecting to motors.
        message = (
            "Preparing to connect to HeadBar and LickPort Zaber controllers. Make sure that ZaberLauncher app is "
            "running before proceeding further. If ZaberLauncher is not running, youi WILL NOT be able to manually "
            "control the HeadBar and LickPort motor positions until you reset the runtime."
        )
        console.echo(message=message, level=LogLevel.WARNING)
        while input("Enter 'y' to continue: ") != "y":
            continue

        # Initializes the binding classes for the HeadBar and LickPort manipulator motors.
        self._headbar: _HeadBar = _HeadBar(
            headbar_port=headbar_port, zaber_positions_path=self._session_data.previous_zaber_positions_path
        )
        self._lickport: _LickPort = _LickPort(
            lickport_port=lickport_port, zaber_positions_path=self._session_data.previous_zaber_positions_path
        )

    def start(self) -> None:
        """Sets up all assets used during the experiment.

        This internal method establishes the communication with the microcontrollers, data logger cores, and video
        system processes. It also requests the cue sequence from Unity game engine and starts mesoscope frame
        acquisition process.

        Notes:
            This method will not run unless the host PC has access to the necessary number of logical CPU cores
            and other required hardware resources (GPU for video encoding, etc.). This prevents using the class on
            machines that are unlikely to sustain the runtime requirements.

            As part of its runtime, this method will attempt to set Zaber motors for the HeadBar and LickPort to the
            positions optimal for mesoscope frame acquisition. Exercise caution and always monitor the system when
            it is running this method, as unexpected motor behavior can damage the mesoscope or harm the animal.

        Raises:
            RuntimeError: If the host PC does not have enough logical CPU cores available.
        """
        # Prevents (re) starting an already started VR process.
        if self._started:
            return

        # 3 cores for microcontrollers, 1 core for the data logger, 6 cores for the current video_system
        # configuration (3 producers, 3 consumer), 1 core for the central process calling this method. 11 cores
        # total.
        if not os.cpu_count() >= 11:
            message = (
                f"Unable to start the MesoscopeExperiment runtime. The host PC must have at least 11 logical CPU "
                f"cores available for this class to work as expected, but only {os.cpu_count()} cores are "
                f"available."
            )
            console.error(message=message, error=RuntimeError)

        message = "Initializing MesoscopeExperiment assets..."
        console.echo(message=message, level=LogLevel.INFO)

        # Starts the data logger
        self._logger.start()

        # Generates and logs the onset timestamp for the VR system as a whole. The MesoscopeExperiment class logs
        # changes to VR and Experiment state during runtime, so it needs to have the onset stamp, just like all other
        # classes that generate data logs.

        # Constructs the timezone-aware stamp using UTC time. This creates a reference point for all later delta time
        # readouts. The time is returned as an array of bytes.
        onset: NDArray[np.uint8] = get_timestamp(as_bytes=True)  # type: ignore
        self._timestamp_timer.reset()  # Immediately resets the timer to make it as close as possible to the onset time

        # Logs the onset timestamp. All further timestamps will be treated as integer time deltas (in microseconds)
        # relative to the onset timestamp. Note, ID of 1 is used to mark the main experiment system.
        package = LogPackage(
            source_id=self._source_id, time_stamp=np.uint8(0), serialized_data=onset
        )  # Packages the id, timestamp, and data.
        self._logger.input_queue.put(package)

        message = "DataLogger: Started."
        console.echo(message=message, level=LogLevel.SUCCESS)

        # Starts the face camera. This starts frame acquisition and displays acquired frames to the user. However,
        # frame saving is disabled at this time. Body cameras are also disabled. This is intentional, as at this point
        # we want to minimize the number of active processes. This is helpful if this method is called while the
        # previous session is still running its data preprocessing pipeline and needs as many free cores as possible.
        self._cameras.start_face_camera()

        # Initializes the Zaber positioning sequence. This relies heavily on user feedback to confirm that it is safe to
        # proceed with motor movements.
        message = (
            "Preparing to move HeadBar into position. Remove the mesoscope objective, swivel out the VR screens, "
            "and make sure the animal is NOT mounted on the rig. Failure to fulfill these steps may DAMAGE the "
            "mesoscope and / or HARM the animal."
        )
        console.echo(message=message, level=LogLevel.WARNING)
        while input("Enter 'y' to continue: ") != "y":
            continue

        # Homes all motors in-parallel. The homing trajectories for the motors as they are used now should not intersect
        # with each other, so it is safe to move both assemblies at the same time.
        self._headbar.prepare_motors(wait_until_idle=False)
        self._lickport.prepare_motors(wait_until_idle=True)
        self._headbar.wait_until_idle()

        # Sets the motors into the mounting position. The HeadBar is either restored to the previous session position or
        # is set to the default mounting position stored in non-volatile memory. The LickPort is moved to a position
        # optimized for putting the animal on the VR rig.
        self._headbar.restore_position(wait_until_idle=False)
        self._lickport.mount_position(wait_until_idle=True)
        self._headbar.wait_until_idle()

        # Gives user time to mount the animal and requires confirmation before proceeding further.
        message = (
            "Preparing to move the LickPort into position. Mount the animal onto the VR rig and install the mesoscope "
            "objetive. If necessary, adjust the HeadBar position to make sure the animal can comfortably run the task."
        )
        console.echo(message=message, level=LogLevel.WARNING)
        while input("Enter 'y' to continue: ") != "y":
            continue

        # Restores the lickPort to the previous session's position or to the default parking position. This positions
        # the LickPort in a way that is easily accessible by the animal.
        self._lickport.restore_position()
        message = (
            "If necessary, adjust LickPort position to be easily reachable by the animal and position the mesoscope "
            "objective above the imaging field. Take extra care when moving the LickPort towards the animal! Run any "
            "mesoscope preparation procedures, such as motion correction, before proceeding further. This is the last "
            "manual checkpoint, entering 'y' after this message will begin the experiment."
        )
        console.echo(message=message, level=LogLevel.WARNING)
        while input("Enter 'y' to continue: ") != "y":
            continue

        # Generates a snapshot of all zaber positions. This serves as an early checkpoint in case the runtime has to be
        # aborted in a non-graceful way (without running the stop() sequence). This way, next runtime will restart with
        # the calibrated zaber positions.
        head_bar_positions = self._headbar.get_positions()
        lickport_positions = self._lickport.get_positions()
        zaber_positions = _ZaberPositions(
            headbar_z=head_bar_positions[0],
            headbar_pitch=head_bar_positions[1],
            headbar_roll=head_bar_positions[2],
            lickport_z=lickport_positions[0],
            lickport_x=lickport_positions[1],
            lickport_y=lickport_positions[2],
        )
        zaber_positions.to_yaml(file_path=self._session_data.zaber_positions_path)
        message = "HeadBar and LickPort positions: Saved."
        console.echo(message=message, level=LogLevel.SUCCESS)

        # Enables body cameras. Starts frame saving for all cameras
        self._cameras.start_body_cameras()
        self._cameras.save_face_camera_frames()
        self._cameras.save_body_camera_frames()

        # Starts all microcontroller interfaces
        self._microcontrollers.start()

        # Establishes a direct communication with Unity over MQTT. This is in addition to some ModuleInterfaces using
        # their own communication channels.
        self._unity.connect()

        # Queries the task cue (segment) sequence from Unity. This also acts as a check for whether Unity is
        # running and is configured appropriately. The extracted sequence data is logged as a sequence of byte
        # values.
        cue_sequence = self._get_cue_sequence()
        package = LogPackage(
            source_id=self._source_id,
            time_stamp=np.uint64(self._timestamp_timer.elapsed),
            serialized_data=cue_sequence,
        )
        self._logger.input_queue.put(package)

        message = "Unity Game Engine: Started."
        console.echo(message=message, level=LogLevel.SUCCESS)

        # Starts monitoring the sensors used regardless of the VR state. Currently, this is the lick sensor state and
        # the mesoscope frame ttl module state.
        self._microcontrollers.enable_mesoscope_frame_monitoring()
        self._microcontrollers.enable_lick_monitoring()

        # Sets the rest of the subsystems to use the REST state.
        self.vr_rest()

        # Starts mesoscope frame acquisition. This also verifies that the mesoscope responds to triggers and
        # actually starts acquiring frames using the _mesoscope_frame interface above.
        self._start_mesoscope()

        message = "Mesoscope frame acquisition: Started."
        console.echo(message=message, level=LogLevel.SUCCESS)

        # The setup procedure is complete.
        self._started = True

        message = "MesoscopeExperiment assets: Initialized."
        console.echo(message=message, level=LogLevel.SUCCESS)

    def stop(self) -> None:
        """Stops and terminates the MesoscopeExperiment runtime.

        This method achieves two main purposes. First, releases the hardware resources used during the experiment
        runtime by various system components. Second, it pulls all collected data to the VRPC and runs the preprocessing
        pipeline on the data to prepare it for long-term storage and further processing.
        """

        # Prevents stopping an already stopped VR process.
        if not self._started:
            return

        message = "Terminating MesoscopeExperiment runtime..."
        console.echo(message=message, level=LogLevel.INFO)

        # Resets the _started tracker
        self._started = False

        # Switches the system into the rest state. Since REST state has most modules set to stop-friendly states,
        # this is used as a shortcut to prepare the VR system for shutdown.
        self.vr_rest()

        # Stops mesoscope frame acquisition.
        self._microcontrollers.stop_mesoscope()
        self._timestamp_timer.reset()  # Resets the timestamp timer. It is now co-opted to enforce the shutdown delay
        message = "Mesoscope stop command: Sent."
        console.echo(message=message, level=LogLevel.SUCCESS)

        # Stops all cameras.
        self._cameras.stop()

        # Manually stops hardware modules not stopped by the REST state. This excludes mesoscope frame monitoring, which
        # is stopped separately after the 5-second delay (see below).
        self._microcontrollers.disable_lick_monitoring()
        self._microcontrollers.disable_torque_monitoring()

        # Delays for 5 seconds to give mesoscope time to stop acquiring frames. Primarily, this ensures that all
        # mesoscope frames have recorded acquisition timestamps. This implementation times the delay relative to the
        # mesoscope shutdown command and allows running other shutdown procedures in-parallel with the mesoscope
        # shutdown processing.
        while self._timestamp_timer.elapsed < 5000000:
            continue

        # Stops mesoscope frame monitoring. At this point, the mesoscope should have stopped acquiring frames.
        self._microcontrollers.disable_mesoscope_frame_monitoring()

        # Stops all microcontroller interfaces
        self._microcontrollers.stop()

        # Stops the data logger instance
        self._logger.stop()

        message = "Data Logger: stopped."
        console.echo(message=message, level=LogLevel.SUCCESS)

        # Updates the contents of the pregenerated descriptor file and dumps it as a .yaml into the root raw_data
        # session directory. This needs to be done after the microcontrollers and loggers have been stopped to ensure
        # that the reported dispensed_water_volume_ul is accurate.
        delivered_water = self._microcontrollers.total_delivered_volume
        # Overwrites the delivered water volume with the volume recorded over the runtime.
        self._descriptor.dispensed_water_volume_ul = delivered_water
        self._descriptor.to_yaml(file_path=self._session_data.session_descriptor_path)

        # Generates the snapshot of the current HeadBar and LickPort positions and saves them as a .yaml file. This has
        # to be done before Zaber motors are reset back to parking position.
        head_bar_positions = self._headbar.get_positions()
        lickport_positions = self._lickport.get_positions()
        zaber_positions = _ZaberPositions(
            headbar_z=head_bar_positions[0],
            headbar_pitch=head_bar_positions[1],
            headbar_roll=head_bar_positions[2],
            lickport_z=lickport_positions[0],
            lickport_x=lickport_positions[1],
            lickport_y=lickport_positions[2],
        )
        zaber_positions.to_yaml(file_path=self._session_data.zaber_positions_path)

        # Moves the LickPort to the mounting position to assist removing the animal from the rig.
        self._lickport.mount_position()

        # Notifies the user about the volume of water dispensed during runtime, so that they can ensure the mouse
        # get any leftover daily water limit.
        message = (
            f"During runtime, the system dispensed ~{delivered_water} uL of water to the animal. "
            f"If the animal is on water restriction, make sure it receives any additional water, if the dispensed "
            f"volume does not cover the daily water limit for that animal."
        )
        console.echo(message=message, level=LogLevel.INFO)

        # Instructs the user to remove the objective and the animal before resetting all zaber motors.
        message = (
            "Preparing to reset the HeadBar and LickPort motors. Uninstall the mesoscope objective, remove the animal "
            "from the VR rig and swivel the VR screens out. Failure to do so may DAMAGE the mesoscope objective and "
            "HARM the animal."
        )
        console.echo(message=message, level=LogLevel.WARNING)
        while input("Enter 'y' to continue: ") != "y":
            continue

        # Parks both controllers and then disconnects from their Connection classes. Note, the parking is performed
        # in-parallel
        self._headbar.park_position(wait_until_idle=False)
        self._lickport.park_position(wait_until_idle=True)
        self._headbar.wait_until_idle()
        self._headbar.disconnect()
        self._lickport.disconnect()

        message = "HeadBar and LickPort motors: Reset."
        console.echo(message=message, level=LogLevel.SUCCESS)

        # Prompts the user to add their notes to the appropriate section of the descriptor file. This has to be done
        # before processing so that the notes are properly transferred to the NAS and server. Also, this makes it more
        # obvious to the user when it is safe to start preparing for the next session and leave the current one
        # processing the data.
        message = (
            f"Data acquisition: Complete. Open the session descriptor file located at "
            f"{self._session_data.session_descriptor_path} and update the notes session with the notes taken during "
            f"runtime. This is the last manual checkpoint, entering 'y' after this message will begin data "
            f"preprocessing and, after that, transmit it to the BioHPC server and the NAS storage. It is safe to start "
            f"preparing for the next session after hitting 'y'."
        )
        console.echo(message=message, level=LogLevel.WARNING)
        while input("Enter 'y' to continue: ") != "y":
            continue

        message = "Initializing data preprocessing..."
        console.echo(message=message, level=LogLevel.INFO)

        # Compresses all logs into a single .npz file. This is done both for long-term storage optimization and to
        # allow parsing the data. Note, to minimize the time taken by data preprocessing, we disable integrity
        # verification and compression. The data is just aggregated into an uncompressed .npz file for each source.
        self._logger.compress_logs(
            remove_sources=True, memory_mapping=False, verbose=True, compress=False, verify_integrity=False
        )

        # Parses behavioral data from the compressed logs and uses it to generate the behavioral_dataset.parquet file.
        # Also, extracts camera frame timestamps for each camera and saves them as a separate .parquet file to optimize
        # further camera frame processing.
        self._process_log_data()

        # Pulls the frames and motion estimation data from the ScanImagePC into the local data directory.
        self._session_data.pull_mesoscope_data()

        # Preprocesses the pulled mesoscope data.
        self._session_data.process_mesoscope_data()

        # Renames the video files generated during runtime to use human-friendly camera names, rather than ID-codes.
        os.renames(
            old=self._session_data.camera_frames_path.joinpath("051.mp4"),
            new=self._session_data.camera_frames_path.joinpath("face_camera.mp4"),
        )
        os.renames(
            old=self._session_data.camera_frames_path.joinpath("062.mp4"),
            new=self._session_data.camera_frames_path.joinpath("left_camera.mp4"),
        )
        os.renames(
            old=self._session_data.camera_frames_path.joinpath("073.mp4"),
            new=self._session_data.camera_frames_path.joinpath("right_camera.mp4"),
        )

        # Pushes the processed data to the NAS and BioHPC server.
        self._session_data.push_data()

        message = "Data preprocessing: complete. MesoscopeExperiment runtime: terminated."
        console.echo(message=message, level=LogLevel.SUCCESS)

    def vr_rest(self) -> None:
        """Switches the VR system to the rest state.

        In the rest state, the break is engaged to prevent the mouse from moving the wheel. The encoder module is
        disabled, and instead the torque sensor is enabled. The VR screens are switched off, cutting off light emission.
        By default, the VR system starts all experimental runtimes using the REST state.
        """

        # Ensures VR screens are turned OFF
        self._microcontrollers.disable_vr_screens()

        # Engages the break to prevent the mouse from moving the wheel
        self._microcontrollers.enable_break()

        # Temporarily suspends encoder monitoring. Since the wheel is locked, the mouse should not be able to produce
        # meaningful motion data.
        self._microcontrollers.disable_encoder_monitoring()

        # Initiates torque monitoring.The torque can only be accurately measured when the wheel is locked, as it
        # requires a resistance force to trigger the sensor.
        self._microcontrollers.enable_torque_monitoring()

        # Configures the state tracker to reflect the REST state
        self._change_vr_state(1)

    def vr_run(self) -> None:
        """Switches the VR system to the run state.

        In the run state, the break is disengaged to allow the mouse to freely move the wheel. The encoder module is
        enabled to record and share live running data with Unity, and the torque sensor is disabled. The VR screens are
        switched on to render the VR environment.
        """
        # Initializes encoder monitoring.
        self._microcontrollers.enable_encoder_monitoring()

        # Disables torque monitoring. To accurately measure torque, the sensor requires a resistance force provided by
        # the break. During running, measuring torque is not very reliable and adds little value compared to the
        # encoder.
        self._microcontrollers.disable_torque_monitoring()

        # Toggles the state of the VR screens ON.
        self._microcontrollers.enable_vr_screens()

        # Disengages the break to allow the mouse to move the wheel
        self._microcontrollers.disable_break()

        # Configures the state tracker to reflect RUN state
        self._change_vr_state(2)

    def _get_cue_sequence(self) -> NDArray[np.uint8]:
        """Requests Unity game engine to transmit the sequence of virtual reality track wall cues for the current task.

        This method is used as part of the experimental runtime startup process to both get the sequence of cues and
        verify that the Unity game engine is running and configured correctly.

        Returns:
            The NumPy array that stores the sequence of virtual reality segments as byte (uint8) values.

        Raises:
            RuntimeError: If no response from Unity is received within 2 seconds or if Unity sends a message to an
                unexpected (different) topic other than "CueSequence/" while this method is running.
        """
        # Initializes a second-precise timer to ensure the request is fulfilled within a 2-second timeout
        timeout_timer = PrecisionTimer("s")

        # Sends a request for the task cue (corridor) sequence to Unity GIMBL package.
        self._unity.send_data(topic="CueSequenceTrigger/")

        # Waits at most 2 seconds to receive the response
        timeout_timer.reset()
        while timeout_timer.elapsed < 2:
            # If Unity responds with the cue sequence message, attempts to parse the message
            if self._unity.has_data:
                topic: str
                payload: bytes
                topic, payload = self._unity.get_data()
                if topic == "CueSequence/":
                    # Extracts the sequence of cues that will be used during task runtime.
                    sequence: NDArray[np.uint8] = np.frombuffer(buffer=payload, dtype=np.uint8)
                    return sequence

                else:
                    # If the topic is not "CueSequence/", aborts with an error
                    message = (
                        f"Received an unexpected topic {topic} while waiting for Unity to respond to the cue sequence "
                        f"request. Make sure the Unity is not configured to send data to other topics monitored by the "
                        f"MesoscopeExperiment instance until the cue sequence is resolved as part of the start() "
                        f"method runtime."
                    )
                    console.error(message=message, error=RuntimeError)

        # If the loop above is escaped, this is due to not receiving any message from Unity. Raises an error.
        message = (
            f"The MesoscopeExperiment has requested the task Cue Sequence by sending the trigger to the "
            f"'CueSequenceTrigger/' topic and received no response for 2 seconds. It is likely that the Unity game "
            f"engine is not running or is not configured to work with MesoscopeExperiment."
        )
        console.error(message=message, error=RuntimeError)

        # This backup statement should not be reached, it is here to appease mypy
        raise RuntimeError(message)  # pragma: no cover

    def _start_mesoscope(self) -> None:
        """Sends the frame acquisition start TTL pulse to the mesoscope and waits for the frame acquisition to begin.

        This method is used internally to start the mesoscope frame acquisition as part of the experiment startup
        process. It is also used to verify that the mesoscope is available and properly configured to acquire frames
        based on the input triggers.

        Raises:
            RuntimeError: If the mesoscope does not confirm frame acquisition within 2 seconds after the
                acquisition trigger is sent.
        """

        # Initializes a second-precise timer to ensure the request is fulfilled within a 2-second timeout
        timeout_timer = PrecisionTimer("s")

        # Instructs the mesoscope to begin acquiring frames
        self._microcontrollers.start_mesoscope()

        # Waits at most 2 seconds for the mesoscope to begin sending frame acquisition timestamps to the PC
        timeout_timer.reset()
        while timeout_timer.elapsed < 2:
            # If the mesoscope starts scanning a frame, the method has successfully started the mesoscope frame
            # acquisition.
            if self._microcontrollers.frame_acquisition_status:
                return

        # If the loop above is escaped, this is due to not receiving the mesoscope frame acquisition pulses.
        message = (
            f"The MesoscopeExperiment has requested the mesoscope to start acquiring frames and received no frame "
            f"acquisition trigger for 2 seconds. It is likely that the mesoscope has not been armed for frame "
            f"acquisition or that the mesoscope trigger or frame timestamp connection is not functional."
        )
        console.error(message=message, error=RuntimeError)

        # This code is here to appease mypy. It should not be reachable
        raise RuntimeError(message)  # pragma: no cover

    def _change_vr_state(self, new_state: int) -> None:
        """Updates and logs the new VR state.

        This method is used internally to timestamp and log VR state (stage) changes, such as transitioning between
        rest and run VR states.

        Args:
            new_state: The byte-code for the newly activated VR state.
        """
        self._vr_state = new_state  # Updates the VR state

        # Notifies the user about the new VR state.
        message = f"VR State: {self._state_map[self._vr_state]}."
        console.echo(message=message, level=LogLevel.INFO)

        # Logs the VR state update. Uses header-code 1 to indicate that the logged value is the VR state-code.
        log_package = LogPackage(
            source_id=self._source_id,
            time_stamp=np.uint64(self._timestamp_timer.elapsed),
            serialized_data=np.array([1, new_state], dtype=np.uint8),
        )
        self._logger.input_queue.put(log_package)

    def change_experiment_state(self, new_state: int) -> None:
        """Updates and logs the new experiment state.

        Use this method to timestamp and log experiment state (stage) changes, such as transitioning between different
        task goals.

        Args:
            new_state: The integer byte-code for the new experiment state. The code will be serialized as an uint8
                value, so only values between 0 and 255 inclusive are supported.
        """
        self._experiment_state = new_state  # Updates the tracked experiment state value

        # Notifies the user about the new Experiment state.
        message = f"Experiment State: {new_state}."
        console.echo(message=message, level=LogLevel.INFO)

        # Logs the VR state update. Uses header-code 2 to indicate that the logged value is the experiment state-code.
        log_package = LogPackage(
            source_id=self._source_id,
            time_stamp=np.uint64(self._timestamp_timer.elapsed),
            serialized_data=np.array([2, new_state], dtype=np.uint8),
        )
        self._logger.input_queue.put(log_package)

    def _extract_logged_data(self) -> dict[int, Any]:
        """Extracts the VR states, Experiment states, and the Virtual Reality cue sequence from the log generated
        by the instance during runtime.

        This method reads the compressed '.npz' archives generated by the MesoscopeExperiment instance during runtime
        and parses all logged data. Unlike extraction methods used by hardware modules and VideoSystem instances, this
        method is not designed to be called by the end-user. Instead, it is called internally as part of the stop()
        method runtime to generate the unified behavioral dataset.

        Returns:
            A dictionary that uses id-codes associated with each type of log entries as keys and stores two-element
            tuples for all keys other than 0. The first element in each tuple is an array of timestamps, where each
            timestamp is a 64-bit unsigned numpy integer and specifies the number of microseconds since the UTC epoch
            onset. The second element of each tuple is an array of 8-bit unsigned numpy integer data-points. Key 0
            stores a byte numpy array that communicates the sequence of wall cues encountered by the animal as it was
            performing the experiment.
        """

        # Generates the log file path using the instance source ID and the data logger instance.
        log_path = self._logger.output_directory.joinpath(f"{self._source_id}_log.npz")

        # Loads the archive into RAM
        archive: NpzFile = np.load(file=log_path)

        # Precreates the dictionary to store the extracted data and temporary lists to store VR and Experiment states
        output_dict: dict[int, Any] = {}
        vr_states = []
        vr_timestamps = []
        experiment_states = []
        experiment_timestamps = []

        # Locates the logging onset timestamp. The onset is used to convert the timestamps for logged data into absolute
        # UTC timestamps. Originally, all timestamps other than onset are stored as elapsed time in microseconds
        # relative to the onset timestamp.
        timestamp_offset = 0
        onset_us = np.uint64(0)
        timestamp: np.uint64
        for number, item in enumerate(archive.files):
            message: NDArray[np.uint8] = archive[item]  # Extracts message payload from the compressed .npy file

            # Recovers the uint64 timestamp value from each message. The timestamp occupies 8 bytes of each logged
            # message starting at index 1. If timestamp value is 0, the message contains the onset timestamp value
            # stored as 8-byte payload. Index 0 stores the source ID (uint8 value)
            if np.uint64(message[1:9].view(np.uint64)[0]) == 0:
                # Extracts the byte-serialized UTC timestamp stored as microseconds since epoch onset.
                onset_us = np.uint64(message[9:].view("<i8")[0].copy())

                # Breaks the loop onc the onset is found. Generally, the onset is expected to be found very early into
                # the loop
                timestamp_offset = number  # Records the item number at which the onset value was found.
                break

        # Once the onset has been discovered, loops over all remaining messages and extracts data stored in these
        # messages.
        for item in archive.files[timestamp_offset + 1 :]:
            message = archive[item]

            # Extracts the elapsed microseconds since timestamp and uses it to calculate the global timestamp for the
            # message, in microseconds since epoch onset.
            elapsed_microseconds = np.uint64(message[1:9].view(np.uint64)[0].copy())
            timestamp = onset_us + elapsed_microseconds

            payload = message[9:]  # Extracts the payload from the message

            # If the message is longer than 10,000 bytes, it is a sequence of wall cues. It is very unlikely that we
            # will log any other data with this length, so it is a safe heuristic to use.
            if len(payload) > 10000:
                cue_sequence: NDArray[np.uint8] = payload.view(np.uint8).copy()  # Keeps the original numpy uint8 format
                output_dict[0] = cue_sequence  # Packages the cue sequence into the dictionary

            # If the message has a length of 2 bytes and the first element is 1, the message communicates the VR state
            # code.
            elif len(payload) == 2 and payload[0] == 1:
                vr_state = np.uint8(payload[2])  # Extracts the VR state code from the second byte of the message.
                vr_states.append(vr_state)
                vr_timestamps.append(timestamp)

            # Otherwise, if the starting code is 2, the message communicates the experiment state code.
            elif len(payload) == 2 and payload[0] == 2:
                # Extracts the experiment state code from the second byte of the message.
                experiment_state = np.uint8(payload[2])
                experiment_states.append(experiment_state)
                experiment_timestamps.append(timestamp)

        # Closes the archive to free up memory
        archive.close()

        # Converts lists to arrays and packages them into the dictionary as a 2-element tuple
        output_dict[1] = (np.array(vr_timestamps, dtype=np.uint64), np.array(vr_states, dtype=np.uint8))
        output_dict[2] = (np.array(experiment_timestamps, dtype=np.uint64), np.array(experiment_states, dtype=np.uint8))

        # Returns the extracted data dictionary
        return output_dict

    def _process_log_data(self) -> None:
        """Extracts the data logged during runtime from .npz archive files and uses it to generate the
        behavioral_dataset.parquet file and the camera_timestamps.parquet file.

        This method is called during the stop() method runtime to extract, align, and output the initial behavioral
        dataset. All data processed as part of our cell registration and video registration pipelines will later be
        aligned to and appended to this dataset.
        """
        # First extracts the timestamps for the mesoscope frames. These timestamps are used as seeds to which all other
        # data sources are aligned during preprocessing and post-processing of the data.
        seeds = self._microcontrollers.get_mesoscope_frame_data()

        # Iteratively goes over all hardware modules and extracts the data recorded by each module during runtime.
        # Uses discrete or continuous interpolation to align the data to the seed timestamps:
        # ENCODER
        timestamps, data = self._microcontrollers.get_encoder_data()
        encoder_data = interpolate_data(timestamps=timestamps, data=data, seed_timestamps=seeds, is_discrete=False)

        # TORQUE
        timestamps, data = self._microcontrollers.get_torque_data()
        torque_data = interpolate_data(timestamps=timestamps, data=data, seed_timestamps=seeds, is_discrete=False)

        # LICKS
        timestamps, data = self._microcontrollers.get_lick_data()
        lick_data = interpolate_data(timestamps=timestamps, data=data, seed_timestamps=seeds, is_discrete=True)

        # WATER REWARD
        timestamps, data = self._microcontrollers.get_valve_data()
        reward_data = interpolate_data(timestamps=timestamps, data=data, seed_timestamps=seeds, is_discrete=True)

        # SCREENS
        timestamps, data = self._microcontrollers.get_screen_data()
        screen_data = interpolate_data(timestamps=timestamps, data=data, seed_timestamps=seeds, is_discrete=True)

        # BREAK
        timestamps, data = self._microcontrollers.get_break_data()
        break_data = interpolate_data(timestamps=timestamps, data=data, seed_timestamps=seeds, is_discrete=True)

        # Extracts the VR states, Experiment states, and Virtual Reality queue sequence logged by the main process.
        extracted_data = self._extract_logged_data()

        # Processes VR states and Experiment states similar to hardware modules data. They are both discrete states:
        # VR STATE
        timestamps, data = extracted_data[1]
        vr_states = interpolate_data(timestamps=timestamps, data=data, seed_timestamps=seeds, is_discrete=True)

        # EXPERIMENT STATE
        timestamps, data = extracted_data[2]
        experiment_states = interpolate_data(timestamps=timestamps, data=data, seed_timestamps=seeds, is_discrete=True)

        # CUE SEQUENCE
        # Unlike other extracted data, we do not have the time points for each cue in the sequence, as this data is
        # pre-generated when the Unity task is initialized. However, since we know the distance associated with each
        # cue, we can use the cumulative traveled distance extracted from the encoder to know the cue experienced by the
        # mouse at each time-point.
        cue_sequence: NDArray[np.uint8] = extracted_data[0]

        # Uses the dictionary provided at class initialization to compute the length of each cue in the sequence. Then
        # computes the cumulative distance the animal needs to travel to reach each cue in the sequence. The first cue
        # is associated with distance of 0 (the animal starts at this cue), the distance to each following cue is the
        # sum of all previous cue lengths
        distance_sequence = np.zeros(len(cue_sequence), dtype=np.float64)
        distance_sequence[1:] = np.cumsum([self._cue_map[int(code)] for code in cue_sequence[:-1]])

        # Finds indices of cues for each encoder distance point, ensuring 0 maps to first cue (index 0). This gives us
        # the cues the animal experiences at each seed timestamp (insofar as each cue is tied to the cumulative distance
        # covered by the animal at each timestamp).
        indices = np.maximum(0, np.searchsorted(distance_sequence, encoder_data, side="right") - 1)
        cue_data = cue_sequence[indices]  # Extracts the cue experienced by the mouse at each seed timestamp.

        # Assembles the aligned behavioral dataset from the data processed above:
        # Define the schema with proper types and units. The schema determines the datatypes and column names used by
        # the dataset
        schema = {
            "time_us": pl.UInt64,
            "distance_cm": pl.Float64,
            "torque_N_cm": pl.Float64,
            "lick_on": pl.UInt8,
            "dispensed_water_μL": pl.Float64,
            "screens_on": pl.UInt8,
            "break_on": pl.UInt8,
            "vr_state": pl.UInt8,
            "experiment_state": pl.UInt8,
            "cue": pl.UInt8,
        }

        # Creates a mapping of our data arrays to schema columns
        data_mapping = {
            "time_us": seeds,
            "distance_cm": encoder_data,
            "torque_N_cm": torque_data,
            "lick_on": lick_data,
            "dispensed_water_μL": reward_data,
            "screens_on": screen_data,
            "break_on": break_data,
            "vr_state": vr_states,
            "experiment_state": experiment_states,
            "cue": cue_data,
        }

        # Creates Polars dataframe with schema
        dataframe = pl.DataFrame(data_mapping, schema=schema)

        # Saves the dataset as a lz4 compressed parquet file
        dataframe.write_parquet(
            file=self._session_data.behavioral_data_path,
            compression="lz4",  # LZ4 is used for compression / decompression speed
            use_pyarrow=True,
            statistics=True,
        )

        # Camera timestamp processing is handled by a dedicated binding class method
        self._cameras.process_log_data(output_file=self._session_data.camera_timestamps_path)


class _BehavioralTraining:
    """The base class for all behavioral training runtimes.

    This class provides methods for running the lick and run training sessions using a subset of the Mesoscope-VR
    system. It abstracts most low-level interactions with the VR system and the mesoscope via a simple
    high-level state API.

    This class also provides methods for limited preprocessing of the collected data. The preprocessing methods are
    designed to be executed after each training runtime to prepare the data for long-term storage and transmission
    over the network. Preprocessing methods use multiprocessing to optimize runtime performance and assume that the
    VRPC is kept mostly idle during data preprocessing.

    Notes:
        Calling this initializer does not start the Mesoscope-VR components. Use the start() method before issuing other
        commands to properly initialize all remote processes. This class reserves up to 11 CPU cores during runtime.

        Use the 'axtl-ports' cli command to discover the USB ports used by Ataraxis Micro Controller (AMC) devices.

        Use the 'axvs-ids' cli command to discover the camera indices used by the Harvesters-managed and
        OpenCV-managed cameras.

        Use the 'sl-devices' cli command to discover the serial ports used by the Zaber motion controllers.

    Args:
        session_data: An initialized SessionData instance. This instance is used to transfer the data between VRPC,
            BioHPC server, and the NAS during runtime. Each instance is initialized for the specific project, animal,
            and session combination for which the data is acquired.
        descriptor: A partially configured _LickTrainingDescriptor or _RunTrainingDescriptor instance. This instance is
            used to store session-specific information in a human-readable format.
        screens_on: Communicates whether the VR screens are currently ON at the time of class initialization.
        actor_port: The USB port used by the actor Microcontroller.
        sensor_port: The USB port used by the sensor Microcontroller.
        encoder_port: The USB port used by the encoder Microcontroller.
        headbar_port: The USB port used by the headbar Zaber motor controllers (devices).
        lickport_port: The USB port used by the lickport Zaber motor controllers (devices).
        valve_calibration_data: A tuple of tuples, with each inner tuple storing a pair of values. The first value is
            the duration, in microseconds, the valve was open. The second value is the volume of dispensed water, in
            microliters.
        face_camera_index: The index of the face camera in the list of all available Harvester-managed cameras.
        left_camera_index: The index of the left camera in the list of all available OpenCV-managed cameras.
        right_camera_index: The index of the right camera in the list of all available OpenCV-managed cameras.
        harvesters_cti_path: The path to the GeniCam CTI file used to connect to Harvesters-managed cameras.

    Attributes:
        _started: Tracks whether the VR system and training runtime are currently running.
        _lick_training: Tracks the training state used by the instance, which is required for log parsing.
        _descriptor: Stores the session descriptor instance.
        _logger: A DataLogger instance that collects behavior log data from all sources: microcontrollers and video
            cameras.
        _microcontrollers: Stores the _MicroControllerInterfaces instance that interfaces with all MicroController
            devices used during runtime.
        _cameras: Stores the _VideoSystems instance that interfaces with video systems (cameras) used during
            runtime.
        _headbar: Stores the _HeadBar class instance that interfaces with all HeadBar manipulator motors.
        _lickport: Stores the _LickPort class instance that interfaces with all LickPort manipulator motors.
        _screen_on: Tracks whether the VR displays are currently ON.
        _session_data: Stores the SessionData instance used to manage the acquired data.

    Raises:
        TypeError: If session_data argument has an invalid type.
    """

    def __init__(
        self,
        session_data: SessionData,
        descriptor: _LickTrainingDescriptor | _RunTrainingDescriptor,
        screens_on: bool = False,
        actor_port: str = "/dev/ttyACM0",
        sensor_port: str = "/dev/ttyACM1",
        encoder_port: str = "/dev/ttyACM2",
        headbar_port: str = "/dev/ttyUSB0",
        lickport_port: str = "/dev/ttyUSB1",
        valve_calibration_data: tuple[tuple[int | float, int | float], ...] = (
            (15000, 1.8556),
            (30000, 3.4844),
            (45000, 7.1846),
            (60000, 10.0854),
        ),
        face_camera_index: int = 0,
        left_camera_index: int = 0,
        right_camera_index: int = 2,
        harvesters_cti_path: Path = Path("/opt/mvIMPACT_Acquire/lib/x86_64/mvGenTLProducer.cti"),
    ) -> None:
        # Activates the console to display messages to the user if the console is disabled when the class is
        # instantiated.
        if not console.enabled:
            console.enable()

        # Creates the _started flag first to avoid leaks if the initialization method fails.
        self._started: bool = False

        # Determines the type of training carried out by the instance. This is needed for log parsing and
        # SessionDescriptor generation.
        self._lick_training: bool = False
        self._descriptor: _LickTrainingDescriptor | _RunTrainingDescriptor = descriptor

        # Input verification:
        if not isinstance(session_data, SessionData):
            message = (
                f"Unable to initialize the BehavioralTraining class. Expected a SessionData instance for "
                f"'session_data' argument, but instead encountered {session_data} of type "
                f"{type(session_data).__name__}."
            )
            console.error(message=message, error=TypeError)

        # Defines other flags used during runtime:
        self._screen_on: bool = screens_on  # Usually this would be false, but this is not guaranteed

        # Saves the SessionData instance to class attribute so that it can be used from class methods. Since SessionData
        # resolves session directory structure at initialization, the instance is ready to resolve all paths used by
        # the training class instance.
        self._session_data: SessionData = session_data

        # Initializes the DataLogger instance used to log data from all microcontrollers, camera frame savers, and this
        # class instance.
        self._logger: DataLogger = DataLogger(
            output_directory=session_data.raw_data_path,
            instance_name="behavior",  # Creates behavior_log subfolder under raw_data
            sleep_timer=0,
            exist_ok=True,
            process_count=1,
            thread_count=10,
        )

        # Initializes the binding class for all MicroController Interfaces.
        self._microcontrollers: _MicroControllerInterfaces = _MicroControllerInterfaces(
            data_logger=self._logger,
            actor_port=actor_port,
            sensor_port=sensor_port,
            encoder_port=encoder_port,
            valve_calibration_data=valve_calibration_data,
            debug=False,
        )

        # Initializes the binding class for all VideoSystems.
        self._cameras: _VideoSystems = _VideoSystems(
            data_logger=self._logger,
            output_directory=self._session_data.camera_frames_path,
            face_camera_index=face_camera_index,
            left_camera_index=left_camera_index,
            right_camera_index=right_camera_index,
            harvesters_cti_path=harvesters_cti_path,
        )

        # While we can connect to ports managed by ZaberLauncher, ZaberLauncher cannot connect to ports managed via
        # software. Therefore, we have to make sure ZaberLauncher is running before connecting to motors.
        message = (
            "Preparing to connect to HeadBar and LickPort Zaber controllers. Make sure that ZaberLauncher app is "
            "running before proceeding further. If ZaberLauncher is not running, youi WILL NOT be able to manually "
            "control the HeadBar and LickPort motor positions until you reset the runtime."
        )
        console.echo(message=message, level=LogLevel.WARNING)
        while input("Enter 'y' to continue: ") != "y":
            continue

        # Initializes the binding classes for the HeadBar and LickPort manipulator motors.
        self._headbar: _HeadBar = _HeadBar(
            headbar_port=headbar_port, zaber_positions_path=self._session_data.previous_zaber_positions_path
        )
        self._lickport: _LickPort = _LickPort(
            lickport_port=lickport_port, zaber_positions_path=self._session_data.previous_zaber_positions_path
        )

    def start(self) -> None:
        """Sets up all assets used during the training.

        This internal method establishes the communication with the microcontrollers, data logger cores, and video
        system processes.

        Notes:
            This method will not run unless the host PC has access to the necessary number of logical CPU cores
            and other required hardware resources (GPU for video encoding, etc.). This prevents using the class on
            machines that are unlikely to sustain the runtime requirements.

            As part of its runtime, this method will attempt to set Zaber motors for the HeadBar and LickPort to the
            positions that would typically be used during the mesoscope experiment runtime. Exercise caution and always
            monitor the system when it is running this method, as unexpected motor behavior can damage the mesoscope or
            harm the animal.

            Unlike the experiment class start(), this method does not preset the hardware module states during runtime.
            Call the desired training state method to configure the hardware modules appropriately for the chosen
            runtime mode.

        Raises:
            RuntimeError: If the host PC does not have enough logical CPU cores available.
        """
        # Prevents (re) starting an already started VR process.
        if self._started:
            return

        # 3 cores for microcontrollers, 1 core for the data logger, 6 cores for the current video_system
        # configuration (3 producers, 3 consumer), 1 core for the central process calling this method. 11 cores
        # total.
        if not os.cpu_count() >= 11:
            message = (
                f"Unable to start the BehavioralTraining runtime. The host PC must have at least 11 logical CPU "
                f"cores available for this class to work as expected, but only {os.cpu_count()} cores are "
                f"available."
            )
            console.error(message=message, error=RuntimeError)

        message = "Initializing BehavioralTraining assets..."
        console.echo(message=message, level=LogLevel.INFO)

        # Starts the data logger
        self._logger.start()
        message = "DataLogger: Started."
        console.echo(message=message, level=LogLevel.SUCCESS)

        # Starts the face camera. This starts frame acquisition and displays acquired frames to the user. However,
        # frame saving is disabled at this time. Body cameras are also disabled. This is intentional, as at this point
        # we want to minimize the number of active processes. This is helpful if this method is called while the
        # previous session is still running its data preprocessing pipeline and needs as many free cores as possible.
        self._cameras.start_face_camera()

        # Initializes the Zaber positioning sequence. This relies heavily on user feedback to confirm that it is safe to
        # proceed with motor movements.
        message = (
            "Preparing to move HeadBar into position. Swivel out the VR screens, and make sure the animal is NOT "
            "mounted on the rig. Failure to fulfill these steps may HARM the animal."
        )
        console.echo(message=message, level=LogLevel.WARNING)
        while input("Enter 'y' to continue: ") != "y":
            continue

        # Homes all motors in-parallel. The homing trajectories for the motors as they are used now should not intersect
        # with each other, so it is safe to move both assemblies at the same time.
        self._headbar.prepare_motors(wait_until_idle=False)
        self._lickport.prepare_motors(wait_until_idle=True)
        self._headbar.wait_until_idle()

        # Sets the motors into the mounting position. The HeadBar is either restored to the previous session position or
        # is set to the default mounting position stored in non-volatile memory. The LickPort is moved to a position
        # optimized for putting the animal on the VR rig.
        self._headbar.restore_position(wait_until_idle=False)
        self._lickport.mount_position(wait_until_idle=True)
        self._headbar.wait_until_idle()

        # Gives user time to mount the animal and requires confirmation before proceeding further.
        message = (
            "Preparing to move the LickPort into position. Mount the animal onto the VR rig. If necessary, adjust the "
            "HeadBar position to make sure the animal can comfortably run the task."
        )
        console.echo(message=message, level=LogLevel.WARNING)
        while input("Enter 'y' to continue: ") != "y":
            continue

        # Restores the lickPort to the previous session's position or to the default parking position. This positions
        # the LickPort in a way that is easily accessible by the animal.
        self._lickport.restore_position()

        message = (
            "If necessary, adjust LickPort position to be easily reachable by the animal. Take extra care when moving "
            "the LickPort towards the animal! This is the last manual checkpoint, entering 'y' after this message will "
            "begin the training."
        )
        console.echo(message=message, level=LogLevel.WARNING)
        while input("Enter 'y' to continue: ") != "y":
            continue

        # Generates a snapshot of all zaber positions. This serves as an early checkpoint in case the runtime has to be
        # aborted in a non-graceful way (without running the stop() sequence). This way, next runtime will restart with
        # the calibrated zaber positions.
        head_bar_positions = self._headbar.get_positions()
        lickport_positions = self._lickport.get_positions()
        zaber_positions = _ZaberPositions(
            headbar_z=head_bar_positions[0],
            headbar_pitch=head_bar_positions[1],
            headbar_roll=head_bar_positions[2],
            lickport_z=lickport_positions[0],
            lickport_x=lickport_positions[1],
            lickport_y=lickport_positions[2],
        )
        zaber_positions.to_yaml(file_path=self._session_data.zaber_positions_path)
        message = "HeadBar and LickPort positions: Saved."
        console.echo(message=message, level=LogLevel.SUCCESS)

        # Enables body cameras. Starts frame saving for all cameras
        self._cameras.start_body_cameras()
        self._cameras.save_face_camera_frames()
        self._cameras.save_body_camera_frames()

        # Initializes communication with the microcontrollers
        self._microcontrollers.start()

        # The setup procedure is complete.
        self._started = True

        message = "BehavioralTraining assets: Initialized."
        console.echo(message=message, level=LogLevel.SUCCESS)

    def stop(self) -> None:
        """Stops and terminates the BehavioralTraining runtime.

        This method achieves two main purposes. First, releases the hardware resources used during the training runtime
        by various system components. Second, it runs the preprocessing pipeline on the data to prepare it for long-term
        storage and further processing.
        """

        # Prevents stopping an already stopped VR process.
        if not self._started:
            return

        message = "Terminating BehavioralTraining runtime..."
        console.echo(message=message, level=LogLevel.INFO)

        # Resets the _started tracker
        self._started = False

        # Stops all cameras.
        self._cameras.stop()

        # Manually stops all hardware modules before shutting down the microcontrollers
        self._microcontrollers.enable_break()
        self._microcontrollers.disable_lick_monitoring()
        self._microcontrollers.disable_torque_monitoring()
        self._microcontrollers.disable_encoder_monitoring()

        # Stops all microcontroller interfaces
        self._microcontrollers.stop()

        # Stops the data logger instance
        self._logger.stop()

        message = "Data Logger: stopped."
        console.echo(message=message, level=LogLevel.SUCCESS)

        # Updates the contents of the pregenerated descriptor file and dumps it as a .yaml into the root raw_data
        # session directory. This needs to be done after the microcontrollers and loggers have been stopped to ensure
        # that the reported dispensed_water_volume_ul is accurate.
        delivered_water = self._microcontrollers.total_delivered_volume
        if self._lick_training:
            # Overwrites the delivered water volume with the volume recorded over the runtime.
            self._descriptor.dispensed_water_volume_ul = delivered_water
            self._descriptor.to_yaml(file_path=self._session_data.session_descriptor_path)

        # Generates the snapshot of the current HeadBar and LickPort positions and saves them as a .yaml file. This has
        # to be done before Zaber motors are reset back to parking position.
        head_bar_positions = self._headbar.get_positions()
        lickport_positions = self._lickport.get_positions()
        zaber_positions = _ZaberPositions(
            headbar_z=head_bar_positions[0],
            headbar_pitch=head_bar_positions[1],
            headbar_roll=head_bar_positions[2],
            lickport_z=lickport_positions[0],
            lickport_x=lickport_positions[1],
            lickport_y=lickport_positions[2],
        )
        zaber_positions.to_yaml(file_path=self._session_data.zaber_positions_path)

        # Moves the LickPort to the mounting position to assist removing the animal from the rig.
        self._lickport.mount_position()

        # Notifies the user about the volume of water dispensed during runtime, so that they can ensure the mouse
        # get any leftover daily water limit.
        message = (
            f"During runtime, the system dispensed ~{delivered_water} uL of water to the animal. "
            f"If the animal is on water restriction, make sure it receives any additional water, if the dispensed "
            f"volume does not cover the daily water limit for that animal."
        )
        console.echo(message=message, level=LogLevel.INFO)

        # Instructs the user to remove the objective and the animal before resetting all zaber motors.
        message = (
            "Preparing to reset the HeadBar and LickPort motors. Remove the animal from the VR rig and swivel the VR "
            "screens out. Failure to do so may HARM the animal."
        )
        console.echo(message=message, level=LogLevel.WARNING)
        while input("Enter 'y' to continue: ") != "y":
            continue

        # Parks both controllers and then disconnects from their Connection classes. Note, the parking is performed
        # in-parallel
        self._headbar.park_position(wait_until_idle=False)
        self._lickport.park_position(wait_until_idle=True)
        self._headbar.wait_until_idle()
        self._headbar.disconnect()
        self._lickport.disconnect()

        message = "HeadBar and LickPort motors: Reset."
        console.echo(message=message, level=LogLevel.SUCCESS)

        # Prompts the user to add their notes to the appropriate section of the descriptor file. This has to be done
        # before processing so that the notes are properly transferred to the NAS and server. Also, this makes it more
        # obvious to the user when it is safe to start preparing for the next session and leave the current one
        # processing the data.
        message = (
            f"Data acquisition: Complete. Open the session descriptor file stored in session's raw_data folder and "
            f"update the notes session with the notes taken during runtime. This is the last manual checkpoint, "
            f"entering 'y' after this message will begin data preprocessing and, after that, transmit it to the BioHPC "
            f"server and the NAS storage. It is safe to start preparing for the next session after hitting 'y'."
        )
        console.echo(message=message, level=LogLevel.WARNING)
        while input("Enter 'y' to continue: ") != "y":
            continue

        message = "Initializing data preprocessing..."
        console.echo(message=message, level=LogLevel.INFO)

        # Compresses all logs into a single .npz file. This is done both for long-term storage optimization and to
        # allow parsing the data. Note, to minimize the time taken by data preprocessing, we disable integrity
        # verification and compression. The data is just aggregated into an uncompressed .npz file for each source.
        self._logger.compress_logs(
            remove_sources=True, memory_mapping=False, verbose=True, compress=False, verify_integrity=False
        )

        # Parses behavioral data from the compressed logs and uses it to generate the behavioral_dataset.parquet file.
        # Also, extracts camera frame timestamps for each camera and saves them as a separate .parquet file to optimize
        # further camera frame processing.
        self._process_log_data()

        # Renames the video files generated during runtime to use human-friendly camera names, rather than ID-codes.
        os.renames(
            old=self._session_data.camera_frames_path.joinpath("051.mp4"),
            new=self._session_data.camera_frames_path.joinpath("face_camera.mp4"),
        )
        os.renames(
            old=self._session_data.camera_frames_path.joinpath("062.mp4"),
            new=self._session_data.camera_frames_path.joinpath("left_camera.mp4"),
        )
        os.renames(
            old=self._session_data.camera_frames_path.joinpath("073.mp4"),
            new=self._session_data.camera_frames_path.joinpath("right_camera.mp4"),
        )

        # Pushes the processed data to the NAS and BioHPC server.
        self._session_data.push_data()

        message = "Data preprocessing: complete. BehavioralTraining runtime: terminated."
        console.echo(message=message, level=LogLevel.SUCCESS)

    def lick_train_state(self) -> None:
        """Configures the VR system for running the lick training.

        In the rest state, the break is engaged to prevent the mouse from moving the wheel. The encoder module is
        disabled, and the torque sensor is enabled. The VR screens are switched off, cutting off light emission.
        The lick sensor monitoring is on to record animal licking data.
        """

        # Ensures VR screens are turned OFF
        self._microcontrollers.disable_vr_screens()

        # Engages the break to prevent the mouse from moving the wheel
        self._microcontrollers.enable_break()

        # Ensures that encoder monitoring is disabled
        self._microcontrollers.disable_encoder_monitoring()

        # Initiates torque monitoring
        self._microcontrollers.enable_torque_monitoring()

        # Initiates lick monitoring
        self._microcontrollers.enable_lick_monitoring()

        # Sets the tracker
        self._lick_training = True

    def run_train_state(self) -> None:
        """Configures the VR system for running the run training.

        In the rest state, the break is disengaged, allowing the mouse to run on the wheel. The encoder module is
        enabled, and the torque sensor is disabled. The VR screens are switched off, cutting off light emission.
        The lick sensor monitoring is on to record animal licking data.
        """

        # Ensures VR screens are turned OFF
        self._microcontrollers.disable_vr_screens()

        # Disengages the break, enabling the mouse to run on the wheel
        self._microcontrollers.disable_break()

        # Ensures that encoder monitoring is enabled
        self._microcontrollers.enable_encoder_monitoring()

        # Ensures torque monitoring is disabled
        self._microcontrollers.disable_torque_monitoring()

        # Initiates lick monitoring
        self._microcontrollers.enable_lick_monitoring()

        # Sets the tracker
        self._lick_training = False

    def deliver_reward(self, reward_size: float = 5.0) -> None:
        """Uses the solenoid valve to deliver the requested volume of water in microliters.

        This method is used by the training runtimes to reward the animal with water as part of the training process.
        """
        self._microcontrollers.deliver_reward(volume=reward_size)

    @property
    def trackers(self) -> tuple[SharedMemoryArray, SharedMemoryArray, SharedMemoryArray]:
        """Returns the tracker SharedMemoryArrays for (in this order) the LickInterface, ValveInterface, and
        EncoderInterface.

        These arrays should be passed to the BehaviorVisualizer class to monitor the lick, valve and running speed data
        in real time during training.
        """
        return (
            self._microcontrollers.lick_tracker,
            self._microcontrollers.valve_tracker,
            self._microcontrollers.speed_tracker,
        )

    def _process_log_data(self) -> None:
        """Extracts the data logged during runtime from .npz archive files and uses it to generate the
        behavioral_dataset.parquet file and the camera_timestamps.parquet file.

        This method is called during the stop() method runtime to extract, align, and output the initial behavioral
        dataset. While we do not use this data during the main analysis pipeline, it is used to assess animal's
        training performance.
        """
        # Depending on the training type, parses either encoder data or torque data
        if self._lick_training:
            motion_stamps, motion_data = self._microcontrollers.get_torque_data()
        else:
            motion_stamps, motion_data = self._microcontrollers.get_encoder_data()

        # Extracts data from modules used by both training types:
        lick_stamps, lick_data = self._microcontrollers.get_lick_data()
        reward_stamps, reward_data = self._microcontrollers.get_valve_data()

        # Combines all timestamps from all sources. Then interpolates the missing values from each data source for
        # all timestamps. This aligns all collected data to each-other.
        combined_timestamps = np.sort(np.concatenate((motion_stamps, lick_stamps, reward_stamps)))
        motion_data = interpolate_data(
            timestamps=motion_stamps, data=motion_data, seed_timestamps=combined_timestamps, is_discrete=False
        )
        lick_data = interpolate_data(
            timestamps=lick_stamps, data=lick_data, seed_timestamps=combined_timestamps, is_discrete=True
        )
        reward_data = interpolate_data(
            timestamps=reward_stamps, data=reward_data, seed_timestamps=combined_timestamps, is_discrete=True
        )

        # Assembles the aligned behavioral dataset from the data processed above:
        if self._lick_training:
            # Defines the schema with proper types and units. The schema determines the datatypes and column names used
            # by the dataset
            schema = {
                "time_us": pl.UInt64,
                "torque_N_cm": pl.Float64,
                "lick_on": pl.UInt8,
                "dispensed_water_μL": pl.Float64,
            }

            # Creates a mapping of our data arrays to schema columns
            data_mapping = {
                "time_us": combined_timestamps,
                "torque_N_cm": motion_data,
                "lick_on": lick_data,
                "dispensed_water_μL": reward_data,
            }
        else:
            # Defines the schema with proper types and units. The schema determines the datatypes and column names used
            # by the dataset
            schema = {
                "time_us": pl.UInt64,
                "distance_cm": pl.Float64,
                "lick_on": pl.UInt8,
                "dispensed_water_μL": pl.Float64,
            }

            # Creates a mapping of our data arrays to schema columns
            data_mapping = {
                "time_us": combined_timestamps,
                "distance_cm": motion_data,
                "lick_on": lick_data,
                "dispensed_water_μL": reward_data,
            }

        # Creates Polars dataframe with schema
        dataframe = pl.DataFrame(data_mapping, schema=schema)

        # Saves the dataset as a lz4 compressed parquet file
        dataframe.write_parquet(
            file=self._session_data.behavioral_data_path,
            compression="lz4",  # LZ4 is used for compression / decompression speed
            use_pyarrow=True,
            statistics=True,
        )

        # Camera timestamp processing is handled by a dedicated binding class method
        self._cameras.process_log_data(output_file=self._session_data.camera_timestamps_path)


def lick_training_logic(
    runtime: _BehavioralTraining,
    average_reward_delay: int = 12,
    maximum_deviation_from_mean: int = 6,
    maximum_water_volume: float = 1.0,
    maximum_training_time: int = 30,
) -> None:
    """Encapsulates the logic used to train animals how to operate the lick port.

    The lick training consists of delivering randomly spaced 5 uL water rewards via the Valve module to teach the
    animal that water comes out of the lick port. Each reward is delivered at a pseudorandom delay after the previous
    reward or training onset. Reward delay sequence is generated before training runtime by sampling a uniform
    distribution centered at 'average_reward_delay' with lower and upper bounds defined by
    'maximum_deviation_from_mean'. The training continues either until the valve delivers the 'maximum_water_volume' in
    milliliters or until the 'maximum_training_time' in minutes is reached, whichever comes first.

    Args:
        runtime: The initialized _BehavioralTraining instance that manages all Mesoscope-VR components used by this
            training runtime.
        average_reward_delay: The average time, in seconds, that separates two reward deliveries. This is used to
            generate the reward delay sequence as the center of the uniform distribution from which delays are sampled.
        maximum_deviation_from_mean: The maximum deviation from the average reward delay, in seconds. This determines
            the upper and lower boundaries for the data sampled from the uniform distribution centered at the
            average_reward_delay.
        maximum_water_volume: The maximum volume of water, in milliliters, that can be delivered during this runtime.
        maximum_training_time: The maximum time, in minutes, to run the training.

    Notes:
        All delays fall in the range of average_reward_delay +- maximum_deviation_from_mean.

        This function acts on top of the BehavioralTraining class and provides the overriding logic for the lick
        training process. During experiments, runtime logic is handled by Unity game engine, so specialized control
        functions are only required when training the animals without Unity.
    """
    # Initializes the timer used to enforce reward delays
    delay_timer = PrecisionTimer("us")

    # Uses runtime tracker extracted from the runtime instance to initialize the visualizer instance
    lick_tracker, valve_tracker, speed_tracker = runtime.trackers
    visualizer = BehaviorVisualizer(lick_tracker=lick_tracker, valve_tracker=valve_tracker, speed_tracker=speed_tracker)

    message = f"Generating the pseudorandom reward delay sequence..."
    console.echo(message=message, level=LogLevel.INFO)

    # Computes lower and upper boundaries for the reward delay
    lower_bound = average_reward_delay - maximum_deviation_from_mean
    upper_bound = average_reward_delay + maximum_deviation_from_mean

    # Converts maximum volume to uL and divides it by 5 uL (reward size) to get the number of delays to sample from
    # the delay distribution
    num_samples = np.floor((maximum_water_volume * 1000) / 5).astype(np.uint64)

    # Generates samples from a uniform distribution within delay bounds
    samples = np.random.uniform(lower_bound, upper_bound, num_samples)

    # Calculates cumulative training time for each sampled delay. This communicates the total time passed when each
    # reward is delivered to the animal
    cumulative_time = np.cumsum(samples)

    # Finds the maximum number of samples that fits within the maximum training time. This assumes that to consume 1
    # ml of water, the animal would likely need more time than the maximum allowed training time, so we need to slice
    # the sampled delay array to fit within the time boundary.
    max_samples_idx = np.searchsorted(cumulative_time, maximum_training_time * 60, side="right")

    # Slices the samples array to make the total training time be roughly the maximum requested duration. Converts each
    # delay from seconds to microseconds and rounds to the nearest integer. This is done to make delays compatible with
    # PrecisionTimer class.
    reward_delays: NDArray[np.uint64] = np.round(samples[:max_samples_idx] * 1000000, decimals=0).astype(np.uint64)

    message = (
        f"Generated a sequence of {len(reward_delays)} rewards with the total cumulative runtime of "
        f"{np.round(cumulative_time[max_samples_idx - 1] / 60, decimals=3)} minutes."
    )
    console.echo(message=message, level=LogLevel.SUCCESS)

    # Since we preset the descriptor class before passing the runtime to this function, the maximum training time may
    # actually not be accurate. This would be the case if the training runtime is limited by the maximum allowed water
    # delivery volume and not time. In this case, updates the training time to reflect the factual training time. This
    # would be the case if the reward delays array size is the same as the cumulative time array size, indicating no
    # slicing was performed due to session time constraints.
    if len(reward_delays) == len(cumulative_time):
        # Actual session time is the accumulated delay converted from seconds to minutes at the last index.
        # noinspection PyProtectedMember
        runtime._descriptor.training_time_m = np.ceil(cumulative_time[-1] / 60)

    # Initializes the runtime class. This starts all necessary processes and guides the user through the steps of
    # putting the animal on the VR rig.
    runtime.start()

    # Starts the visualizer process
    visualizer.initialize()

    # Configures all system components to support lick training
    runtime.lick_train_state()

    # Initializes the listener instance used to detect training abort signals and manual reward trigger signals sent
    # via the keyboard.
    listener = KeyboardListener()

    message = (
        f"Initiating lick training procedure. Press 'ESC' + 'q' to immediately abort the training at any "
        f"time. Press 'ESC' + 'r' to deliver 5 uL of water to the animal."
    )
    console.echo(message=message, level=LogLevel.INFO)

    # This tracker is used to terminate the training if manual abort command is sent via the keyboard
    terminate = False

    # Loops over all delays and delivers reward via the lick tube as soon as the delay expires.
    delay_timer.reset()
    for delay in tqdm(
        reward_delays,
        desc="Running lick training",
        unit="reward",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} rewards [{elapsed}<",
        postfix=f"{np.round(cumulative_time[max_samples_idx - 1] / 60, decimals=0)}]"
    ):
        # This loop is executed while the code is waiting for the delay to pass. Anything that needs to be done during
        # the delay has to go here
        while delay_timer.elapsed < delay:

            # Updates the visualizer plot ~every 30 ms. This should be enough to reliably capture all events of
            # interest and appear visually smooth to human observers.
            visualizer.update()

            # If the listener detects the default abort sequence, terminates the runtime.
            if listener.exit_signal:
                terminate = True  # Sets the terminate flag
                break  # Breaks the while loop

            # If the listener detects a reward delivery signal, delivers the reward to the animal
            if listener.reward_signal:
                runtime.deliver_reward(reward_size=5.0)  # Delivers 5 uL of water

        # If the user sent the abort command, terminates the training early
        if terminate:
            message = (
                "Lick training abort signal detected. Aborting the lick training with a graceful shutdown procedure."
            )
            console.echo(message=message, level=LogLevel.ERROR)
            break  # Breaks the for loop

        # Once the delay is up, triggers the solenoid valve to deliver water to the animal and starts timing the next
        # reward delay
        runtime.deliver_reward(reward_size=5.0)  # Delivers 5 uL of water
        delay_timer.reset()

    # Shutdown sequence:
    message = (
        f"Training runtime: Complete. Delaying for additional {lower_bound} seconds to ensure the animal "
        f"has time to consume the final dispensed reward."
    )
    console.echo(message=message, level=LogLevel.SUCCESS)
    delay_timer.delay_noblock(lower_bound * 1000000)  # Converts to microseconds before delaying

    # Closes the visualizer, as the runtime is now over
    visualizer.close()

    # Terminates the runtime. This also triggers data preprocessing and, after that, moves the data to storage
    # destinations.
    runtime.stop()


def calibrate_valve_logic(
    actor_port: str,
    headbar_port: str,
    lickport_port: str,
    valve_calibration_data: tuple[tuple[int | float, int | float], ...],
):
    """Encapsulates the logic used to fill, empty, check, and calibrate the water valve.

    This runtime allows interfacing with the water valve outside of training and experiment runtime contexts. Usually,
    this is done at the beginning and the end of each experimental / training day to ensure the valve operates smoothly
    during runtimes.

    Args:
        actor_port: The USB port to which the Actor Ataraxis Micro Controller (AMC) is connected.
        headbar_port: The USB port used by the headbar Zaber motor controllers (devices).
        lickport_port: The USB port used by the lickport Zaber motor controllers (devices).
        valve_calibration_data: A tuple of tuples, with each inner tuple storing a pair of values. The first value is
            the duration, in microseconds, the valve was open. The second value is the volume of dispensed water, in
            microliters.

    Notes:
        This runtime will position the Zaber motors to facilitate working with the valve.
    """
    # Enables the console
    if not console.enabled:
        console.enable()

    # Initializes a timer used to optimize console printouts for using the valve in debug mode (which also posts
    # things to console).
    delay_timer = PrecisionTimer("s")

    message = f"Initializing calibration assets..."
    console.echo(message=message, level=LogLevel.INFO)

    # Runs all calibration procedures inside a temporary directory which is deleted at the end of runtime.
    with tempfile.TemporaryDirectory(prefix="sl_valve_") as output_path:
        output_path = Path(output_path)

        # Initializes the data logger. Due to how the MicroControllerInterface class is implemented, this is required
        # even for runtimes that do not need to save data.
        logger = DataLogger(
            output_directory=output_path,
            instance_name="temp",
            exist_ok=True,
            process_count=1,
            thread_count=10,
        )
        logger.start()

        message = f"DataLogger: Started."
        console.echo(message=message, level=LogLevel.SUCCESS)

        # Initializes HeadBar and LickPort binding classes
        headbar = _HeadBar(headbar_port, output_path.joinpath("zaber_positions.yaml"))
        lickport = _LickPort(lickport_port, output_path.joinpath("zaber_positions.yaml"))

        message = f"Zaber controllers: Started."
        console.echo(message=message, level=LogLevel.SUCCESS)

        # Initializes the Actor MicroController with the valve module. Ignores all other modules at this time.
        valve: ValveInterface = ValveInterface(valve_calibration_data=valve_calibration_data, debug=True)
        controller: MicroControllerInterface = MicroControllerInterface(
            controller_id=np.uint8(101),
            microcontroller_serial_buffer_size=8192,
            microcontroller_usb_port=actor_port,
            data_logger=logger,
            module_interfaces=(valve,),
        )
        controller.start()
        controller.unlock_controller()

        # Delays for 2 seconds for the valve to initialize and send the state message. This avoids the visual clash
        # with he zaber positioning dialog
        delay_timer.delay_noblock(delay=1)

        message = f"Actor MicroController: Started."
        console.echo(message=message, level=LogLevel.SUCCESS)

        # Initializes the Zaber positioning sequence. This relies heavily on user feedback to confirm that it is safe to
        # proceed with motor movements.
        message = (
            "Preparing to move HeadBar and LickPort motors. Remove the mesoscope objective, swivel out the VR screens, "
            "and make sure the animal is NOT mounted on the rig. Failure to fulfill these steps may DAMAGE the "
            "mesoscope and / or HARM the animal."
        )
        console.echo(message=message, level=LogLevel.WARNING)
        while input("Enter 'y' to continue: ") != "y":
            continue

        # Homes all motors in-parallel. The homing trajectories for the motors as they are used now should not intersect
        # with each other, so it is safe to move both assemblies at the same time.
        headbar.prepare_motors(wait_until_idle=False)
        lickport.prepare_motors(wait_until_idle=True)
        headbar.wait_until_idle()

        # Moves the motors in calibration position.
        headbar.calibrate_position(wait_until_idle=False)
        lickport.calibrate_position(wait_until_idle=True)
        headbar.wait_until_idle()

        message = f"HeadBar and LickPort motors: Positioned for calibration runtime."
        console.echo(message=message, level=LogLevel.SUCCESS)

        # Notifies the user about supported calibration commands
        message = (
            "Supported Calibration commands: open, close, reference, reward, "
            "calibrate_15, calibrate_30, calibrate_45, calibrate__60. Use 'q' command to terminate the runtime."
        )
        console.echo(message=message, level=LogLevel.INFO)

        while True:
            command = input()  # Silent input to avoid visual spam.

            if command == "open":
                message = f"Opening valve."
                console.echo(message=message, level=LogLevel.INFO)
                valve.toggle(state=True)

            if command == "close":
                message = f"Closing valve."
                console.echo(message=message, level=LogLevel.INFO)
                valve.toggle(state=False)

            if command == "reward":
                message = f"Delivering 5 uL water reward."
                console.echo(message=message, level=LogLevel.INFO)
                pulse_duration = valve.get_duration_from_volume(target_volume=5.0)
                valve.set_parameters(pulse_duration=pulse_duration)
                valve.send_pulse()

            if command == "reference":
                message = f"Running the reference (200 x 5 uL pulse time) valve calibration procedure."
                console.echo(message=message, level=LogLevel.INFO)
                pulse_duration = valve.get_duration_from_volume(target_volume=5.0)
                valve.set_parameters(pulse_duration=pulse_duration)
                valve.calibrate()

            if command == "calibrate_15":
                message = f"Running 15 ms pulse valve calibration."
                console.echo(message=message, level=LogLevel.INFO)
                valve.set_parameters(pulse_duration=np.uint32(15000))  # 15 ms in us
                valve.calibrate()

            if command == "calibrate_30":
                message = f"Running 30 ms pulse valve calibration."
                console.echo(message=message, level=LogLevel.INFO)
                valve.set_parameters(pulse_duration=np.uint32(30000))  # 30 ms in us
                valve.calibrate()

            if command == "calibrate_45":
                message = f"Running 45 ms pulse valve calibration."
                console.echo(message=message, level=LogLevel.INFO)
                valve.set_parameters(pulse_duration=np.uint32(45000))  # 45 ms in us
                valve.calibrate()

            if command == "calibrate_60":
                message = f"Running 60 ms pulse valve calibration."
                console.echo(message=message, level=LogLevel.INFO)
                valve.set_parameters(pulse_duration=np.uint32(60000))  # 60 ms in us
                valve.calibrate()

            if command == "q":
                message = f"Terminating valve calibration runtime."
                console.echo(message=message, level=LogLevel.INFO)
                break

        # Instructs the user to remove the objective and the animal before resetting all zaber motors.
        message = (
            "Preparing to reset the HeadBar and LickPort motors. Remove all objects sued during calibration, such as "
            "water collection flasks, from the Mesoscope-VR cage."
        )
        console.echo(message=message, level=LogLevel.WARNING)
        while input("Enter 'y' to continue: ") != "y":
            continue

        # Shuts down zaber bindings
        headbar.park_position(wait_until_idle=False)
        lickport.park_position(wait_until_idle=True)
        headbar.wait_until_idle()
        headbar.disconnect()
        lickport.disconnect()

        message = f"HeadBar and LickPort connections: terminated."
        console.echo(message=message, level=LogLevel.SUCCESS)

        # Shuts down microcontroller interfaces
        controller.stop()

        message = f"Actor MicroController: Terminated."
        console.echo(message=message, level=LogLevel.SUCCESS)

        # Stops the data logger
        logger.stop()

        message = f"DataLogger: Terminated."
        console.echo(message=message, level=LogLevel.SUCCESS)

        # The logs will be cleaned up by deleting the temporary directory when this runtime exits.


def run_train_logic(
    runtime: _BehavioralTraining,
    initial_speed_threshold: float = 0.1,
    initial_duration_threshold: float = 0.1,
    maximum_speed_threshold: float = 10.0,
    maximum_duration_threshold: float = 10.0,
    maximum_water_volume: float = 1.0,
    maximum_training_time: int = 30,
    speed_increase_step: float = 0.5,
    duration_increase_step: float = 0.5,
    increase_water_threshold: float = 0.1,
) -> None:
    """Encapsulates the logic used to train animals how to operate the lick port.

    The lick training consists of delivering randomly spaced 5 uL water rewards via the Valve module to teach the
    animal that water comes out of the lick port. Each reward is delivered at a pseudorandom delay after the previous
    reward or training onset. Reward delay sequence is generated before training runtime by sampling a uniform
    distribution centered at 'average_reward_delay' with lower and upper bounds defined by
    'maximum_deviation_from_mean'. The training continues either until the valve delivers the 'maximum_water_volume' in
    milliliters or until the 'maximum_training_time' in minutes is reached, whichever comes first.

    Args:
        runtime: The initialized _BehavioralTraining instance that manages all Mesoscope-VR components used by this
            training runtime.
        average_reward_delay: The average time, in seconds, that separates two reward deliveries. This is used to
            generate the reward delay sequence as the center of the uniform distribution from which delays are sampled.
        maximum_deviation_from_mean: The maximum deviation from the average reward delay, in seconds. This determines
            the upper and lower boundaries for the data sampled from the uniform distribution centered at the
            average_reward_delay.
        maximum_water_volume: The maximum volume of water, in milliliters, that can be delivered during this runtime.
        maximum_training_time: The maximum time, in minutes, to run the training.

    Notes:
        All delays fall in the range of average_reward_delay +- maximum_deviation_from_mean.

        This function acts on top of the BehavioralTraining class and provides the overriding logic for the lick
        training process. During experiments, runtime logic is handled by Unity game engine, so specialized control
        functions are only required when training the animals without Unity.
    """
    # Initializes the timer that keeps the training running until the time threshold is reached
    delay_timer = PrecisionTimer("s")

    # Also initializes the timer used to track how long the animal maintains above-threshold running speed.
    lap_timer = PrecisionTimer("ms")

    # Uses runtime tracker extracted from the runtime instance to initialize the visualizer instance
    lick_tracker, valve_tracker, speed_tracker = runtime.trackers
    visualizer = BehaviorVisualizer(lick_tracker=lick_tracker, valve_tracker=valve_tracker, speed_tracker=speed_tracker)

    # Converts the training time from minutes to seconds to make it compatible with the timer precision.
    training_time = maximum_training_time * 60

    # Initializes the runtime class. This starts all necessary processes and guides the user through the steps of
    # putting the animal on the VR rig.
    runtime.start()

    # Starts the visualizer process
    visualizer.initialize()

    # Configures all system components to support lick training
    runtime.run_train_state()

    # Initializes the listener instance used to detect training abort signals sent via the keyboard.
    listener = KeyboardListener()

    message = (
        f"Initiating run training procedure. Press 'ESC' + 'q' to immediately abort the training at any "
        f"time. Press 'ESC' + 'r' to deliver 5 uL of water to the animal."
    )
    console.echo(message=message, level=LogLevel.INFO)

    # Creates a tqdm progress bar for time tracking (total = maximum training time in seconds)
    progress_bar = tqdm(
        total=training_time,
        desc="Run training progress",
        unit="sec",
        bar_format="{l_bar}{bar}| {n:.1f}/{total:.1f} sec [{elapsed}<{remaining}]"
    )

    # Defines tracker variables used below to determine when the animal is performing the training task satisfactorily.
    previous_speed = 0.0

    # Loops over all delays and delivers reward via the lick tube as soon as the delay expires.
    delay_timer.reset()
    last_update_time = 0  # Track when we last updated the progress bar

    # Main training loop
    while delay_timer.elapsed < training_time:

        # Reads the running speed from the speed tracker at each iteration
        current_speed = speed_tracker.read_data(index=0, convert_output=True)

        # If the speed is above the initial speed threshold,
        if current_speed > maximum_speed_threshold:
            lap_time = lap_timer.elapsed

            if previous_speed < maximum_speed_threshold:
                lap_timer.reset()

        # Updates the progress bar every second
        current_time = int(delay_timer.elapsed)
        if current_time > last_update_time:

            # Updates the progress bar by the number of seconds that have passed
            progress_bar.update(current_time - last_update_time)
            last_update_time = current_time

        # Updates the visualizer plot
        visualizer.update()

        # If the listener detects a reward delivery signal, delivers the reward to the animal
        if listener.reward_signal:
            runtime.deliver_reward(reward_size=5.0)  # Delivers 5 uL of water

        # If the user sent the abort command, terminates the training early
        if listener.exit_signal:
            message = (
                "Run training abort signal detected. Aborting the training with a graceful shutdown procedure."
            )
            console.echo(message=message, level=LogLevel.ERROR)
            break

    # Close the progress bar
    progress_bar.close()

    # Shutdown sequence:
    message = (
        f"Training runtime: Complete."
    )
    console.echo(message=message, level=LogLevel.SUCCESS)

    # Closes the visualizer, as the runtime is now over
    visualizer.close()

    # Terminates the runtime. This also triggers data preprocessing and, after that, moves the data to storage
    # destinations.
    runtime.stop()
