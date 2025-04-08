"""This module binds Ataraxis classes for all Mesoscope-VR components (cameras, microcontrollers, Zaber motors). These
bindings streamline the API used to interface with these components during experiment and training runtimes. Critically,
these classes statically define optimal runtime configuration parameters for all managed components. Source code
refactoring and a new library release are required each time these settings need to be updated."""

from pathlib import Path

import numpy as np
from ataraxis_video_system import (
    VideoCodecs,
    VideoSystem,
    VideoFormats,
    CameraBackends,
    GPUEncoderPresets,
    InputPixelFormats,
    OutputPixelFormats,
)
from ataraxis_base_utilities import LogLevel, console
from ataraxis_data_structures import DataLogger, SharedMemoryArray
from ataraxis_communication_interface import MicroControllerInterface

from .data_classes import ZaberPositions
from .zaber_bindings import ZaberAxis, ZaberConnection
from .module_interfaces import (
    TTLInterface,
    LickInterface,
    BreakInterface,
    ValveInterface,
    ScreenInterface,
    TorqueInterface,
    EncoderInterface,
)


class HeadBar:
    """Interfaces with Zaber motors that control the position of the HeadBar manipulator arm.

    This class abstracts working with Zaber motors that move the HeadBar in Z, Pitch, and Roll axes. It is used
    by the major runtime classes, such as MesoscopeExperiment, to interface with HeadBar motors. The class is designed
    to transition the HeadBar between a set of predefined states and should not be used directly by the user.

    Notes:
        This class does not contain the guards that notify users about risks associated with moving the motors. Do not
        use any methods from this class unless you know what you are doing. It is very easy to damage the motors, the
        mesoscope, or harm the animal.

        To fine-tune the position of any HeadBar motors in real time, use the main Zaber Launcher interface
        (https://software.zaber.com/zaber-launcher/download) installed on the VRPC.

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
        self._previous_positions: None | ZaberPositions = None
        if zaber_positions_path.exists():
            self._previous_positions = ZaberPositions.from_yaml(zaber_positions_path)  # type: ignore

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


class LickPort:
    """Interfaces with Zaber motors that control the position of the LickPort manipulator arm.

    This class abstracts working with Zaber motors that move the LickPort in Z, X, and Y axes. It is used
    by the major runtime classes, such as MesoscopeExperiment, to interface with LickPort motors. The class is designed
    to transition the LickPort between a set of predefined states and should not be used directly by the user.

    Notes:
        This class does not contain the guards that notify users about risks associated with moving the motors. Do not
        use any methods from this class unless you know what you are doing. It is very easy to damage the motors, the
        mesoscope, or harm the animal.

        To fine-tune the position of any HeadBar motors in real time, use the main Zaber Launcher interface
        (https://software.zaber.com/zaber-launcher/download) installed on the VRPC.

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
        self._previous_positions: None | ZaberPositions = None
        if zaber_positions_path.exists():
            self._previous_positions = ZaberPositions.from_yaml(zaber_positions_path)  # type: ignore

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


class MicroControllerInterfaces:
    """Interfaces with all Ataraxis Micro Controller (AMC) devices that control and record non-video behavioral data
    from the Mesoscope-VR system.

    This class interfaces with the three AMC controllers used during various runtimes: Actor, Sensor, and Encoder. The
    class exposes methods to send commands to the hardware modules managed by these microcontrollers. In turn, these
    modules control specific components of the Mesoscope-Vr system, such as rotary encoders, solenoid valves, and
    conductive lick sensors.

    Notes:
        This class is primarily intended to be used internally by the MesoscopeExperiment and BehavioralTraining
        classes. Our vr-maintenance CLI (sl-maintain-vr) uses this class directly to calibrate the water valve, but
        this is a unique use scenario. Do not initialize this class directly unless you know what you are doing.

        Calling the initializer does not start the underlying processes. Use the start() method before issuing other
        commands to properly initialize all remote processes. This design is intentional and is used during experiment
        and training runtimes to parallelize data preprocessing and starting the next animal's session.

    Args:
        data_logger: The initialized DataLogger instance used to log the data generated by the managed microcontrollers.
            For most runtimes, this argument is resolved by the MesoscopeExperiment or BehavioralTraining classes that
            initialize this class.
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
        mesoscope_start: The interface that starts mesoscope frame acquisition via TTL pulse.
        mesoscope_stop: The interface that stops mesoscope frame acquisition via TTL pulse.
        wheel_break: The interface that controls the electromagnetic break attached to the running wheel.
        valve: The interface that controls the solenoid water valve that delivers water to the animal.
        screens: The interface that controls the power state of the VR display screens.
        _actor: The main interface for the 'Actor' Ataraxis Micro Controller (AMC) device.
        mesoscope_frame: The interface that monitors frame acquisition timestamp signals sent by the mesoscope.
        lick: The interface that monitors animal's interactions with the lick sensor (detects licks).
        torque: The interface that monitors the torque applied by the animal to the running wheel.
        _sensor: The main interface for the 'Sensor' Ataraxis Micro Controller (AMC) device.
        wheel_encoder: The interface that monitors the rotation of the running wheel and converts it into the distance
            traveled by the animal.
        _encoder: The main interface for the 'Encoder' Ataraxis Micro Controller (AMC) device.

    Raises:
        TypeError: If the provided valve_calibration_data argument is not a tuple or does not contain valid elements.
    """

    def __init__(
        self,
        data_logger: DataLogger,
        actor_port: str,
        sensor_port: str,
        encoder_port: str,
        valve_calibration_data: tuple[tuple[int | float, int | float], ...],
        debug: bool = False,
    ) -> None:
        # Initializes the start state tracker first
        self._started: bool = False

        self._previous_volume: float = 0.0
        #  Tracks the current screen state. Assumes the screens are always OFF at class initialization.
        self._screen_state: bool = False

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
        self.mesoscope_start: TTLInterface = TTLInterface(module_id=np.uint8(1), debug=debug)
        self.mesoscope_stop: TTLInterface = TTLInterface(module_id=np.uint8(2), debug=debug)
        self.wheel_break = BreakInterface(
            minimum_break_strength=43.2047,  # 0.6 in oz
            maximum_break_strength=1152.1246,  # 16 in oz
            object_diameter=15.0333,  # 15 cm diameter + 0.0333 to account for the wrap
            debug=debug,
        )
        self.valve = ValveInterface(valve_calibration_data=valve_calibration_data, debug=debug)
        self.screens = ScreenInterface(initially_on=False, debug=debug)

        # Main interface:
        self._actor: MicroControllerInterface = MicroControllerInterface(
            controller_id=np.uint8(101),
            microcontroller_serial_buffer_size=8192,
            microcontroller_usb_port=actor_port,
            data_logger=data_logger,
            module_interfaces=(self.mesoscope_start, self.mesoscope_stop, self.wheel_break, self.valve, self.screens),
        )

        # SENSOR. Sensor AMC controls the hardware that collects data at regular intervals. This includes lick sensors,
        # torque sensors, and input TTL recorders. Critically, all managed hardware does not rely on hardware interrupt
        # logic to maintain the necessary precision.

        # Module interfaces:
        # Mesoscope frame timestamp recorder. THe class is configured to report detected pulses during runtime to
        # support checking whether mesoscope start trigger correctly starts the frame acquisition process.
        self.mesoscope_frame: TTLInterface = TTLInterface(module_id=np.uint8(1), report_pulses=True, debug=debug)
        self.lick: LickInterface = LickInterface(lick_threshold=300, debug=debug)  # Lick sensor
        self.torque: TorqueInterface = TorqueInterface(
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
            module_interfaces=(self.mesoscope_frame, self.lick, self.torque),
        )

        # ENCODER. Encoder AMC is specifically designed to interface with a rotary encoder connected to the running
        # wheel. The encoder uses hardware interrupt logic to maintain high precision and, therefore, it is isolated
        # to a separate microcontroller to ensure adequate throughput.

        # Module interfaces:
        self.wheel_encoder: EncoderInterface = EncoderInterface(
            encoder_ppr=8192, object_diameter=15.0333, cm_per_unity_unit=10.0, debug=debug
        )

        # Main interface:
        self._encoder: MicroControllerInterface = MicroControllerInterface(
            controller_id=np.uint8(203),
            microcontroller_serial_buffer_size=8192,
            microcontroller_usb_port=encoder_port,
            data_logger=data_logger,
            module_interfaces=(self.wheel_encoder,),
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
        self.wheel_encoder.set_parameters(report_cw=False, report_ccw=True, delta_threshold=15)

        # Configures mesoscope start and stop triggers to use 10 ms pulses
        self.mesoscope_start.set_parameters(pulse_duration=np.uint32(10000))
        self.mesoscope_stop.set_parameters(pulse_duration=np.uint32(10000))

        # Configures screen trigger to use 500 ms pulses
        self.screens.set_parameters(pulse_duration=np.uint32(500000))

        # Configures the water valve to deliver ~ 5 uL of water. Also configures the valve calibration method to run the
        # 'reference' calibration for 5 uL rewards used to verify the valve calibration before every experiment.
        self.valve.set_parameters(
            pulse_duration=np.uint32(35590),
            calibration_delay=np.uint32(200000),
            calibration_count=np.uint16(200),
            tone_duration=np.uint32(300000),
        )

        # Configures the lick sensor to filter out dry touches and only report significant changes in detected voltage
        # (used as a proxy for detecting licks).
        self.lick.set_parameters(
            signal_threshold=np.uint16(250), delta_threshold=np.uint16(250), averaging_pool_size=np.uint8(10)
        )

        # Configures the torque sensor to filter out noise and sub-threshold 'slack' torque signals.
        self.torque.set_parameters(
            report_ccw=np.bool(True),
            report_cw=np.bool(True),
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
        self.wheel_encoder.reset_pulse_count()
        self.wheel_encoder.check_state(repetition_delay=np.uint32(500))

    def disable_encoder_monitoring(self) -> None:
        """Stops monitoring the wheel encoder."""
        self.wheel_encoder.reset_command_queue()

    def start_mesoscope(self) -> None:
        """Sends the acquisition start TTL pulse to the mesoscope."""
        self.mesoscope_start.send_pulse()

    def stop_mesoscope(self) -> None:
        """Sends the acquisition stop TTL pulse to the mesoscope."""
        self.mesoscope_stop.send_pulse()

    def enable_break(self) -> None:
        """Engages the wheel break at maximum strength, preventing the animal from running on the wheel."""
        self.wheel_break.toggle(state=True)

    def disable_break(self) -> None:
        """Disengages the wheel break, enabling the animal to run on the wheel."""
        self.wheel_break.toggle(state=False)

    def enable_vr_screens(self) -> None:
        """Sets the VR screens to be ON."""
        if not self._screen_state:  # If screens are OFF
            self.screens.toggle()  # Sets them ON
            self._screen_state = True

    def disable_vr_screens(self) -> None:
        """Sets the VR screens to be OFF."""
        if self._screen_state:  # If screens are ON
            self.screens.toggle()  # Sets them OFF
            self._screen_state = False

    def enable_mesoscope_frame_monitoring(self) -> None:
        """Enables monitoring the TTL pulses sent by the mesoscope to communicate when it is scanning a frame at
        ~ 1 kHZ rate.

        The mesoscope sends the HIGH phase of the TTL pulse while it is scanning the frame, which produces a pulse of
        ~100ms. This is followed by ~5ms LOW phase during which the Galvos are executing the flyback procedure. This
        command checks the state of the TTL pin at the 1 kHZ rate, which is enough to accurately report both phases.
        """
        self.mesoscope_frame.check_state(repetition_delay=np.uint32(1000))

    def disable_mesoscope_frame_monitoring(self) -> None:
        """Stops monitoring the TTL pulses sent by the mesoscope to communicate when it is scanning a frame."""
        self.mesoscope_frame.reset_command_queue()

    def enable_lick_monitoring(self) -> None:
        """Enables monitoring the state of the conductive lick sensor at ~ 1 kHZ rate.

        The lick sensor measures the voltage across the lick sensor and reports surges in voltage to the PC as a
        reliable proxy for tongue-to-sensor contact. Most lick events span at least 100 ms of time and, therefore, the
        rate of 1 kHZ is adequate for resolving all expected single-lick events.
        """
        self.lick.check_state(repetition_delay=np.uint32(1000))

    def disable_lick_monitoring(self) -> None:
        """Stops monitoring the conductive lick sensor."""
        self.lick.reset_command_queue()

    def enable_torque_monitoring(self) -> None:
        """Enables monitoring the torque sensor at ~ 1 kHZ rate.

        The torque sensor detects CW and CCW torques applied by the animal to the wheel. Currently, we do not have a
        way of reliably calibrating the sensor, so detected torque magnitudes are only approximate. However, the sensor
        reliably distinguishes large torques from small torques and accurately tracks animal motion activity when the
        wheel break is engaged.
        """
        self.torque.check_state(repetition_delay=np.uint32(1000))

    def disable_torque_monitoring(self) -> None:
        """Stops monitoring the torque sensor."""
        self.torque.reset_command_queue()

    def open_valve(self) -> None:
        """Opens the water reward solenoid valve.

        This method is primarily used to prime the water line with water before the first experiment or training session
        of the day.
        """
        self.valve.toggle(state=True)

    def close_valve(self) -> None:
        """Closes the water reward solenoid valve."""
        self.valve.toggle(state=False)

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
            self.valve.set_parameters(
                pulse_duration=self.valve.get_duration_from_volume(volume),
                calibration_delay=np.uint32(200000),
                calibration_count=np.uint16(200),
            )
        self.valve.send_pulse()

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
        self.valve.set_parameters(
            pulse_duration=np.uint32(self.valve.get_duration_from_volume(target_volume=5.0)),
            calibration_delay=np.uint32(200000),
            calibration_count=np.uint16(200),
        )  # 5 ul x 200 times
        self.valve.calibrate()

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
        self.valve.set_parameters(
            pulse_duration=np.uint32(pulse_us), calibration_delay=np.uint32(200000), calibration_count=np.uint16(500)
        )
        self.valve.calibrate()

    @property
    def mesoscope_frame_count(self) -> int:
        """Returns the total number of mesoscope frame acquisition pulses recorded since runtime onset."""
        return self.mesoscope_frame.pulse_count

    @property
    def total_delivered_volume(self) -> float:
        """Returns the total volume of water, in microliters, dispensed by the valve since runtime onset."""
        return self.valve.delivered_volume

    @property
    def distance_tracker(self) -> SharedMemoryArray:
        """Returns the SharedMemoryArray used to communicate the total distance traveled by the animal since runtime
        onset.

        This array should be passed to a Visualizer class so that it can sample the shared data to generate real-time
        running speed plots. It is also used by the run training logic to evaluate animal's performance during training.
        """
        return self.wheel_encoder.distance_tracker

    @property
    def lick_tracker(self) -> SharedMemoryArray:
        """Returns the SharedMemoryArray used to communicate the lick sensor status.

        This array should be passed to a Visualizer class so that it can sample the shared data to generate real-time
        lick detection plots.
        """
        return self.lick.lick_tracker

    @property
    def valve_tracker(self) -> SharedMemoryArray:
        """Returns the SharedMemoryArray used to communicate the water reward valve state.

        This array should be passed to a Visualizer class so that it can sample the shared data to generate real-time
        reward delivery plots.
        """
        return self.valve.valve_tracker


class VideoSystems:
    """Interfaces with all cameras managed by Ataraxis Video System (AVS) classes that acquire and save camera frames
    as .mp4 video files.

    This class interfaces with the three AVS cameras used during various runtimes to record animal behavior: the face
    camera and the two body cameras (the left camera and the right camera). The face camera is a high-grade scientific
    camera that records the animal's face and pupil. The left and right cameras are lower-end security cameras recording
    the animal's body from the left and right sides.

    Notes:
        This class is primarily intended to be used internally by the MesoscopeExperiment and BehavioralTraining
        classes. Do not initialize this class directly unless you know what you are doing.

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
        face_camera_index: int,
        left_camera_index: int,
        right_camera_index: int,
        harvesters_cti_path: Path,
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
            preset=GPUEncoderPresets.SLOW,
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
            preset=GPUEncoderPresets.FAST,
            input_pixel_format=InputPixelFormats.MONOCHROME,
            output_pixel_format=OutputPixelFormats.YUV420,
            quantization_parameter=15,
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
            preset=GPUEncoderPresets.FAST,
            input_pixel_format=InputPixelFormats.MONOCHROME,
            output_pixel_format=OutputPixelFormats.YUV420,
            quantization_parameter=15,
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

    @property
    def face_camera_log_path(self) -> Path:
        """Returns the path to the compressed .npz archive that stores the data logged by the face camera during
        runtime."""
        return self._face_camera.log_path

    @property
    def left_camera_log_path(self) -> Path:
        """Returns the path to the compressed .npz archive that stores the data logged by the left body camera during
        runtime."""
        return self._left_camera.log_path

    @property
    def right_camera_log_path(self) -> Path:
        """Returns the path to the compressed .npz archive that stores the data logged by the right body camera during
        runtime."""
        return self._right_camera.log_path
