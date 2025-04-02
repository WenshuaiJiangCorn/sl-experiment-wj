from pathlib import Path

from _typeshed import Incomplete
from ataraxis_video_system import VideoSystem
from ataraxis_data_structures import DataLogger, SharedMemoryArray
from ataraxis_communication_interface import MicroControllerInterface

from .data_classes import ZaberPositions as ZaberPositions
from .zaber_bindings import (
    ZaberAxis as ZaberAxis,
    ZaberConnection as ZaberConnection,
)
from .module_interfaces import (
    TTLInterface as TTLInterface,
    LickInterface as LickInterface,
    BreakInterface as BreakInterface,
    ValveInterface as ValveInterface,
    ScreenInterface as ScreenInterface,
    TorqueInterface as TorqueInterface,
    EncoderInterface as EncoderInterface,
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

    _headbar: ZaberConnection
    _headbar_z: ZaberAxis
    _headbar_pitch: ZaberAxis
    _headbar_roll: ZaberAxis
    _previous_positions: None | ZaberPositions
    def __init__(self, headbar_port: str, zaber_positions_path: Path) -> None: ...
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
    def get_positions(self) -> tuple[int, int, int]:
        """Returns the current position of all HeadBar motors in native motor units.

        The positions are returned in the order of : Z, Pitch, and Roll. These positions can be saves as a
        zaber_positions.yaml file to be used during the following runtimes.
        """
    def wait_until_idle(self) -> None:
        """This method blocks in-place while at least one motor in the managed motor group is moving.

        Primarily, this method is used to issue commands to multiple motor groups and then block until all motors in
        all groups finish moving. This optimizes the overall time taken to move the motors.
        """
    def disconnect(self) -> None:
        """Disconnects from the access port of the motor group.

        This method should be called after the motors are parked (moved to their final parking position) to release
        the connection resources. If this method is not called, the runtime will NOT be able to terminate.

        Notes:
            Calling this method will execute the motor parking sequence, which involves moving the motors to their
            parking position. Make sure there are no animals mounted on the rig and that the mesoscope objective is
            removed from the rig before executing this command.
        """

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

    _lickport: ZaberConnection
    _lickport_z: ZaberAxis
    _lickport_x: ZaberAxis
    _lickport_y: ZaberAxis
    _previous_positions: None | ZaberPositions
    def __init__(self, lickport_port: str, zaber_positions_path: Path) -> None: ...
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
    def get_positions(self) -> tuple[int, int, int]:
        """Returns the current position of all LickPort motors in native motor units.

        The positions are returned in the order of : Z, X, and Y. These positions can be saves as a zaber_positions.yaml
        file to be used during the following runtimes.
        """
    def wait_until_idle(self) -> None:
        """This method blocks in-place while at least one motor in the managed motor group is moving.

        Primarily, this method is used to issue commands to multiple motor groups and then block until all motors in
        all groups finish moving. This optimizes the overall time taken to move the motors.
        """
    def disconnect(self) -> None:
        """Disconnects from the access port of the motor group.

        This method should be called after the motors are parked (moved to their final parking position) to release
        the connection resources. If this method is not called, the runtime will NOT be able to terminate.

        Notes:
            Calling this method will execute the motor parking sequence, which involves moving the motors to their
            parking position. Make sure there are no animals mounted on the rig and that the mesoscope objective is
            removed from the rig before executing this command.
        """

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

    _started: bool
    _previous_volume: float
    _screen_state: bool
    mesoscope_start: TTLInterface
    mesoscope_stop: TTLInterface
    wheel_break: Incomplete
    valve: Incomplete
    screens: Incomplete
    _actor: MicroControllerInterface
    mesoscope_frame: TTLInterface
    lick: LickInterface
    torque: TorqueInterface
    _sensor: MicroControllerInterface
    wheel_encoder: EncoderInterface
    _encoder: MicroControllerInterface
    def __init__(
        self,
        data_logger: DataLogger,
        actor_port: str,
        sensor_port: str,
        encoder_port: str,
        valve_calibration_data: tuple[tuple[int | float, int | float], ...],
        debug: bool = False,
    ) -> None: ...
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
    def stop(self) -> None:
        """Stops all MicroController communication processes and releases all resources.

        This method needs to be called at the end of each runtime to release the resources reserved by the start()
        method. Until the stop() method is called, the DataLogger instance may receive data from running
        MicroControllers, so calling this method also guarantees no MicroController data will be lost if the DataLogger
        process is terminated.
        """
    def enable_encoder_monitoring(self) -> None:
        """Enables wheel encoder monitoring at 2 kHz rate.

        This means that, at most, the Encoder will send the data to the PC at the 2 kHz rate. The Encoder collects data
        at the native rate supported by the microcontroller hardware, which likely exceeds the reporting rate.
        """
    def disable_encoder_monitoring(self) -> None:
        """Stops monitoring the wheel encoder."""
    def start_mesoscope(self) -> None:
        """Sends the acquisition start TTL pulse to the mesoscope."""
    def stop_mesoscope(self) -> None:
        """Sends the acquisition stop TTL pulse to the mesoscope."""
    def enable_break(self) -> None:
        """Engages the wheel break at maximum strength, preventing the animal from running on the wheel."""
    def disable_break(self) -> None:
        """Disengages the wheel break, enabling the animal to run on the wheel."""
    def enable_vr_screens(self) -> None:
        """Sets the VR screens to be ON."""
    def disable_vr_screens(self) -> None:
        """Sets the VR screens to be OFF."""
    def enable_mesoscope_frame_monitoring(self) -> None:
        """Enables monitoring the TTL pulses sent by the mesoscope to communicate when it is scanning a frame at
        ~ 1 kHZ rate.

        The mesoscope sends the HIGH phase of the TTL pulse while it is scanning the frame, which produces a pulse of
        ~100ms. This is followed by ~5ms LOW phase during which the Galvos are executing the flyback procedure. This
        command checks the state of the TTL pin at the 1 kHZ rate, which is enough to accurately report both phases.
        """
    def disable_mesoscope_frame_monitoring(self) -> None:
        """Stops monitoring the TTL pulses sent by the mesoscope to communicate when it is scanning a frame."""
    def enable_lick_monitoring(self) -> None:
        """Enables monitoring the state of the conductive lick sensor at ~ 1 kHZ rate.

        The lick sensor measures the voltage across the lick sensor and reports surges in voltage to the PC as a
        reliable proxy for tongue-to-sensor contact. Most lick events span at least 100 ms of time and, therefore, the
        rate of 1 kHZ is adequate for resolving all expected single-lick events.
        """
    def disable_lick_monitoring(self) -> None:
        """Stops monitoring the conductive lick sensor."""
    def enable_torque_monitoring(self) -> None:
        """Enables monitoring the torque sensor at ~ 1 kHZ rate.

        The torque sensor detects CW and CCW torques applied by the animal to the wheel. Currently, we do not have a
        way of reliably calibrating the sensor, so detected torque magnitudes are only approximate. However, the sensor
        reliably distinguishes large torques from small torques and accurately tracks animal motion activity when the
        wheel break is engaged.
        """
    def disable_torque_monitoring(self) -> None:
        """Stops monitoring the torque sensor."""
    def open_valve(self) -> None:
        """Opens the water reward solenoid valve.

        This method is primarily used to prime the water line with water before the first experiment or training session
        of the day.
        """
    def close_valve(self) -> None:
        """Closes the water reward solenoid valve."""
    def deliver_reward(self, volume: float = 5.0) -> None:
        """Pulses the water reward solenoid valve for the duration of time necessary to deliver the provided volume of
        water.

        This method assumes that the valve has been calibrated before calling this method. It uses the calibration data
        provided at class instantiation to determine the period of time the valve should be kept open to deliver the
        requested volume of water.

        Args:
            volume: The volume of water to deliver, in microliters.
        """
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
    @property
    def mesoscope_frame_count(self) -> int:
        """Returns the total number of mesoscope frame acquisition pulses recorded since runtime onset."""
    @property
    def total_delivered_volume(self) -> float:
        """Returns the total volume of water, in microliters, dispensed by the valve since runtime onset."""
    @property
    def distance_tracker(self) -> SharedMemoryArray:
        """Returns the SharedMemoryArray used to communicate the total distance traveled by the animal since runtime
        onset.

        This array should be passed to a Visualizer class so that it can sample the shared data to generate real-time
        running speed plots. It is also used by the run training logic to evaluate animal's performance during training.
        """
    @property
    def lick_tracker(self) -> SharedMemoryArray:
        """Returns the SharedMemoryArray used to communicate the lick sensor status.

        This array should be passed to a Visualizer class so that it can sample the shared data to generate real-time
        lick detection plots.
        """
    @property
    def valve_tracker(self) -> SharedMemoryArray:
        """Returns the SharedMemoryArray used to communicate the water reward valve state.

        This array should be passed to a Visualizer class so that it can sample the shared data to generate real-time
        reward delivery plots.
        """

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

    _face_camera_started: bool
    _body_cameras_started: bool
    _face_camera: VideoSystem
    _left_camera: VideoSystem
    _right_camera: VideoSystem
    def __init__(
        self,
        data_logger: DataLogger,
        output_directory: Path,
        face_camera_index: int,
        left_camera_index: int,
        right_camera_index: int,
        harvesters_cti_path: Path,
    ) -> None: ...
    def start_face_camera(self) -> None:
        """Starts face camera frame acquisition.

        This method sets up both the frame acquisition (producer) process and the frame saver (consumer) process.
        However, the consumer process will not save any frames until save_face_camera_frames() method is called.
        """
    def start_body_cameras(self) -> None:
        """Starts left and right (body) camera frame acquisition.

        This method sets up both the frame acquisition (producer) process and the frame saver (consumer) process for
        both cameras. However, the consumer processes will not save any frames until save_body_camera_frames() method is
        called.
        """
    def save_face_camera_frames(self) -> None:
        """Starts saving the frames acquired by the face camera as a video file."""
    def save_body_camera_frames(self) -> None:
        """Starts saving the frames acquired by the left and right body cameras as a video file."""
    def stop(self) -> None:
        """Stops saving all camera frames and terminates the managed VideoSystems.

        This method needs to be called at the end of each runtime to release the resources reserved by the start()
        methods. Until the stop() method is called, the DataLogger instance may receive data from running
        VideoSystems, so calling this method also guarantees no VideoSystem data will be lost if the DataLogger
        process is terminated. Similarly, this guarantees the integrity of the generated video files.
        """
    @property
    def face_camera_log_path(self) -> Path:
        """Returns the path to the compressed .npz archive that stores the data logged by the face camera during
        runtime."""
    @property
    def left_camera_log_path(self) -> Path:
        """Returns the path to the compressed .npz archive that stores the data logged by the left body camera during
        runtime."""
    @property
    def right_camera_log_path(self) -> Path:
        """Returns the path to the compressed .npz archive that stores the data logged by the right body camera during
        runtime."""
