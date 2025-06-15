"""This module provides additional tools and classes used by other modules of the mesoscope_vr package. Primarily, this
includes various dataclasses specific to the Mesoscope-VR systems and utility functions used by other package modules.
The contents of this module are not intended to be used outside the mesoscope_vr package."""

from typing import Any
from pathlib import Path
from dataclasses import field, dataclass
from multiprocessing import Process

import numpy as np
from pynput import keyboard
from ataraxis_time import PrecisionTimer
from sl_shared_assets import SessionData, MesoscopeSystemConfiguration, get_system_configuration_data
from ataraxis_base_utilities import console, ensure_directory_exists
from ataraxis_data_structures import SharedMemoryArray


def get_system_configuration() -> MesoscopeSystemConfiguration:
    """Verifies that the current data acquisition system is the Mesoscope-VR and returns its configuration data.

    Raises:
        ValueError: If the local data acquisition system is not a Mesoscope-VR system.
    """
    system_configuration = get_system_configuration_data()
    if not isinstance(system_configuration, MesoscopeSystemConfiguration):
        message = (
            f"Unable to instantiate the MesoscopeData class, as the local data acquisition system is not a "
            f"Mesoscope-VR system. This either indicates a user error (calling incorrect Data class) or local data "
            f"acquisition system misconfiguration. To reconfigured the data-acquisition system, use the "
            f"sl-create-system-config' CLI command."
        )
        console.error(message, error=ValueError)
    return system_configuration


@dataclass()
class _VRPCPersistentData:
    """Stores the paths to the directories and files that make up the 'persistent_data' directory on the VRPC.

    VRPC persistent data directory is used to preserve configuration data, such as the positions of Zaber motors and
    Meososcope objective, so that they can be reused across sessions of the same animals. The data in this directory
    is read at the beginning of each session and replaced at the end of each session.
    """

    session_type: str
    """Stores the type of the Mesoscope-VR-compatible session for which this additional dataclass is instantiated. This 
    is used to resolve the cached session_Descriptor instance, as different session types use different descriptor 
    files."""
    persistent_data_path: Path
    """Stores the path to the project- and animal-specific 'persistent_data' directory relative to the VRPC root."""
    zaber_positions_path: Path = field(default_factory=Path, init=False)
    """Stores the path to the Zaber motor positions snapshot generated at the end of the previous session runtime. This 
    is used to automatically restore all Zaber motors to the same position across all sessions."""
    mesoscope_positions_path: Path = field(default_factory=Path, init=False)
    """Stores the path to the Mesoscope positions snapshot generated at the end of the previous session runtime. This 
    is used to help the user to (manually) restore the Mesoscope to the same position across all sessions."""
    session_descriptor_path: Path = field(default_factory=Path, init=False)
    """Stores the path to the session_descriptor.yaml file generated at the end of the previous session runtime. This 
    is used to automatically restore session runtime parameters used during the previous session. Primarily, this is 
    used during animal training."""

    def __post_init__(self) -> None:
        # Resolves paths that can be derived from the root path.
        self.zaber_positions_path = self.persistent_data_path.joinpath("zaber_positions.yaml")
        self.mesoscope_positions_path = self.persistent_data_path.joinpath("mesoscope_positions.yaml")

        # Resolves the session descriptor path based on the session type.
        if self.session_type == "lick training":
            self.session_descriptor_path = self.persistent_data_path.joinpath(f"lick_training_session_descriptor.yaml")
        elif self.session_type == "run training":
            self.session_descriptor_path = self.persistent_data_path.joinpath(f"run_training_session_descriptor.yaml")
        elif self.session_type == "mesoscope experiment":
            self.session_descriptor_path = self.persistent_data_path.joinpath(
                f"mesoscope_experiment_session_descriptor.yaml"
            )
        # Does not raise the error for window checking sessions, but also does not resolve the descriptor, as it is not
        # used during window checking
        elif self.session_type != "window checking":
            message = (
                f"Unsupported session type '{self.session_type}' encountered when initializing additional path "
                f"dataclasses for the Mesoscope-VR data acquisition system. Supported session types are "
                f"'lick training', 'run training', 'window checking' and 'mesoscope experiment'."
            )
            console.error(message, error=ValueError)

        # Ensures that the target persistent directory exists
        ensure_directory_exists(self.persistent_data_path)


@dataclass()
class _ScanImagePCData:
    """Stores the paths to the directories and files that make up the 'meso_data' directory on the ScanImagePC.

    During runtime, the ScanImagePC should organize all collected data under this root directory. During preprocessing,
    the VRPC uses SMB to access the data in this directory and merge it into the 'raw_data' session directory. The root
    ScanImagePC directory also includes the persistent_data directories for all animals and projects whose data is
    acquired via the Mesoscope-VR system.
    """

    session_name: str
    """Stores the name of the session for which this data management class is instantiated. This is used to rename the 
    general mesoscope data directory on the ScanImagePC to include the session-specific name."""
    meso_data_path: Path
    """Stores the path to the root ScanImagePC data directory, mounted to the VRPC filesystem via the SMB or equivalent 
    protocol. All mesoscope-generated data is stored under this root directory before it is merged into the VRPC-managed
    raw_data directory of each session."""
    persistent_data_path: Path
    """Stores the path to the project- and animal-specific 'persistent_data' directory relative to the ScanImagePC 
    root directory ('meso-data' directory)."""
    mesoscope_data_path: Path = field(default_factory=Path, init=False)
    """Stores the path to the 'default' mesoscope_data directory. All experiment sessions across all animals and 
    projects use the same mesoscope_data directory to save the data generated by the mesoscope via ScanImage 
    software. This simplifies ScanImagePC configuration process during runtime, as all data is always saved in the same
    directory. During preprocessing, the data is moved from the default directory first into a session-specific 
    ScanImagePC directory and then into the VRPC raw_data session directory."""
    session_specific_path: Path = field(default_factory=Path, init=False)
    """Stores the path to the session-specific data directory. This directory is generated at the end of each experiment
    runtime to prepare mesoscope data for being moved to the VRPC-managed raw_data directory and to reset the 'default' 
    mesoscope_data directory for the next session's runtime."""
    ubiquitin_path: Path = field(default_factory=Path, init=False)
    """Stores the path to the 'ubiquitin.bin' file. This file is automatically generated inside the session-specific 
    data directory after its contents are safely transferred to the VRPC as part of preprocessing. During redundant data
    removal step of preprocessing, the VRPC searches for directories marked with ubiquitin.bin and deletes them from the
    ScanImagePC filesystem."""
    motion_estimator_path: Path = field(default_factory=Path, init=False)
    """Stores the path to the 'reference' motion estimator file generated during the first experiment session of each 
    animal. This file is kept on the ScanImagePC to image the same population of cells across all experiment 
    sessions."""

    def __post_init__(
        self,
    ) -> None:
        # Resolves additional paths using the input root paths
        self.motion_estimator_path = self.persistent_data_path.joinpath("MotionEstimator.me")
        self.session_specific_path = self.meso_data_path.joinpath(self.session_name)
        self.ubiquitin_path = self.session_specific_path.joinpath("ubiquitin.bin")
        self.mesoscope_data_path = self.meso_data_path.joinpath("mesoscope_data")

        # Ensures that the shared data directory and the persistent data directory exist.
        ensure_directory_exists(self.mesoscope_data_path)
        ensure_directory_exists(self.persistent_data_path)


@dataclass()
class _VRPCDestinations:
    """Stores the paths to the VRPC filesystem-mounted directories of the Synology NAS and BioHPC server.

    The paths from this section are primarily used to transfer preprocessed data to the long-term storage destinations.
    """

    nas_raw_data_path: Path
    """Stores the path to the session's raw_data directory on the Synology NAS, which is mounted to the VRPC via the 
    SMB or equivalent protocol."""
    server_raw_data_path: Path
    """Stores the path to the session's raw_data directory on the BioHPC server, which is mounted to the VRPC via the 
    SMB or equivalent protocol."""
    server_processed_data_path: Path
    """Stores the path to the session's processed_data directory on the BioHPC server, which is mounted to the VRPC via 
    the SMB or equivalent protocol."""
    telomere_path: Path = field(default_factory=Path, init=False)
    """Stores the path to the session's telomere.bin marker. This marker is generated as part of data preprocessing on 
    the VRPC and can be removed by the BioHPC server to notify the VRPC that the server received preprocessed in a 
    compromised (damaged) state. If the telomere.bin file is present on the BioHPC server after the VRPC instructs the
    server to verify the integrity opf the transferred data, the VRPC concludes that the data was transferred intact and
    removes (purges) the local copy of raw_data."""

    def __post_init__(self) -> None:
        # Resolves the server-side telomere.bin marker path using the root directory.
        self.telomere_path = self.server_raw_data_path.joinpath("telomere.bin")

        # Ensures all destination directories exist
        ensure_directory_exists(self.nas_raw_data_path)
        ensure_directory_exists(self.server_raw_data_path)
        ensure_directory_exists(self.server_processed_data_path)


class MesoscopeData:
    """This works together with SessionData class to define additional filesystem paths used by the Mesoscope-VR
    data acquisition system during runtime.

    Specifically, the paths from this class are used during both data acquisition and preprocessing to work with
    the managed session's data across the machines (PCs) that make up the acquisition system, sucha s the VRPC and the
    ScanImagePC, and long-term storage infrastructure used in the Sun lab, such as the NAS and the BioHPC server.

    Args:
        session_data: The SessionData instance for the managed session.

    Attributes:
        vrpc_persistent_data: Stores paths to files inside the VRPC persistent_data directory for the managed session's
            project and animal.
        scanimagepc_data: Stores paths to all ScanImagePC (Mesoscope PC) files and directories used during data
            acquisition and processing.
        destinations: Stores paths to the long-term data storage destinations.
    """

    def __init__(self, session_data: SessionData):
        # Prevents this class from being instantiated on any acquisition system other than the Mesoscope-VR system.
        system_configuration = get_system_configuration()

        # Unpacks session paths nodes from the SessionData instance
        project = session_data.project_name
        animal = session_data.animal_id
        session = session_data.session_name

        # Instantiates additional path data classes
        self.vrpc_persistent_data = _VRPCPersistentData(
            session_type=session_data.session_type,
            persistent_data_path=system_configuration.paths.root_directory.joinpath(project, animal, "persistent_data"),
        )

        self.scanimagepc_data = _ScanImagePCData(
            session_name=session,
            meso_data_path=system_configuration.paths.mesoscope_directory,
            persistent_data_path=system_configuration.paths.mesoscope_directory.joinpath(
                project, animal, "persistent_data"
            ),
        )

        self.destinations = _VRPCDestinations(
            nas_raw_data_path=system_configuration.paths.nas_directory.joinpath(project, animal, session, "raw_data"),
            server_raw_data_path=system_configuration.paths.server_storage_directory.joinpath(
                project, animal, session, "raw_data"
            ),
            server_processed_data_path=system_configuration.paths.server_working_directory.joinpath(
                project, animal, session, "processed_data"
            ),
        )


class KeyboardListener:
    """Monitors the keyboard input for various runtime control signals and changes internal attributes to communicate
    detected signals to outside callers.

    This class is used during most Mesoscope-VR runtimes to allow the user to manually control various aspects of the
    Mesoscope-VR system and data acquisition runtime. For example, it is used to abort a data acquisition session early
    by gracefully stopping all Mesoscope-VR assets and running data preprocessing on the collected data.

    Notes:
        This keyboard monitor will pick up keyboard strokes directed at other applications during runtime. While our
        unique key combinations are likely not used elsewhere, exercise caution when using other applications while this
        monitor class is active.

        The monitor runs in a separate process (on a separate core) and sends the data to the main process via
        shared memory arrays. This prevents the listener from competing for resources with the runtime logic and the
        visualizer class.

    Attributes:
        _data_array: A SharedMemoryArray used to store the data recorded by the remote listener process.
        _currently_pressed: Stores the keys that are currently being pressed.
        _keyboard_process: The Listener instance used to monitor keyboard strokes. The listener runs in a remote
            process.
        _started: A static flag used to prevent the __del__ method from shutting down an already terminated instance.
        _previous_pause_flag: Keeps track of the 'pause runtime' flag value between key listening cycles. This is used
            to make repeated 'ESC + p' (pause combination) presses to flip the pause flag between 0 and 1.
    """

    def __init__(self) -> None:
        self._data_array = SharedMemoryArray.create_array(
            name="keyboard_listener", prototype=np.zeros(shape=6, dtype=np.int32), exist_ok=True
        )
        self._currently_pressed: set[str] = set()

        # Starts the listener process
        self._keyboard_process = Process(target=self._run_keyboard_listener, daemon=True)
        self._keyboard_process.start()
        self._started = True
        self._previous_pause_flag = False

    def __del__(self) -> None:
        """Ensures all class resources are released before the instance is destroyed.

        This is a fallback method, using shutdown() should be standard.
        """
        if self._started:
            self.shutdown()

    def shutdown(self) -> None:
        """This method should be called at the end of runtime to properly release all resources and terminate the
        remote process."""
        if self._keyboard_process.is_alive():
            self._data_array.write_data(index=0, data=np.int32(1))  # Termination signal
            self._keyboard_process.terminate()
            self._keyboard_process.join(timeout=1.0)
        self._data_array.disconnect()
        self._data_array.destroy()
        self._started = False

    def _run_keyboard_listener(self) -> None:
        """The main function that runs in the parallel process to monitor keyboard inputs."""

        # Sets up listeners for both press and release
        self._listener: keyboard.Listener = keyboard.Listener(on_press=self._on_press, on_release=self._on_release)
        self._listener.daemon = True
        self._listener.start()

        # Connects to the shared memory array from the remote process
        self._data_array.connect()

        # Initializes the timer used to delay the process. This reduces the CPU load
        delay_timer = PrecisionTimer("ms")

        # Keeps the process alive until it receives the shutdown command via the SharedMemoryArray instance
        while self._data_array.read_data(index=0, convert_output=True) == 0:
            delay_timer.delay_noblock(delay=10, allow_sleep=True)  # 10 ms delay

        # If the loop above is escaped, this indicates that the listener process has been terminated. Disconnects from
        # the shared memory array and exits
        self._data_array.disconnect()

    def _on_press(self, key: Any) -> None:
        """Adds newly pressed keys to the storage set and determines whether the pressed key combination matches one of
        the expected combinations.

        This method is used as the 'on_press' callback for the Listener instance.
        """
        # Updates the set with current data
        self._currently_pressed.add(str(key))

        # Checks if ESC is pressed (required for all combinations)
        if "Key.esc" in self._currently_pressed:
            # Exit combination: ESC + q
            if "'q'" in self._currently_pressed:
                self._data_array.write_data(index=1, data=np.int32(1))

            # Reward combination: ESC + r
            if "'r'" in self._currently_pressed:
                self._data_array.write_data(index=2, data=np.int32(1))

            # Running speed threshold control: ESC + Up/Down arrows
            if "Key.up" in self._currently_pressed:
                previous_value = self._data_array.read_data(index=3, convert_output=False)
                previous_value += 1
                self._data_array.write_data(index=3, data=previous_value)

            if "Key.down" in self._currently_pressed:
                previous_value = self._data_array.read_data(index=3, convert_output=False)
                previous_value -= 1
                self._data_array.write_data(index=3, data=previous_value)

            # Running duration threshold control: ESC + Left/Right arrows
            if "Key.right" in self._currently_pressed:
                previous_value = self._data_array.read_data(index=4, convert_output=False)
                previous_value -= 1
                self._data_array.write_data(index=4, data=previous_value)

            if "Key.left" in self._currently_pressed:
                previous_value = self._data_array.read_data(index=4, convert_output=False)
                previous_value += 1
                self._data_array.write_data(index=4, data=previous_value)

            # Runtime pause command: ESC + p
            # Note, repeated uses of this command toggle the system between paused and unpaused states.
            if "'p'" in self._currently_pressed:
                if self._previous_pause_flag:
                    self._data_array.write_data(index=5, data=np.int32(0))
                    self._previous_pause_flag = False
                else:
                    self._data_array.write_data(index=5, data=np.int32(1))
                    self._previous_pause_flag = True

    def _on_release(self, key: Any) -> None:
        """Removes no longer pressed keys from the storage set.

        This method is used as the 'on_release' callback for the Listener instance.
        """
        # Removes no longer pressed keys from the set
        key_str = str(key)
        if key_str in self._currently_pressed:
            self._currently_pressed.remove(key_str)

    @property
    def exit_signal(self) -> bool:
        """Returns True if the listener has detected the runtime abort keys combination (ESC + q) being pressed.

        This indicates that the user has requested the runtime to gracefully abort.
        """
        return bool(self._data_array.read_data(index=1, convert_output=True))

    @property
    def reward_signal(self) -> bool:
        """Returns True if the listener has detected the water reward delivery keys combination (ESC + r) being
        pressed.

        This indicates that the user has requested the system to deliver a water reward to the animal.

        Notes:
            Each time this property is accessed, the flag is reset to 0.
        """
        reward_flag = bool(self._data_array.read_data(index=2, convert_output=True))
        self._data_array.write_data(index=2, data=np.int32(0))
        return reward_flag

    @property
    def speed_modifier(self) -> int:
        """Returns the current user-defined modifier to apply to the running speed threshold.

        This is used during run training to manually update the running speed threshold.
        """
        return int(self._data_array.read_data(index=3, convert_output=True))

    @property
    def duration_modifier(self) -> int:
        """Returns the current user-defined modifier to apply to the running epoch duration threshold.

        This is used during run training to manually update the running epoch duration threshold.
        """
        return int(self._data_array.read_data(index=4, convert_output=True))

    @property
    def pause_runtime(self) -> bool:
        """Returns True if the listener has detected the runtime pause keys combination (ESC + p) being pressed.

        This indicates that the user has requested the acquisition system to pause the current runtime. When the system
        receives this signal, it suspends the ongoing runtime and switches to the 'idle' state until the user unpauses
        the system by using the "Esc + p' combination again
        """
        return bool(self._data_array.read_data(index=5, convert_output=True))
