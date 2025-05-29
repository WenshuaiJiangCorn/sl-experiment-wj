from typing import Any
from pathlib import Path
from dataclasses import field, dataclass

from pynput import keyboard
from _typeshed import Incomplete
from sl_shared_assets import SessionData, MesoscopeSystemConfiguration

def get_system_configuration() -> MesoscopeSystemConfiguration:
    """Verifies that the current data acquisition system is the Mesoscope-VR and returns its configuration data.

    Raises:
        ValueError: If the local data acquisition system is not a Mesoscope-VR system.
    """
@dataclass()
class _VRPCPersistentData:
    """Stores the paths to the directories and files that make up the 'persistent_data' directory on the VRPC.

    VRPC persistent data directory is used to preserve configuration data, such as the positions of Zaber motors and
    Meososcope objective, so that they can be reused across sessions of the same animals. The data in this directory
    is read at the beginning of each session and replaced at the end of each session.
    """

    session_type: str
    persistent_data_path: Path
    zaber_positions_path: Path = field(default_factory=Path, init=False)
    mesoscope_positions_path: Path = field(default_factory=Path, init=False)
    session_descriptor_path: Path = field(default_factory=Path, init=False)
    def __post_init__(self) -> None: ...

@dataclass()
class _ScanImagePCData:
    """Stores the paths to the directories and files that make up the 'meso_data' directory on the ScanImagePC.

    During runtime, the ScanImagePC should organize all collected data under this root directory. During preprocessing,
    the VRPC uses SMB to access the data in this directory and merge it into the 'raw_data' session directory. The root
    ScanImagePC directory also includes the persistent_data directories for all animals and projects whose data is
    acquired via the Mesoscope-VR system.
    """

    session_name: str
    meso_data_path: Path
    persistent_data_path: Path
    mesoscope_data_path: Path = field(default_factory=Path, init=False)
    session_specific_path: Path = field(default_factory=Path, init=False)
    ubiquitin_path: Path = field(default_factory=Path, init=False)
    motion_estimator_path: Path = field(default_factory=Path, init=False)
    def __post_init__(self) -> None: ...

@dataclass()
class _VRPCDestinations:
    """Stores the paths to the VRPC filesystem-mounted directories of the Synology NAS and BioHPC server.

    The paths from this section are primarily used to transfer preprocessed data to the long-term storage destinations.
    """

    nas_raw_data_path: Path
    server_raw_data_path: Path
    server_processed_data_path: Path
    telomere_path: Path = field(default_factory=Path, init=False)
    def __post_init__(self) -> None: ...

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

    vrpc_persistent_data: Incomplete
    scanimagepc_data: Incomplete
    destinations: Incomplete
    def __init__(self, session_data: SessionData) -> None: ...

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
    """

    _data_array: Incomplete
    _currently_pressed: set[str]
    _keyboard_process: Incomplete
    _started: bool
    def __init__(self) -> None: ...
    def __del__(self) -> None:
        """Ensures all class resources are released before the instance is destroyed.

        This is a fallback method, using shutdown() should be standard.
        """
    def shutdown(self) -> None:
        """This method should be called at the end of runtime to properly release all resources and terminate the
        remote process."""
    _listener: keyboard.Listener
    def _run_keyboard_listener(self) -> None:
        """The main function that runs in the parallel process to monitor keyboard inputs."""
    def _on_press(self, key: Any) -> None:
        """Adds newly pressed keys to the storage set and determines whether the pressed key combination matches one of
        the expected combinations.

        This method is used as the 'on_press' callback for the Listener instance.
        """
    def _on_release(self, key: Any) -> None:
        """Removes no longer pressed keys from the storage set.

        This method is used as the 'on_release' callback for the Listener instance.
        """
    @property
    def exit_signal(self) -> bool:
        """Returns True if the listener has detected the runtime abort keys combination (ESC + q) being pressed.

        This indicates that the user has requested the runtime to gracefully abort.
        """
    @property
    def reward_signal(self) -> bool:
        """Returns True if the listener has detected the water reward delivery keys combination (ESC + r) being
        pressed.

        This indicates that the user has requested the system to deliver a water reward to the animal.

        Notes:
            Each time this property is accessed, the flag is reset to 0.
        """
    @property
    def speed_modifier(self) -> int:
        """Returns the current user-defined modifier to apply to the running speed threshold.

        This is used during run training to manually update the running speed threshold.
        """
    @property
    def duration_modifier(self) -> int:
        """Returns the current user-defined modifier to apply to the running epoch duration threshold.

        This is used during run training to manually update the running epoch duration threshold.
        """
