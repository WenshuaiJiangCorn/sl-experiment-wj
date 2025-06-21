from pathlib import Path
from dataclasses import field, dataclass

from _typeshed import Incomplete
from PyQt6.QtGui import QCloseEvent
from PyQt6.QtWidgets import QMainWindow
from sl_shared_assets import SessionData, MesoscopeSystemConfiguration
from ataraxis_data_structures import SharedMemoryArray

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

class RuntimeControlUI:
    """Provides a real-time Graphical User Interface (GUI) that allows interactively controlling certain Mesoscope-VR
    runtime parameters.

    The UI itself runs in a parallel process and communicates with an instance of this class via the SharedMemoryArray
    instance. This optimizes the UI's responsiveness without overburdening the main thread that runs the task logic and
    the animal performance visualization.

    Notes:
        This class is specialized to work with the Qt5 framework. In the future, it may be refactored to support the Qt6
        framework.

        The UI starts the runtime in the 'paused' state to allow the user to check the valve and all other runtime
        components before formally starting the runtime.

    Attributes:
        _data_array: A SharedMemoryArray used to store the data recorded by the remote UI process.
        _ui_process: The Process instance running the Qt5 UI.
        _started: A static flag used to prevent the __del__ method from shutting down an already terminated instance.
    """

    _data_array: Incomplete
    _ui_process: Incomplete
    _started: bool
    def __init__(self) -> None: ...
    def __del__(self) -> None:
        """Ensures all class resources are released before the instance is destroyed.

        This is a fallback method, using shutdown() directly is the preferred way of releasing resources.
        """
    def shutdown(self) -> None:
        """Shuts down the UI and releases all SharedMemoryArray resources.

        This method should be called at the end of runtime to properly release all resources and terminate the
        remote UI process.
        """
    def _run_ui_process(self) -> None:
        """The main function that runs in the parallel process to display and manage the Qt5 UI.

        This runs Qt5 in the main thread of the separate process, which is perfectly valid.
        """
    @property
    def exit_signal(self) -> bool:
        """Returns True if the user has requested the runtime to gracefully abort.

        Notes:
            Each time this property is accessed, the flag is reset to 0.
        """
    @property
    def reward_signal(self) -> bool:
        """Returns True if the user has requested the system to deliver a water reward.

        Notes:
            Each time this property is accessed, the flag is reset to 0.
        """
    @property
    def speed_modifier(self) -> int:
        """Returns the current user-defined modifier to apply to the running speed threshold."""
    @property
    def duration_modifier(self) -> int:
        """Returns the current user-defined modifier to apply to the running epoch duration threshold."""
    @property
    def pause_runtime(self) -> bool:
        """Returns True if the user has requested the acquisition system to pause the current runtime.

        Notes:
            Unlike most other flags,t he state of this flag does NOT change when it is accessed. Instead, the UI flips
            it between True and False to pause and resume the managed runtime.
        """
    @property
    def open_valve(self) -> bool:
        """Returns True if the user has requested the acquisition system to permanently open the water delivery
        valve.

        Notes:
            Each time this property is accessed, the flag is reset to 0.
        """
    @property
    def close_valve(self) -> bool:
        """Returns True if the user has requested the acquisition system to permanently close the water delivery
        valve.

        Notes:
            Each time this property is accessed, the flag is reset to 0.
        """
    @property
    def reward_volume(self) -> int:
        """Returns the current user-defined water reward volume value.

        Querying this property allows dynamically configuring the Mesoscope-VR system to use the user-defined volume of
        water each time it rewards the animal.
        """

class _ControlUIWindow(QMainWindow):
    """Generates, renders, and maintains the main Mesoscope-VR acquisition system Graphical User Interface Qt5
    application window.

    This class specializes the Qt5 GUI elements and statically defines the GUI element layout used by the main interface
    window application. The interface enables sl-experiment users to control certain runtime parameters in real time via
    an interactive GUI.
    """

    _data_array: SharedMemoryArray
    _is_paused: bool
    _speed_modifier: int
    _duration_modifier: int
    def __init__(self, data_array: SharedMemoryArray) -> None: ...
    exit_btn: Incomplete
    pause_btn: Incomplete
    runtime_status_label: Incomplete
    valve_open_btn: Incomplete
    valve_close_btn: Incomplete
    reward_btn: Incomplete
    volume_spinbox: Incomplete
    valve_status_label: Incomplete
    speed_spinbox: Incomplete
    duration_spinbox: Incomplete
    def _setup_ui(self) -> None:
        """Creates and arranges all UI elements optimized for Qt5 with proper scaling."""
    def _apply_qt6_styles(self) -> None:
        """Applies optimized styling to all UI elements managed by this class.

        This configured the UI to display properly, assuming the UI window uses the default resolution.
        """
    monitor_timer: Incomplete
    def _setup_monitoring(self) -> None:
        """Sets up a QTimer to monitor the runtime termination status.

        This monitors the value stored under index 0 of the communication SharedMemoryArray and, if the value becomes 1,
        triggers the GUI termination sequence.
        """
    def _check_termination(self) -> None:
        """Checks for the runtime termination signal and, if it has been received, terminates the runtime."""
    def closeEvent(self, event: QCloseEvent | None) -> None:
        """Handles GUI window close events.

        This function is called when the user manually closes the GUI window. This is treated as the request to
        terminate the ongoing runtime.

        Notes:
            Do not call this function manually! It is designed to be used by Qt GUI manager only.

        Args:
            event: The Qt-generated window shutdown event object.
        """
    def _exit_runtime(self) -> None:
        """Signals the runtime to gracefully terminate."""
    def _deliver_reward(self) -> None:
        """Triggers the Mesoscope-VR system to deliver a single water reward to the animal.

        The size of the reward is addressable (configurable) via the reward volume box under the Valve control buttons.
        """
    def _open_valve(self) -> None:
        """Permanently opens the water delivery valve."""
    def _close_valve(self) -> None:
        """Permanently closes the water delivery valve."""
    def _toggle_pause(self) -> None:
        """Toggles the runtime between paused and unpaused (active) states."""
    def _update_reward_volume(self) -> None:
        """Updates the reward volume in the data array in response to user modifying the GUI field value."""
    def _update_speed_modifier(self) -> None:
        """Updates the speed modifier in the data array in response to user modifying the GUI field value."""
    def _update_duration_modifier(self) -> None:
        """Updates the duration modifier in the data array in response to user modifying the GUI field value."""
