"""This module provides additional tools and classes used by other modules of the mesoscope_vr package. Primarily, this
includes various dataclasses specific to the Mesoscope-VR systems and utility functions used by other package modules.
The contents of this module are not intended to be used outside the mesoscope_vr package."""

import sys
from pathlib import Path
from dataclasses import field, dataclass
from multiprocessing import Process

import numpy as np
from PyQt6.QtGui import QFont, QCloseEvent
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QLabel,
    QWidget,
    QGroupBox,
    QHBoxLayout,
    QMainWindow,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QApplication,
    QDoubleSpinBox,
)
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

    def __init__(self) -> None:
        self._data_array = SharedMemoryArray.create_array(
            name="runtime_control_ui", prototype=np.zeros(shape=12, dtype=np.int32), exist_ok=True
        )

        # Starts the UI process
        self._ui_process = Process(target=self._run_ui_process, daemon=True)
        self._ui_process.start()
        self._started = True

    def __del__(self) -> None:
        """Ensures all class resources are released before the instance is destroyed.

        This is a fallback method, using shutdown() directly is the preferred way of releasing resources.
        """
        if self._started:
            self.shutdown()

    def shutdown(self) -> None:
        """Shuts down the UI and releases all SharedMemoryArray resources.

        This method should be called at the end of runtime to properly release all resources and terminate the
        remote UI process.
        """
        # If the UI process is still alive, shuts it down
        if self._ui_process.is_alive():
            self._data_array.write_data(index=0, data=np.int32(1))  # Sends the termination signal to the remote process
            self._ui_process.terminate()
            self._ui_process.join(timeout=2.0)  # Waits for at most 2 seconds to terminate the process gracefully

        # Destroys the SharedMemoryArray
        self._data_array.disconnect()
        self._data_array.destroy()

        # Toggles the flag
        self._started = False

    def _run_ui_process(self) -> None:
        """The main function that runs in the parallel process to display and manage the Qt5 UI.

        This runs Qt5 in the main thread of the separate process, which is perfectly valid.
        """

        # Connects to the shared memory array from the remote process
        self._data_array.connect()

        # Create and run the Qt5 application in this process's main thread
        try:
            # Creates the QT5 GUI application
            app = QApplication(sys.argv)
            app.setApplicationName("Mesoscope-VR Control Panel")
            app.setOrganizationName("SunLab")

            # Sets Qt5 application-wide style
            app.setStyle("Fusion")  # Modern flat style available in Qt5

            # Creates the main application window
            window = _ControlUIWindow(self._data_array)
            window.show()

            # Runs the app
            app.exec()

        # Terminates with an exception which will be propagated to the main process
        except Exception as e:
            message = f"Unable to initialize the QT5 GUI application. Encountered the following error {e}."
            console.error(message=message, error=RuntimeError)

        # Ensures proper UI shutdown when runtime encounters errors
        finally:
            self._data_array.disconnect()

    @property
    def exit_signal(self) -> bool:
        """Returns True if the user has requested the runtime to gracefully abort.

        Notes:
            Each time this property is accessed, the flag is reset to 0.
        """
        exit_flag = bool(self._data_array.read_data(index=1, convert_output=True))
        self._data_array.write_data(index=1, data=np.int32(0))
        return exit_flag

    @property
    def reward_signal(self) -> bool:
        """Returns True if the user has requested the system to deliver a water reward.

        Notes:
            Each time this property is accessed, the flag is reset to 0.
        """
        reward_flag = bool(self._data_array.read_data(index=2, convert_output=True))
        self._data_array.write_data(index=2, data=np.int32(0))
        return reward_flag

    @property
    def speed_modifier(self) -> int:
        """Returns the current user-defined modifier to apply to the running speed threshold."""
        return int(self._data_array.read_data(index=3, convert_output=True))

    @property
    def duration_modifier(self) -> int:
        """Returns the current user-defined modifier to apply to the running epoch duration threshold."""
        return int(self._data_array.read_data(index=4, convert_output=True))

    @property
    def pause_runtime(self) -> bool:
        """Returns True if the user has requested the acquisition system to pause the current runtime.

        Notes:
            Unlike most other flags,t he state of this flag does NOT change when it is accessed. Instead, the UI flips
            it between True and False to pause and resume the managed runtime.
        """
        return bool(self._data_array.read_data(index=5, convert_output=True))

    @property
    def open_valve(self) -> bool:
        """Returns True if the user has requested the acquisition system to permanently open the water delivery
        valve.

        Notes:
            Each time this property is accessed, the flag is reset to 0.
        """
        open_flag = bool(self._data_array.read_data(index=6, convert_output=True))
        self._data_array.write_data(index=6, data=np.int32(0))
        return open_flag

    @property
    def close_valve(self) -> bool:
        """Returns True if the user has requested the acquisition system to permanently close the water delivery
        valve.

        Notes:
            Each time this property is accessed, the flag is reset to 0.
        """
        close_flag = bool(self._data_array.read_data(index=7, convert_output=True))
        self._data_array.write_data(index=7, data=np.int32(0))
        return close_flag

    @property
    def reward_volume(self) -> int:
        """Returns the current user-defined water reward volume value.

        Querying this property allows dynamically configuring the Mesoscope-VR system to use the user-defined volume of
        water each time it rewards the animal.
        """
        return int(self._data_array.read_data(index=8, convert_output=True))


class _ControlUIWindow(QMainWindow):
    """Generates, renders, and maintains the main Mesoscope-VR acquisition system Graphical User Interface Qt5
    application window.

    This class specializes the Qt5 GUI elements and statically defines the GUI element layout used by the main interface
    window application. The interface enables sl-experiment users to control certain runtime parameters in real time via
    an interactive GUI.
    """

    def __init__(self, data_array: SharedMemoryArray):
        super().__init__()  # Initializes the main window superclass

        # Defines internal attributes.
        self._data_array: SharedMemoryArray = data_array
        self._is_paused: bool = True
        self._speed_modifier: int = 0
        self._duration_modifier: int = 0

        # Configures the window title
        self.setWindowTitle("Mesoscope-VR Control Panel")

        # Uses fixed size
        self.setFixedSize(450, 500)

        # Sets up the interactive UI
        self._setup_ui()
        self._setup_monitoring()

        # Applies Qt5-optimized styling and scaling parameters
        self._apply_qt6_styles()

    def _setup_ui(self) -> None:
        """Creates and arranges all UI elements optimized for Qt5 with proper scaling."""

        # Initializes the main widget container
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Generates the central bounding box (the bounding box around all UI elements)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(12)
        main_layout.setContentsMargins(15, 15, 15, 15)

        # Runtime Control Group
        runtime_control_group = QGroupBox("Runtime Control")
        runtime_control_layout = QVBoxLayout(runtime_control_group)
        runtime_control_layout.setSpacing(6)

        # Runtime termination (exit) button
        self.exit_btn = QPushButton("✖ Terminate Runtime")
        self.exit_btn.setToolTip("Gracefully ends the runtime and initiates the shutdown procedure.")
        # noinspection PyUnresolvedReferences
        self.exit_btn.clicked.connect(self._exit_runtime)
        self.exit_btn.setObjectName("exitButton")

        # Runtime Pause / Unpause (resume) button
        self._data_array.write_data(index=5, data=np.int32(1))  # Ensures all runtimes start in a paused state
        self.pause_btn = QPushButton("▶️ Resume Runtime")
        self.pause_btn.setToolTip("Pauses or resumes the runtime.")
        # noinspection PyUnresolvedReferences
        self.pause_btn.clicked.connect(self._toggle_pause)
        self.pause_btn.setObjectName("resumeButton")

        # Configures the buttons to expand when UI is resized, but use a fixed height of 35 points
        for btn in [self.exit_btn, self.pause_btn]:
            btn.setMinimumHeight(35)
            btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            runtime_control_layout.addWidget(btn)

        # Adds runtime status tracker to the same box
        self.runtime_status_label = QLabel("Runtime Status: ⏸️ Paused")
        self.runtime_status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        runtime_status_font = QFont()
        runtime_status_font.setPointSize(35)
        runtime_status_font.setBold(True)
        self.runtime_status_label.setFont(runtime_status_font)
        self.runtime_status_label.setStyleSheet("QLabel { color: #f39c12; font-weight: bold; }")
        runtime_control_layout.addWidget(self.runtime_status_label)

        # Adds the runtime control box to the UI widget
        main_layout.addWidget(runtime_control_group)

        # Valve Control Group
        valve_group = QGroupBox("Valve Control")
        valve_layout = QVBoxLayout(valve_group)
        valve_layout.setSpacing(6)

        # Arranges valve control buttons in a horizontal layout
        valve_buttons_layout = QHBoxLayout()

        # Valve open
        self.valve_open_btn = QPushButton("🔓 Open")
        self.valve_open_btn.setToolTip("Opens the solenoid valve.")
        # noinspection PyUnresolvedReferences
        self.valve_open_btn.clicked.connect(self._open_valve)
        self.valve_open_btn.setObjectName("valveOpenButton")

        # Valve close
        self.valve_close_btn = QPushButton("🔒 Close")
        self.valve_close_btn.setToolTip("Closes the solenoid valve.")
        # noinspection PyUnresolvedReferences
        self.valve_close_btn.clicked.connect(self._close_valve)
        self.valve_close_btn.setObjectName("valveCloseButton")

        # Reward button
        self.reward_btn = QPushButton("● Reward")
        self.reward_btn.setToolTip("Delivers 5 uL of water through the solenoid valve.")
        # noinspection PyUnresolvedReferences
        self.reward_btn.clicked.connect(self._deliver_reward)
        self.reward_btn.setObjectName("rewardButton")

        # Configures the buttons to expand when UI is resized, but use a fixed height of 35 points
        for btn in [self.valve_open_btn, self.valve_close_btn, self.reward_btn]:
            btn.setMinimumHeight(35)
            btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            valve_buttons_layout.addWidget(btn)

        valve_layout.addLayout(valve_buttons_layout)

        # Valve status and volume control section - horizontal layout
        valve_status_layout = QHBoxLayout()
        valve_status_layout.setSpacing(6)

        # Volume control on the left
        volume_label = QLabel("Reward volume:")
        volume_label.setObjectName("volumeLabel")

        # Ensures the array is also set to the default value
        self._data_array.write_data(index=8, data=np.int32(5))

        self.volume_spinbox = QDoubleSpinBox()
        self.volume_spinbox.setRange(1, 20)  # Ranges from 1 to 20
        self.volume_spinbox.setValue(5)  # Default value
        self.volume_spinbox.setDecimals(0)  # Integer precision
        self.volume_spinbox.setSuffix(" μL")  # Adds units suffix
        self.volume_spinbox.setToolTip("Sets water reward volume. Accepts values between 1 and 2 μL.")
        self.volume_spinbox.setMinimumHeight(30)
        # noinspection PyUnresolvedReferences
        self.volume_spinbox.valueChanged.connect(self._update_reward_volume)

        # Adds volume controls to left side
        valve_status_layout.addWidget(volume_label)
        valve_status_layout.addWidget(self.volume_spinbox)

        # Adds the valve status tracker on the right
        self.valve_status_label = QLabel("Valve: 🔒 Closed")
        self.valve_status_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        valve_status_font = QFont()
        valve_status_font.setPointSize(35)
        valve_status_font.setBold(True)
        self.valve_status_label.setFont(valve_status_font)
        self.valve_status_label.setStyleSheet("QLabel { color: #e67e22; font-weight: bold; }")
        valve_status_layout.addWidget(self.valve_status_label)

        # Add the horizontal status layout to the main valve layout
        valve_layout.addLayout(valve_status_layout)

        # Adds the valve control box to the UI widget
        main_layout.addWidget(valve_group)

        # Adds Run Training controls in a horizontal layout
        controls_layout = QHBoxLayout()
        controls_layout.setSpacing(6)

        # Running Speed Threshold Control Group
        speed_group = QGroupBox("Speed Threshold")
        speed_layout = QVBoxLayout(speed_group)

        # Speed Modifier
        speed_status_label = QLabel("Current Modifier:")
        speed_status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        speed_status_label.setStyleSheet("QLabel { font-weight: bold; color: #34495e; }")
        speed_layout.addWidget(speed_status_label)
        self.speed_spinbox = QDoubleSpinBox()
        self.speed_spinbox.setRange(-1000, 1000)  # Factoring in the step of 0.01, this allows -20 to +20 cm/s
        self.speed_spinbox.setValue(self._speed_modifier)  # Default value
        self.speed_spinbox.setDecimals(0)  # Integer precision
        self.speed_spinbox.setToolTip("Sets the running speed threshold modifier value.")
        self.speed_spinbox.setMinimumHeight(30)
        # noinspection PyUnresolvedReferences
        self.speed_spinbox.valueChanged.connect(self._update_speed_modifier)
        speed_layout.addWidget(self.speed_spinbox)

        # Running Duration Threshold Control Group
        duration_group = QGroupBox("Duration Threshold")
        duration_layout = QVBoxLayout(duration_group)

        # Duration modifier
        duration_status_label = QLabel("Current Modifier:")
        duration_status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        duration_status_label.setStyleSheet("QLabel { font-weight: bold; color: #34495e; }")
        duration_layout.addWidget(duration_status_label)
        self.duration_spinbox = QDoubleSpinBox()
        self.duration_spinbox.setRange(-1000, 1000)  # Factoring in the step of 0.01, this allows -20 to +20 s
        self.duration_spinbox.setValue(self._duration_modifier)  # Default value
        self.duration_spinbox.setDecimals(0)  # Integer precision
        self.duration_spinbox.setToolTip("Sets the running duration threshold modifier value.")
        # noinspection PyUnresolvedReferences
        self.duration_spinbox.valueChanged.connect(self._update_duration_modifier)
        duration_layout.addWidget(self.duration_spinbox)

        # Adds speed and duration threshold modifiers to the main UI widget
        controls_layout.addWidget(speed_group)
        controls_layout.addWidget(duration_group)
        main_layout.addLayout(controls_layout)

    def _apply_qt6_styles(self) -> None:
        """Applies optimized styling to all UI elements managed by this class.

        This configured the UI to display properly, assuming the UI window uses the default resolution.
        """

        self.setStyleSheet(f"""
                    QMainWindow {{
                        background-color: #ecf0f1;
                    }}

                    QGroupBox {{
                        font-weight: bold;
                        font-size: 14pt;
                        border: 2px solid #bdc3c7;
                        border-radius: 8px;
                        margin: 25px 6px 6px 6px;
                        padding-top: 10px;
                        background-color: #ffffff;
                    }}

                    QGroupBox::title {{
                        subcontrol-origin: margin;
                        subcontrol-position: top center;
                        left: 0px;
                        right: 0px;
                        padding: 0 8px 0 8px;
                        color: #2c3e50;
                        background-color: transparent;
                        border: none;
                    }}

                    QPushButton {{
                        background-color: #ffffff;
                        border: 2px solid #bdc3c7;
                        border-radius: 6px;
                        padding: 6px 8px;
                        font-size: 12pt;
                        font-weight: 500;
                        color: #2c3e50;
                        min-height: 20px;
                    }}

                    QPushButton:hover {{
                        background-color: #f8f9fa;
                        border-color: #3498db;
                        color: #2980b9;
                    }}

                    QPushButton:pressed {{
                        background-color: #e9ecef;
                        border-color: #2980b9;
                    }}

                    QPushButton#exitButton {{
                        background-color: #e74c3c;
                        color: white;
                        border-color: #c0392b;
                        font-weight: bold;
                    }}

                    QPushButton#exitButton:hover {{
                        background-color: #c0392b;
                        border-color: #a93226;
                    }}

                    QPushButton#pauseButton {{
                        background-color: #f39c12;
                        color: white;
                        border-color: #e67e22;
                        font-weight: bold;
                    }}

                    QPushButton#pauseButton:hover {{
                        background-color: #e67e22;
                        border-color: #d35400;
                    }}

                    QPushButton#resumeButton {{
                        background-color: #27ae60;
                        color: white;
                        border-color: #229954;
                        font-weight: bold;
                    }}

                    QPushButton#resumeButton:hover {{
                        background-color: #229954;
                        border-color: #1e8449;
                    }}

                    QPushButton#valveOpenButton {{
                        background-color: #27ae60;
                        color: white;
                        border-color: #229954;
                        font-weight: bold;
                    }}

                    QPushButton#valveOpenButton:hover {{
                        background-color: #229954;
                        border-color: #1e8449;
                    }}

                    QPushButton#valveCloseButton {{
                        background-color: #e67e22;
                        color: white;
                        border-color: #d35400;
                        font-weight: bold;
                    }}

                    QPushButton#valveCloseButton:hover {{
                        background-color: #d35400;
                        border-color: #ba4a00;
                    }}
                    
                    QPushButton#rewardButton {{
                        background-color: #3498db;
                        color: white;
                        border-color: #2980b9;
                        font-weight: bold;
                    }}

                    QPushButton#rewardButton:hover {{
                        background-color: #2980b9;
                        border-color: #21618c;
                    }}

                    QLabel {{
                        color: #2c3e50;
                        font-size: 12pt;
                    }}
                    
                    QLabel#volumeLabel {{
                        color: #2c3e50;
                        font-size: 12pt;
                        font-weight: bold;
                    }}
    
                    QDoubleSpinBox {{
                        border: 2px solid #bdc3c7;
                        border-radius: 4px;
                        padding: 4px 8px;
                        font-weight: bold;
                        font-size: 12pt;
                        background-color: white;
                        color: #2c3e50;
                        min-height: 20px;
                    }}
    
                    QDoubleSpinBox:focus {{
                        border-color: #3498db;
                    }}
    
                    QDoubleSpinBox::up-button {{
                        subcontrol-origin: border;
                        subcontrol-position: top right;
                        width: 20px;
                        background-color: #f8f9fa;
                        border: 1px solid #bdc3c7;
                        border-top-right-radius: 4px;
                        border-bottom: none;
                    }}
    
                    QDoubleSpinBox::up-button:hover {{
                        background-color: #e9ecef;
                        border-color: #3498db;
                    }}
    
                    QDoubleSpinBox::up-button:pressed {{
                        background-color: #dee2e6;
                    }}
    
                    QDoubleSpinBox::up-arrow {{
                        image: none;
                        border-left: 4px solid transparent;
                        border-right: 4px solid transparent;
                        border-bottom: 6px solid #2c3e50;
                        width: 0px;
                        height: 0px;
                    }}
    
                    QDoubleSpinBox::down-button {{
                        subcontrol-origin: border;
                        subcontrol-position: bottom right;
                        width: 20px;
                        background-color: #f8f9fa;
                        border: 1px solid #bdc3c7;
                        border-bottom-right-radius: 4px;
                        border-top: none;
                    }}
    
                    QDoubleSpinBox::down-button:hover {{
                        background-color: #e9ecef;
                        border-color: #3498db;
                    }}
    
                    QDoubleSpinBox::down-button:pressed {{
                        background-color: #dee2e6;
                    }}
    
                    QDoubleSpinBox::down-arrow {{
                        image: none;
                        border-left: 4px solid transparent;
                        border-right: 4px solid transparent;
                        border-top: 6px solid #2c3e50;
                        width: 0px;
                        height: 0px;
                    }}
    
                    QSlider::groove:horizontal {{
                        border: 1px solid #bdc3c7;
                        height: 8px;
                        background: #ecf0f1;
                        margin: 2px 0;
                        border-radius: 4px;
                    }}
    
                    QSlider::handle:horizontal {{
                        background: #3498db;
                        border: 2px solid #2980b9;
                        width: 20px;
                        margin: -6px 0;
                        border-radius: 10px;
                    }}
    
                    QSlider::handle:horizontal:hover {{
                        background: #2980b9;
                        border-color: #21618c;
                    }}
    
                    QSlider::handle:horizontal:pressed {{
                        background: #21618c;
                    }}
    
                    QSlider::sub-page:horizontal {{
                        background: #3498db;
                        border: 1px solid #2980b9;
                        height: 8px;
                        border-radius: 4px;
                    }}
    
                    QSlider::add-page:horizontal {{
                        background: #ecf0f1;
                        border: 1px solid #bdc3c7;
                        height: 8px;
                        border-radius: 4px;
                    }}
    
                    QSlider::groove:vertical {{
                        border: 1px solid #bdc3c7;
                        width: 8px;
                        background: #ecf0f1;
                        margin: 0 2px;
                        border-radius: 4px;
                    }}
    
                    QSlider::handle:vertical {{
                        background: #3498db;
                        border: 2px solid #2980b9;
                        height: 20px;
                        margin: 0 -6px;
                        border-radius: 10px;
                    }}
    
                    QSlider::handle:vertical:hover {{
                        background: #2980b9;
                        border-color: #21618c;
                    }}
    
                    QSlider::handle:vertical:pressed {{
                        background: #21618c;
                    }}
    
                    QSlider::sub-page:vertical {{
                        background: #ecf0f1;
                        border: 1px solid #bdc3c7;
                        width: 8px;
                        border-radius: 4px;
                    }}
    
                    QSlider::add-page:vertical {{
                        background: #3498db;
                        border: 1px solid #2980b9;
                        width: 8px;
                        border-radius: 4px;
                    }}
                """)

    def _setup_monitoring(self) -> None:
        """Sets up a QTimer to monitor the runtime termination status.

        This monitors the value stored under index 0 of the communication SharedMemoryArray and, if the value becomes 1,
        triggers the GUI termination sequence.
        """
        self.monitor_timer = QTimer(self)
        # noinspection PyUnresolvedReferences
        self.monitor_timer.timeout.connect(self._check_termination)
        self.monitor_timer.start(100)  # Checks every 100 ms

    def _check_termination(self) -> None:
        """Checks for the runtime termination signal and, if it has been received, terminates the runtime."""
        try:
            if self._data_array.read_data(index=0, convert_output=True) == 1:
                self.close()
        except:
            self.close()

    def closeEvent(self, event: QCloseEvent) -> None:
        """Handles GUI window close events.

        This function is called when the user manually closes the GUI window. This is treated as the request to
        terminate the ongoing runtime.

        Notes:
            Do not call this function manually! It is designed to be used by Qt GUI manager only.

        Args:
            event: The Qt-generated window shutdown event object.
        """
        # Sends runtime termination signal via the SharedMemoryArray before accepting the close event.
        try:
            self._data_array.write_data(index=0, data=np.int32(1))
        except:
            pass
        event.accept()

    def _exit_runtime(self) -> None:
        """Signals the runtime to gracefully terminate."""
        previous_status = self.runtime_status_label.text()
        style = self.runtime_status_label.styleSheet()
        self._data_array.write_data(index=1, data=np.int32(1))
        self.runtime_status_label.setText("✖ Exit signal sent")
        self.runtime_status_label.setStyleSheet("QLabel { color: #e74c3c; font-weight: bold; }")
        self.exit_btn.setText("✖ Exit Requested")
        self.exit_btn.setEnabled(False)

        # Resets the button after 2 seconds
        QTimer.singleShot(2000, lambda: self.exit_btn.setText("✖ Terminate Runtime"))
        QTimer.singleShot(2000, lambda: self.exit_btn.setStyleSheet("QLabel { color: #c0392b; font-weight: bold; }"))
        QTimer.singleShot(2000, lambda: self.exit_btn.setEnabled(True))

        # Restores the status back to the previous state
        QTimer.singleShot(2000, lambda: self.runtime_status_label.setText(previous_status))
        QTimer.singleShot(2000, lambda: self.runtime_status_label.setStyleSheet(style))

    def _deliver_reward(self) -> None:
        """Triggers the Mesoscope-VR system to deliver a single water reward to the animal.

        The size of the reward is addressable (configurable) via the reward volume box under the Valve control buttons.
        """
        # Sends the reward command via the SharedMemoryArray and temporarily sets the statsu to indicate that the
        # reward is sent.
        self._data_array.write_data(index=2, data=np.int32(1))
        self.valve_status_label.setText("Reward: 🟢 Sent")
        self.valve_status_label.setStyleSheet("QLabel { color: #3498db; font-weight: bold; }")

        # Resets the status to 'closed' after 1 second using Qt5 single shot timer. This is realistically the longest
        # time the system would take to start and finish delivering the reward
        QTimer.singleShot(2000, lambda: self.valve_status_label.setText("Valve: 🔒 Closed"))
        QTimer.singleShot(
            2000, lambda: self.valve_status_label.setStyleSheet("QLabel { color: #e67e22; font-weight: bold; }")
        )

    def _open_valve(self) -> None:
        """Permanently opens the water delivery valve."""
        self._data_array.write_data(index=6, data=np.int32(1))
        self.valve_status_label.setText("Valve: 🔓 Opened")
        self.valve_status_label.setStyleSheet("QLabel { color: #27ae60; font-weight: bold; }")

    def _close_valve(self) -> None:
        """Permanently closes the water delivery valve."""
        self._data_array.write_data(index=7, data=np.int32(1))
        self.valve_status_label.setText("Valve: 🔒 Closed")
        self.valve_status_label.setStyleSheet("QLabel { color: #e67e22; font-weight: bold; }")

    def _toggle_pause(self) -> None:
        """Toggles the runtime between paused and unpaused (active) states."""
        self._is_paused = not self._is_paused
        if self._is_paused:
            self._data_array.write_data(index=5, data=np.int32(1))
            self.pause_btn.setText("▶️ Resume Runtime")
            self.pause_btn.setObjectName("resumeButton")
            self.runtime_status_label.setText("Runtime Status: ⏸️ Paused")
            self.runtime_status_label.setStyleSheet("QLabel { color: #f39c12; font-weight: bold; }")
        else:
            self._data_array.write_data(index=5, data=np.int32(0))
            self.pause_btn.setText("⏸️ Pause Runtime")
            self.pause_btn.setObjectName("pauseButton")
            self.runtime_status_label.setText("Runtime Status: 🟢 Running")
            self.runtime_status_label.setStyleSheet("QLabel { color: #27ae60; font-weight: bold; }")

        # Refreshes styles after object name change
        self.pause_btn.style().unpolish(self.pause_btn)
        self.pause_btn.style().polish(self.pause_btn)
        self.pause_btn.update()  # Force update to apply new styles

    def _update_reward_volume(self) -> None:
        """Updates the reward volume in the data array in response to user modifying the GUI field value."""
        volume = int(self.volume_spinbox.value())
        self._data_array.write_data(index=8, data=np.int32(volume))

    def _update_speed_modifier(self) -> None:
        """Updates the speed modifier in the data array in response to user modifying the GUI field value."""
        self._speed_modifier = int(self.speed_spinbox.value())
        self._data_array.write_data(index=3, data=np.int32(self._speed_modifier))

    def _update_duration_modifier(self) -> None:
        """Updates the duration modifier in the data array in response to user modifying the GUI field value."""
        self._duration_modifier = int(self.duration_spinbox.value())
        self._data_array.write_data(index=4, data=np.int32(self._duration_modifier))
