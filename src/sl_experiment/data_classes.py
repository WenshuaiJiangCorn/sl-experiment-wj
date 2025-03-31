from ataraxis_data_structures import YamlConfig
from dataclasses import dataclass, field
from pathlib import Path
from ataraxis_time.time_helpers import get_timestamp
import warnings
from ataraxis_base_utilities import ensure_directory_exists, console, LogLevel


@dataclass()
class ProjectConfiguration(YamlConfig):
    """Stores the project-specific configuration parameters that do not change between different animals and runtime
    sessions.

    An instance of this class is generated and saved as a .yaml file in the root directory of the project during the
    first runtime for that project. After that, the stored data is reused for each new runtime session of the project.

    Notes:
        This class allows configuring this library to work for every project in the Sun lab while sharing (and hiding)
        the internal APIs and runtime control functions across all projects. This achieves a streamlined user experience
        (users do not see nor care about inner workings of this library), while supporting project-specific
        customization.
    """

    surgery_sheet_id: str = ""
    """The ID for the Google Sheet file that stores surgery information for the animal whose data is managed by this 
    instance. This is used to parse and write the surgery data for each managed animal into its 'metadata' folder, so 
    that the surgery data is always kept together with the rest of the training and experiment data."""
    water_log_sheet_id: str = ""
    """The ID for the Google Sheet file that stores water restriction information for the animal whose data is managed 
    by this instance. This is used to synchronize the information inside the water restriction log with the state of 
    the animal at the end of each training or experiment runtime.
    """


@dataclass()
class ExperimentState:
    """Encapsulates the information used to set and maintain the Mesoscope-VR state-machine according to the desired
    experiment state.

    Primarily, experiment runtime logic is resolved by the Unity game engine. However, the Mesoscope-VR system
    configuration may need to change throughout the experiment, for example, between the run and rest configurations.
    Overall, the Mesoscope-VR system functions like a state-machine, with multiple statically configured states that
    can be activated and maintained throughout the experiment. During runtime, the main function expects a sequence of
    ExperimentState instances that will be traversed, start-to-end, to determine the flow of the experiment runtime.
    """

    experiment_state_code: int
    """The integer code of the experiment state. Experiment states do not have a predefined meaning, Instead, each 
    project is expected to define and follow its own experiment state code mapping. Typically, the experiment state 
    code is used to denote major experiment stages, such as 'baseline', 'task1', 'cooldown', etc. Note, the same 
    experiment state code can be used by multiple sequential ExperimentState instances to change the VR system states 
    while maintaining the same experiment state."""
    vr_state_code: int
    """One of the supported VR system state-codes. Currently, the Mesoscope-VR system supports 2 state codes. State 
    code 1 denotes 'REST' state and code 2 denotes 'RUN' state. In the rest state, the running wheel is locked and the
    screens are off. In the run state, the screens are on and the wheel is unlocked. Note, multiple consecutive 
    ExperimentState instances with different experiment state codes can reuse the same VR state code."""
    state_duration_s: float
    """The time, in seconds, to maintain the current combination of the experiment and VR states."""


@dataclass()
class ExperimentConfiguration(YamlConfig):
    """Stores the data that describes a single experiment session runtime.

    Primarily, this includes the sequence of experiment and Virtual Reality (Mesoscope-VR) states that define the flow
    of the experiment runtime. Currently, an instance of this class is generated during the first runtime for each
    animal and is saved inside the 'metadata' folder for that animal. it is expected that the user manually adjusts the
    'default' instance as necessary to define the experiment runtime for each animal.
    """

    cue_map: dict[int, float] = field(default_factory=lambda: {0: 30.0, 1: 30.0, 2: 30.0, 3: 30.0, 4: 30.0})
    experiment_states: dict[str, ExperimentState] = field(
        default_factory=lambda: {
            "baseline": ExperimentState(experiment_state_code=1, vr_state_code=1, state_duration_s=30),
            "experiment": ExperimentState(experiment_state_code=2, vr_state_code=2, state_duration_s=120),
            "cooldown": ExperimentState(experiment_state_code=3, vr_state_code=1, state_duration_s=15),
        }
    )


@dataclass()
class RuntimeHardwareConfiguration(YamlConfig):
    """This class is used to save the runtime hardware configuration information as a .yaml file.

    This information is used to read the data saved to the .npz log files during runtime during data processing.

    Notes:
        All fields in this dataclass initialize to None. During log processing, any log associated with a hardware
        module that provides the data stored in a field will be processed, unless that field is None. Therefore, setting
        any field in this dataclass to None also functions as a flag for whether to parse the log associated with the
        module that provides this field's information.

        This class is automatically configured by MesoscopeExperiment and BehaviorTraining classes to facilitate log
        parsing.
    """

    cue_map: dict[int, float] | None = None
    """MesoscopeExperiment instance property."""
    cm_per_pulse: float | None = None
    """EncoderInterface instance property."""
    maximum_break_strength: float | None = None
    """BreakInterface instance property."""
    minimum_break_strength: float | None = None
    """BreakInterface instance property."""
    lick_threshold: int | None = None
    """BreakInterface instance property."""
    scale_coefficient: float | None = None
    """ValveInterface instance property."""
    nonlinearity_exponent: float | None = None
    """ValveInterface instance property."""
    torque_per_adc_unit: float | None = None
    """TorqueInterface instance property."""
    initially_on: bool | None = None
    """ScreenInterface instance property."""
    has_ttl: bool | None = None
    """TTLInterface instance property."""


@dataclass
class SessionData(YamlConfig):
    """Provides methods for managing the data acquired during one experiment or training session.

    This class functions as the central hub for collecting the data from all local PCs involved in the data acquisition
    process and pushing it to the NAS and the BioHPC server. Its primary purpose is to maintain the session data
    structure across all supported destinations and to efficiently and safely move the data to these destinations with
    minimal redundancy and footprint. Additionally, this class generates the paths used by all other classes from
    this library to determine where to load and saved various data during runtime. Finally, it also carries out basic
    data preprocessing to optimize raw data for network transmission and long-term storage.

    As part of its initialization, the class generates the session directory for the input animal and project
    combination. Session directories use the current UTC timestamp, down to microseconds, as the directory name. This
    ensures that each session name is unique and preserves the overall session order.

    Notes:
        Do not call methods from this class directly. This class is intended to be used primarily through the runtime
        logic functions from the experiment.py module and general command-line-interfaces installed with the library.
        The only reason the class is defined as public is to support reconfiguring data destinations and session details
        when implementing custom CLI functions for projects that use this library.

        It is expected that the server, NAS, and mesoscope data directories are mounted on the host-machine via the
        SMB or equivalent protocol. All manipulations with these destinations are carried out with the assumption that
        the OS has full access to these directories and filesystems.

        This class is specifically designed for working with raw data from a single animal participating in a single
        experimental project session. Processed data is managed by the processing library methods and classes.

        This class generates an xxHash-128 checksum stored inside the ax_checksum.txt file at the root of each
        experimental session 'raw_data' directory. The checksum verifies the data of each file and the paths to each
        file relative to the 'raw_data' root directory.
    """

    # Main attributes that are expected to be provided by the user during class initialization
    project_name: str
    """The name of the project for which the data is acquired."""
    animal_id: str
    """The ID code of the animal for which the data is acquired."""
    surgery_sheet_id: str
    """The ID for the Google Sheet file that stores surgery information for the animal whose data is managed by this 
    instance. This is used to parse and write the surgery data for each managed animal into its 'metadata' folder, so 
    that the surgery data is always kept together with the rest of the training and experiment data."""
    water_log_sheet_id: str
    """The ID for the Google Sheet file that stores water restriction information for the animal whose data is managed 
    by this instance. This is used to synchronize the information inside the water restriction log with the state of 
    the animal at the end of each training or experiment runtime.
    """
    session_type: str
    """Stores the type of the session. Primarily, this determines how to read the session_descriptor.yaml file. Has 
    to be set to one of the three supported types: 'lick_training', 'run_training' or 'experiment'.
    """
    credentials_path: str
    """
    The path to the locally stored .JSON file that stores the service account credentials used to read and write Google 
    Sheet data. This is used to access and work with the surgery log and the water restriction log.
    """
    local_root_directory: str
    """The path to the root directory where all projects are stored on the host-machine (VRPC)."""
    server_root_directory: str
    """The path to the root directory where all projects are stored on the BioHPC server machine."""
    nas_root_directory: str
    """The path to the root directory where all projects are stored on the Synology NAS."""
    mesoscope_root_directory: str
    """The path to the root directory used to store all mesoscope-acquired data on the ScanImagePC."""
    session_name: str = "None"
    """Stores the name of the session for which the data is acquired. This name is generated at class initialization 
    based on the current microsecond-accurate timestamp. Do NOT manually provide this name at class initialization.
    Use 'from_path' class method to initialize a SessionData instance for an already existing session data directory.
    """

    def __post_init__(self) -> None:
        """Generates the session name and creates the session directory structure on all involved PCs."""

        # If the session name is provided, ends the runtime early. This is here to support initializing the
        # SessionData class from the path to the root directory of a previous created session.
        if "None" not in self.session_name:
            return

        # Acquires the UTC timestamp to use as the session name
        self.session_name = str(get_timestamp(time_separator="-"))

        # Converts root strings to Path objects.
        local_root_directory = Path(self.local_root_directory)
        server_root_directory = Path(self.server_root_directory)
        nas_root_directory = Path(self.nas_root_directory)
        mesoscope_root_directory = Path(self.mesoscope_root_directory)

        # Constructs the session directory path and generates the directory
        raw_session_path = local_root_directory.joinpath(self.project_name, self.animal_id, self.session_name)

        # Handles potential session name conflicts
        counter = 0
        while raw_session_path.exists():
            counter += 1
            new_session_name = f"{self.session_name}_{counter}"
            raw_session_path = local_root_directory.joinpath(self.project_name, self.animal_id, new_session_name)

        # If a conflict is detected and resolved, warns the user about the resolved conflict.
        if counter > 0:
            message = (
                f"Session name conflict occurred for animal '{self.animal_id}' of project '{self.project_name}' "
                f"when adding the new session with timestamp {self.session_name}. The session with identical name "
                f"already exists. The newly created session directory uses a '_{counter}' postfix to distinguish "
                f"itself from the already existing session directory."
            )
            warnings.warn(message=message)

        # Saves the final session name to class attribute
        self.session_name = raw_session_path.stem

        # Generates the directory structures on all computers used in data management:
        # Raw Data directory and all subdirectories.
        ensure_directory_exists(
            local_root_directory.joinpath(self.project_name, self.animal_id, self.session_name, "raw_data")
        )
        ensure_directory_exists(
            local_root_directory.joinpath(
                self.project_name, self.animal_id, self.session_name, "raw_data", "camera_frames"
            )
        )
        ensure_directory_exists(
            local_root_directory.joinpath(
                self.project_name, self.animal_id, self.session_name, "raw_data", "mesoscope_frames"
            )
        )
        ensure_directory_exists(
            local_root_directory.joinpath(
                self.project_name, self.animal_id, self.session_name, "raw_data", "behavior_data_log"
            )
        )

        ensure_directory_exists(local_root_directory.joinpath(self.project_name, self.animal_id, "persistent_data"))
        ensure_directory_exists(nas_root_directory.joinpath(self.project_name, self.animal_id, self.session_name))
        ensure_directory_exists(server_root_directory.joinpath(self.project_name, self.animal_id, self.session_name))
        ensure_directory_exists(local_root_directory.joinpath(self.project_name, self.animal_id, "metadata"))
        ensure_directory_exists(server_root_directory.joinpath(self.project_name, self.animal_id, "metadata"))
        ensure_directory_exists(nas_root_directory.joinpath(self.project_name, self.animal_id, "metadata"))
        ensure_directory_exists(mesoscope_root_directory.joinpath("mesoscope_frames"))
        ensure_directory_exists(mesoscope_root_directory.joinpath("persistent_data", self.project_name, self.animal_id))

    @classmethod
    def from_path(cls, path: Path) -> "SessionData":
        """Initializes a SessionData instance to represent the data of an already existing session.

        Typically, this initialization mode is used to preprocess an interrupted session. This method uses the cached
        data stored in the 'session_data.yaml' file in the 'raw_data' subdirectory of the provided session directory.

        Args:
            path: The path to the session directory on the local (VRPC) machine.

        Returns:
            An initialized SessionData instance for the session whose data is stored at the provided path.

        Raises:
            FileNotFoundError: If the 'session_data.yaml' file is not found after resolving the provided path.
        """
        path = path.joinpath("raw_data", "session_data.yaml")

        if not path.exists():
            message = (
                f"No 'session_data.yaml' file found at the provided path: {path}. Unable to preprocess the target "
                f"session, as session_data.yaml is required to run preprocessing. This likely indicates that the "
                f"session runtime was interrupted before recording any data, as the session_data.yaml snapshot is "
                f"generated very early in the session runtime."
            )
            console.error(message=message, error=FileNotFoundError)

        return cls.from_yaml(file_path=path)  # type: ignore

    def to_path(self) -> None:
        """Saves the data of the instance to the 'raw_data' directory of the managed session as a 'session_data.yaml'
        file.

        This is used to save the data stored in the instance to disk, so that it can be reused during preprocessing or
        data processing. This also serves as the repository for the identification information about the project,
        animal, and session that generated the data.
        """
        self.to_yaml(file_path=self.raw_data_path.joinpath("session_data.yaml"))

    @property
    def raw_data_path(self) -> Path:
        """Returns the path to the 'raw_data' directory of the managed session on the VRPC.

        This directory functions as the root directory that stores all raw data acquired during training or experiment
        runtime for a given session.
        """
        local_root_directory = Path(self.local_root_directory)
        return local_root_directory.joinpath(self.project_name, self.animal_id, self.session_name, "raw_data")

    @property
    def camera_frames_path(self) -> Path:
        """Returns the path to the 'camera_frames' directory of the managed session.

        This subdirectory is stored under the 'raw_data' directory and aggregates all video camera data.
        """
        return self.raw_data_path.joinpath("camera_frames")

    @property
    def zaber_positions_path(self) -> Path:
        """Returns the path to the 'zaber_positions.yaml' file of the managed session.

        This path is used to save the positions for all Zaber motors of the HeadBar and LickPort controllers at the
        end of the experimental session.
        """
        return self.raw_data_path.joinpath("zaber_positions.yaml")

    @property
    def session_descriptor_path(self) -> Path:
        """Returns the path to the 'session_descriptor.yaml' file of the managed session.

        This path is used to save important session information to be viewed by experimenters post-runtime and to use
        for further processing.
        """
        return self.raw_data_path.joinpath("session_descriptor.yaml")

    @property
    def hardware_configuration_path(self) -> Path:
        """Returns the path to the 'hardware_configuration.yaml' file of the managed session.

        This file stores hardware module parameters used to read and parse .npz log files during data processing.
        """
        return self.raw_data_path.joinpath("hardware_configuration.yaml")

    @property
    def previous_zaber_positions_path(self) -> Path:
        """Returns the path to the 'zaber_positions.yaml' file of the previous session.

        The file is stored inside the 'persistent_data' directory of the managed animal.
        """
        local_root_directory = Path(self.local_root_directory)
        return local_root_directory.joinpath(
            self.project_name, self.animal_id, "persistent_data", "zaber_positions.yaml"
        )

    @property
    def mesoscope_root_path(self) -> Path:
        """Returns the path to the root directory of the Mesoscope pc (ScanImagePC) used to store all
        mesoscope-acquired data.
        """
        return Path(self.mesoscope_root_directory)

    @property
    def nas_root_path(self) -> Path:
        """Returns the path to the root directory of the Synology NAS (Network Attached Storage) used to store all
        training and experiment data after preprocessing (backup cold long-term storage)."""
        return Path(self.nas_root_directory)

    @property
    def server_root_path(self) -> Path:
        """Returns the path to the root directory of the BioHPC server used to process and store all training and e
        experiment data (main long-term storage)."""
        return Path(self.server_root_directory)

    @property
    def mesoscope_persistent_path(self) -> Path:
        """Returns the path to the 'persistent_data' directory of the Mesoscope pc (ScanImagePC).

        This directory is primarily used to store the reference MotionEstimator.me files for each animal.
        """
        return self.mesoscope_root_path.joinpath("persistent_data", self.project_name, self.animal_id)

    @property
    def local_metadata_path(self) -> Path:
        """Returns the path to the 'metadata' directory of the managed animal on the VRPC."""
        local_root_directory = Path(self.local_root_directory)
        return local_root_directory.joinpath(self.project_name, self.animal_id, "metadata")

    @property
    def server_metadata_path(self) -> Path:
        """Returns the path to the 'metadata' directory of the managed animal on the BioHPC server."""
        return self.server_root_path.joinpath(self.project_name, self.animal_id, "metadata")

    @property
    def nas_metadata_path(self) -> Path:
        """Returns the path to the 'metadata' directory of the managed animal on the Synology NAS."""
        return self.nas_root_path.joinpath(self.project_name, self.animal_id, "metadata")


@dataclass()
class LickTrainingDescriptor(YamlConfig):
    """This class is used to save the description information specific to lick training sessions as a .yaml file."""

    experimenter: str
    """The ID of the experimenter running the session."""
    mouse_weight_g: float
    """The weight of the animal, in grams, at the beginning of the session."""
    dispensed_water_volume_ml: float = 0.0
    """Stores the total water volume, in milliliters, dispensed during runtime."""
    average_reward_delay_s: int = 12
    """Stores the center-point for the reward delay distribution, in seconds."""
    maximum_deviation_from_average_s: int = 6
    """Stores the deviation value, in seconds, used to determine the upper and lower bounds for the reward delay 
    distribution."""
    maximum_water_volume_ml: float = 1.0
    """Stores the maximum volume of water the system is allowed to dispense during training."""
    maximum_training_time_m: int = 40
    """Stores the maximum time, in minutes, the system is allowed to run the training for."""
    experimenter_notes: str = "Replace this with your notes."
    """This field is not set during runtime. It is expected that each experimenter will replace this field with their 
    notes made during runtime."""
    experimenter_given_water_volume_ml: float = 0.0
    """The additional volume of water, in milliliters, administered by the experimenter to the animal after the session.
    """


@dataclass()
class RunTrainingDescriptor(YamlConfig):
    """This class is used to save the description information specific to run training sessions as a .yaml file."""

    experimenter: str
    """The ID of the experimenter running the session."""
    mouse_weight_g: float
    """The weight of the animal, in grams, at the beginning of the session."""
    dispensed_water_volume_ml: float = 0.0
    """Stores the total water volume, in milliliters, dispensed during runtime."""
    final_running_speed_cm_s: float = 0.0
    """Stores the final running speed threshold that was active at the end of training."""
    final_speed_duration_s: float = 0.0
    """Stores the final running duration threshold that was active at the end of training."""
    initial_running_speed_cm_s: float = 0.0
    """Stores the initial running speed threshold, in centimeters per second, used during training."""
    initial_speed_duration_s: float = 0.0
    """Stores the initial above-threshold running duration, in seconds, used during training."""
    increase_threshold_ml: float = 0.0
    """Stores the volume of water delivered to the animal, in milliliters, that triggers the increase in the running 
    speed and duration thresholds."""
    increase_running_speed_cm_s: float = 0.0
    """Stores the value, in centimeters per second, used by the system to increment the running speed threshold each 
    time the animal receives 'increase_threshold' volume of water."""
    increase_speed_duration_s: float = 0.0
    """Stores the value, in seconds, used by the system to increment the duration threshold each time the animal 
    receives 'increase_threshold' volume of water."""
    maximum_running_speed_cm_s: float = 0.0
    """Stores the maximum running speed threshold, in centimeters per second, the system is allowed to use during 
    training."""
    maximum_speed_duration_s: float = 0.0
    """Stores the maximum above-threshold running duration, in seconds, the system is allowed to use during training."""
    maximum_water_volume_ml: float = 1.0
    """Stores the maximum volume of water the system is allowed to dispensed during training."""
    maximum_training_time_m: int = 40
    """Stores the maximum time, in minutes, the system is allowed to run the training for."""
    experimenter_notes: str = "Replace this with your notes."
    """This field is not set during runtime. It is expected that each experimenter will replace this field with their 
    notes made during runtime."""
    experimenter_given_water_volume_ml: float = 0.0
    """The additional volume of water, in milliliters, administered by the experimenter to the animal after the session.
    """


@dataclass()
class MesoscopeExperimentDescriptor(YamlConfig):
    """This class is used to save the description information specific to experiment sessions as a .yaml file."""

    experimenter: str
    """The ID of the experimenter running the session."""
    mouse_weight_g: float
    """The weight of the animal, in grams, at the beginning of the session."""
    dispensed_water_volume_ml: float = 0.0
    """Stores the total water volume, in milliliters, dispensed during runtime."""
    experimenter_notes: str = "Replace this with your notes."
    """This field is not set during runtime. It is expected that each experimenter will replace this field with their 
    notes made during runtime."""
    experimenter_given_water_volume_ml: float = 0.0
    """The additional volume of water, in milliliters, administered by the experimenter to the animal after the session.
    """


import sys

y = ExperimentConfiguration()
opath = Path("/home/cybermouse/Desktop/test/out.yaml")
# y.to_yaml(file_path=opath)
x = ExperimentConfiguration.from_yaml(file_path=opath)
print(x)
sys.exit(111)
