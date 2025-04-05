from pathlib import Path
from dataclasses import field, dataclass

from _typeshed import Incomplete
from ataraxis_data_structures import YamlConfig

def replace_root_path(path: Path) -> None:
    """Replaces the path to the local root directory saved in the library's static configuration file with the
    provided path.

    When the library is used for the first time, it asks the user to provide the path to the local directory where to
    save all projects managed by the library. This path is stored in the default user directory file, and it is reused
    for all future projects. To support replacing this path without searching for the default user directory
    (usually hidden), this function finds and updates the contents of the file that stores the local root path.

    Args:
        path: The path to the new local root directory.
    """
@dataclass()
class ProjectConfiguration(YamlConfig):
    """Stores the project-specific configuration parameters that do not change between different animals and runtime
    sessions.

    An instance of this class is generated and saved as a .yaml file in the 'configuration' directory of the project
    when it is created. After that, the stored data is reused for each training or experiment session executed for each
    animal of the project.

    Notes:
        This class allows configuring this library to work for every project in the Sun lab while sharing (and hiding)
        the internal APIs and runtime control functions. This achieves a streamlined user experience, as users do not
        see nor care about inner workings of this library, while supporting project-specific customization.

        The class is primarily designed to specify the 'surgery_sheet_id' and the 'water_log_sheet_id' values,
        which likely differ between projects. However, the class also allows configuring the hardware interfaces and
        directory paths used during data acquisition. While this information should not change between projects, having
        the ability to adjust it on a per-project basis may be helpful in the future.
    """

    surgery_sheet_id: str = ...
    water_log_sheet_id: str = ...
    credentials_path: str | Path = ...
    local_root_directory: str | Path = ...
    server_root_directory: str | Path = ...
    nas_root_directory: str | Path = ...
    mesoscope_root_directory: str | Path = ...
    face_camera_index: int = ...
    left_camera_index: int = ...
    right_camera_index: int = ...
    harvesters_cti_path: str | Path = ...
    actor_port: str = ...
    sensor_port: str = ...
    encoder_port: str = ...
    headbar_port: str = ...
    lickport_port: str = ...
    unity_ip: str = ...
    unity_port: int = ...
    valve_calibration_data: dict[int | float, int | float] | tuple[tuple[int | float, int | float], ...] = ...
    @classmethod
    def load(cls, project_name: str) -> ProjectConfiguration:
        """Loads the project configuration parameters from a .yaml file and uses the loaded data to initialize a
        ProjectConfiguration instance.

        This method is called for each project runtime to reuse the configuration parameters generated at project
        creation. When it is called for the first time (during new project creation), the method generates a default
        configuration file and prompts the user to update the configuration before proceeding with the runtime.

        Notes:
            As part of its runtime, the method may prompt the user to provide the path to the local root directory.
            This directory stores all project subdirectories and acts as the top level of the local data hierarchy.
            The path to the directory will be saved in the static library-specific configuration file inside user's
            default data directory, so that it can be reused for all future runtimes. Use 'replace_root_path' function
            to replace the path that is saved in this way.

        Args:
            project_name: the name of the project whose configuration file needs to be discovered and loaded.

        Returns:
            An initialized ProjectConfiguration instance.
        """
    def _to_path(self, path: Path) -> None:
        """Saves the instance data to disk as a .yaml file.

        This method is automatically called when the project is created. All future runtimes use the from_path() method
        to load and reuse the configuration data saved to the .yaml file.

        Args:
            path: The path to the .yaml file to save the data to.
        """
    def _verify_data(self) -> None:
        """Verifies the data loaded from a .yaml file to ensure its validity.

        Since this class is explicitly designed to be modified by the user, this verification step is carried out to
        ensure that the loaded data matches expectations. This reduces the potential for user errors to impact the
        runtime behavior of the library. This internal method is automatically called by the from_path() method.

        Notes:
            The method does not verify all fields loaded from the configuration file and instead focuses on paths and
            Google Sheet IDs. The binding classes verify uSB ports and camera indices during instantiation.

        Raises:
            ValueError: If the loaded data does not match expected formats or values.
        """

@dataclass()
class ExperimentState:
    """Encapsulates the information used to set and maintain the desired Mesoscope-VR and experiment state.

    Primarily, experiment runtime logic (experiment task logic) is resolved by the Unity game engine. However, the
    Mesoscope-VR system configuration may also need to change throughout the experiment to optimize the runtime by
    disabling or reconfiguring specific hardware modules. For example, some experiment stages may require the running
    wheel to be locked to prevent the animal from running, but others may require it to be unlocked, to facilitate
    running behavior.

    Overall, the Mesoscope-VR system functions like a state-machine, with multiple statically configured states that
    can be activated and maintained throughout the experiment. During runtime, the runtime control function expects a
    sequence of ExperimentState instances that will be traversed, start-to-end, to determine the flow of the experiment
    runtime.

    Notes:
        Do not instantiate this class directly. It is managed by the ExperimentConfiguration wrapper class.
    """

    experiment_state_code: int
    vr_state_code: int
    state_duration_s: float

@dataclass()
class ExperimentConfiguration(YamlConfig):
    """Stores the configuration of a single experiment runtime.

    Primarily, this includes the sequence of experiment and Virtual Reality (Mesoscope-VR) states that define the flow
    of the experiment runtime. During runtime, the main control function traverses the sequence of states stored in
    this class instance start-to-end in the exact order specified by the user. Together with custom Unity projects, this
    class allows flexibly implementing a wide range of experiments.

    Each project should define one or more experiment configurations and save them as .yaml files inside the project
    'configuration' folder. The name for each configuration file is defined by the user and is used to identify and load
    the experiment configuration when 'sl-run-experiment' CLI command is executed.
    """

    cue_map: dict[int, float] = field(default_factory=Incomplete)
    experiment_states: dict[str, ExperimentState] = field(default_factory=Incomplete)

@dataclass()
class HardwareConfiguration(YamlConfig):
    """This class is used to save the runtime hardware configuration parameters as a .yaml file.

    This information is used to read and decode the data saved to the .npz log files during runtime as part of data
    processing.

    Notes:
        All fields in this dataclass initialize to None. During log processing, any log associated with a hardware
        module that provides the data stored in a field will be processed, unless that field is None. Therefore, setting
        any field in this dataclass to None also functions as a flag for whether to parse the log associated with the
        module that provides this field's information.

        This class is automatically configured by MesoscopeExperiment and BehaviorTraining classes to facilitate log
        parsing.
    """

    cue_map: dict[int, float] | None = ...
    cm_per_pulse: float | None = ...
    maximum_break_strength: float | None = ...
    minimum_break_strength: float | None = ...
    lick_threshold: int | None = ...
    valve_scale_coefficient: float | None = ...
    valve_nonlinearity_exponent: float | None = ...
    torque_per_adc_unit: float | None = ...
    screens_initially_on: bool | None = ...
    recorded_mesoscope_ttl: bool | None = ...

@dataclass()
class LickTrainingDescriptor(YamlConfig):
    """This class is used to save the description information specific to lick training sessions as a .yaml file.

    The information stored in this class instance is filled in two steps. The main runtime function fills most fields
    of the class, before it is saved as a .yaml file. After runtime, the experimenter manually fills leftover fields,
    such as 'experimenter_notes,' before the class instance is transferred to the long-term storage destination.

    The fully filled instance data is also used during preprocessing to write the water restriction log entry for the
    trained animal.
    """

    experimenter: str
    mouse_weight_g: float
    dispensed_water_volume_ml: float
    minimum_reward_delay: int
    maximum_reward_delay_s: int
    maximum_water_volume_ml: float
    maximum_training_time_m: int
    experimenter_notes: str = ...
    experimenter_given_water_volume_ml: float = ...

@dataclass()
class RunTrainingDescriptor(YamlConfig):
    """This class is used to save the description information specific to run training sessions as a .yaml file.

    The information stored in this class instance is filled in two steps. The main runtime function fills most fields
    of the class, before it is saved as a .yaml file. After runtime, the experimenter manually fills leftover fields,
    such as 'experimenter_notes,' before the class instance is transferred to the long-term storage destination.

    The fully filled instance data is also used during preprocessing to write the water restriction log entry for the
    trained animal.
    """

    experimenter: str
    mouse_weight_g: float
    dispensed_water_volume_ml: float
    final_run_speed_threshold_cm_s: float
    final_run_duration_threshold_s: float
    initial_run_speed_threshold_cm_s: float
    initial_run_duration_threshold_s: float
    increase_threshold_ml: float
    run_speed_increase_step_cm_s: float
    run_duration_increase_step_s: float
    maximum_water_volume_ml: float
    maximum_training_time_m: int
    experimenter_notes: str = ...
    experimenter_given_water_volume_ml: float = ...

@dataclass()
class MesoscopeExperimentDescriptor(YamlConfig):
    """This class is used to save the description information specific to experiment sessions as a .yaml file.

    The information stored in this class instance is filled in two steps. The main runtime function fills most fields
    of the class, before it is saved as a .yaml file. After runtime, the experimenter manually fills leftover fields,
    such as 'experimenter_notes,' before the class instance is transferred to the long-term storage destination.

    The fully filled instance data is also used during preprocessing to write the water restriction log entry for the
    animal participating in the experiment runtime.

    Notes:
        Critically, this class is used to save the mesoscope objective coordinates in the 'persistent' directory of an
        animal after each experiment session. This information is used during the following experiment runtimes to help
        the experimenter to restore the Mesoscope to the same position used during the previous session. This has to be
        done manually, as ThorLabs does not provide an API to work with the Mesoscope motors directly at this time.
    """

    experimenter: str
    mouse_weight_g: float
    dispensed_water_volume_ml: float
    experimenter_notes: str = ...
    experimenter_given_water_volume_ml: float = ...
    mesoscope_x_position: float = ...
    mesoscope_y_position: float = ...
    mesoscope_roll_position: float = ...
    mesoscope_z_position: float = ...
    mesoscope_fast_z_position: float = ...
    mesoscope_tip_position: float = ...
    mesoscope_tilt_position: float = ...

@dataclass()
class ZaberPositions(YamlConfig):
    """This class is used to save Zaber motor positions as a .yaml file to reuse them between sessions.

    The class is specifically designed to store, save, and load the positions of the LickPort and HeadBar motors
    (axes). It is used to both store Zaber motor positions for each session for future analysis and to restore the same
    Zaber motor positions across consecutive runtimes for the same project and animal combination.

    Notes:
        All positions are saved using native motor units. All class fields initialize to default placeholders that are
        likely NOT safe to apply to the VR system. Do not apply the positions loaded from the file unless you are
        certain they are safe to use.

        Exercise caution when working with Zaber motors. The motors are powerful enough to damage the surrounding
        equipment and manipulated objects. Do not modify the data stored inside the .yaml file unless you know what you
        are doing.
    """

    headbar_z: int = ...
    headbar_pitch: int = ...
    headbar_roll: int = ...
    lickport_z: int = ...
    lickport_x: int = ...
    lickport_y: int = ...

@dataclass()
class MesoscopePositions(YamlConfig):
    """This class is used to save the Mesoscope position as a .yaml file to reuse it between experiment sessions.

    Primarily, the class is used to help the experimenter to position the Mesoscope at the same position across
    multiple imaging sessions.

    Notes:
        The same information as in this class is stored in the MesoscopeExperimentDescriptor class. The key difference
        is that the data from this class is kept in the 'persistent' VRPC directory and updated with each session, while
        each descriptor is permanently stored in each session raw_data directory.
    """

    mesoscope_x_position: float = ...
    mesoscope_y_position: float = ...
    mesoscope_roll_position: float = ...
    mesoscope_z_position: float = ...
    mesoscope_fast_z_position: float = ...
    mesoscope_tip_position: float = ...
    mesoscope_tilt_position: float = ...

@dataclass
class SessionData(YamlConfig):
    """Provides methods for managing the data acquired during one experiment or training session.

    The primary purpose of this class is to maintain the session data structure across all supported destinations. It
    generates the paths used by all other classes from this library to determine where to save and load various session
    data during runtime.

    As part of its initialization, the class generates the session directory for the input animal and project
    combination. Session directories use the current UTC timestamp, down to microseconds, as the directory name. This
    ensures that each session name is unique and preserves the overall session order.

    Notes:
        It is expected that the server, NAS, and mesoscope data directories are mounted on the host-machine via the
        SMB or equivalent protocol. All manipulations with these destinations are carried out with the assumption that
        the OS has full access to these directories and filesystems.

        This class is specifically designed for working with raw data from a single animal participating in a single
        experimental project session. Processed data is managed by the processing library methods and classes.
    """

    project_name: str
    animal_id: str
    surgery_sheet_id: str
    water_log_sheet_id: str
    session_type: str
    credentials_path: str | Path
    local_root_directory: str | Path
    server_root_directory: str | Path
    nas_root_directory: str | Path
    mesoscope_root_directory: str | Path
    session_name: str = ...
    experiment_name: str | None = ...
    def __post_init__(self) -> None:
        """Generates the session name and creates the session directory structure on all involved PCs."""
    @classmethod
    def from_path(cls, path: Path) -> SessionData:
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
    def to_path(self) -> None:
        """Saves the data of the instance to the 'raw_data' directory of the managed session as a 'session_data.yaml'
        file.

        This is used to save the data stored in the instance to disk, so that it can be reused during preprocessing or
        data processing. This also serves as the repository for the identification information about the project,
        animal, and session that generated the data.
        """
    @property
    def raw_data_path(self) -> Path:
        """Returns the path to the 'raw_data' directory of the managed session on the VRPC.

        This directory functions as the root directory that stores all raw data acquired during training or experiment
        runtime for a given session.
        """
    @property
    def camera_frames_path(self) -> Path:
        """Returns the path to the 'camera_frames' directory of the managed session.

        This subdirectory is stored under the 'raw_data' directory and aggregates all video camera data.
        """
    @property
    def zaber_positions_path(self) -> Path:
        """Returns the path to the 'zaber_positions.yaml' file of the managed session.

        This path is used to save the positions for all Zaber motors of the HeadBar and LickPort controllers at the
        end of the experimental session.
        """
    @property
    def session_descriptor_path(self) -> Path:
        """Returns the path to the 'session_descriptor.yaml' file of the managed session.

        This path is used to save important session information to be viewed by experimenters post-runtime and to use
        for further processing.
        """
    @property
    def hardware_configuration_path(self) -> Path:
        """Returns the path to the 'hardware_configuration.yaml' file of the managed session.

        This file stores hardware module parameters used to read and parse .npz log files during data processing.
        """
    @property
    def previous_zaber_positions_path(self) -> Path:
        """Returns the path to the 'zaber_positions.yaml' file of the previous session.

        The file is stored inside the 'persistent_data' directory of the managed animal.
        """
    @property
    def mesoscope_root_path(self) -> Path:
        """Returns the path to the root directory of the Mesoscope pc (ScanImagePC) used to store all
        mesoscope-acquired data.
        """
    @property
    def nas_root_path(self) -> Path:
        """Returns the path to the root directory of the Synology NAS (Network Attached Storage) used to store all
        training and experiment data after preprocessing (backup cold long-term storage)."""
    @property
    def server_root_path(self) -> Path:
        """Returns the path to the root directory of the BioHPC server used to process and store all training and e
        experiment data (main long-term storage)."""
    @property
    def mesoscope_persistent_path(self) -> Path:
        """Returns the path to the 'persistent_data' directory of the Mesoscope pc (ScanImagePC).

        This directory is primarily used to store the reference MotionEstimator.me files for each animal.
        """
    @property
    def local_metadata_path(self) -> Path:
        """Returns the path to the 'metadata' directory of the managed animal on the VRPC."""
    @property
    def server_metadata_path(self) -> Path:
        """Returns the path to the 'metadata' directory of the managed animal on the BioHPC server."""
    @property
    def nas_metadata_path(self) -> Path:
        """Returns the path to the 'metadata' directory of the managed animal on the Synology NAS."""
    @property
    def experiment_configuration_path(self) -> Path:
        """Returns the path to the .yaml file that stores the configuration of the experiment runtime for the managed
        session.

        This information is used during experiment runtimes to determine how to run the experiment.
        """
    @property
    def local_experiment_configuration_path(self) -> Path:
        """Returns the path to the .yaml file used to save the managed session's experiment configuration.

        This is used to preserve the experiment configuration inside the raw_data directory of the managed session.
        """
    @property
    def previous_mesoscope_positions_path(self) -> Path:
        """Returns the path to the 'mesoscope_positions.yaml' file of the previous session.

        The file is stored inside the 'persistent_data' directory of the managed animal and is used to help restore the
        Mesoscope to the same position during following session(s).
        """

@dataclass()
class SubjectData:
    """Stores the surgical procedure subject (mouse) ID information."""

    id: int
    ear_punch: str
    sex: str
    genotype: str
    date_of_birth_us: int
    weight_g: float
    cage: int
    location_housed: str
    status: str

@dataclass()
class ProcedureData:
    """Stores the general information about the surgical procedure."""

    surgery_start_us: int
    surgery_end_us: int
    surgeon: str
    protocol: str
    surgery_notes: str
    post_op_notes: str

@dataclass
class ImplantData:
    """Stores the information about a single implantation performed during surgery.

    Multiple ImplantData instances are used at the same time if the surgery involved multiple implants.
    """

    implant: str
    implant_target: str
    implant_code: int
    implant_ap_coordinate_mm: float
    implant_ml_coordinate_mm: float
    implant_dv_coordinate_mm: float

@dataclass
class InjectionData:
    """Stores the information about a single injection performed during surgery.

    Multiple InjectionData instances are used at the same time if the surgery involved multiple injections.
    """

    injection: str
    injection_target: str
    injection_volume_nl: float
    injection_code: int
    injection_ap_coordinate_mm: float
    injection_ml_coordinate_mm: float
    injection_dv_coordinate_mm: float

@dataclass
class DrugData:
    """Stores the information about all drugs administered to the subject before, during, and immediately after the
    surgical procedure.
    """

    lactated_ringers_solution_volume_ml: float
    lactated_ringers_solution_code: int
    ketoprofen_volume_ml: float
    ketoprofen_code: int
    buprenorphine_volume_ml: float
    buprenorphine_code: int
    dexamethasone_volume_ml: float
    dexamethasone_code: int

@dataclass
class SurgeryData(YamlConfig):
    """Aggregates all data for a single mouse surgery procedure.

    This class aggregates other dataclass instances that store specific data about the surgical procedure. Primarily, it
    is used to save the data as a .yaml file to the metadata directory of each animal used in every lab project.
    This way, the surgery data is always stored alongside the behavior and brain activity data collected during training
    and experiment runtimes.
    """

    subject: SubjectData
    procedure: ProcedureData
    drugs: DrugData
    implants: list[ImplantData]
    injections: list[InjectionData]
