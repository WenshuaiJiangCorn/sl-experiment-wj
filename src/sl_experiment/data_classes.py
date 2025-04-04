"""This module provides classes that manage, save, and load varius library-generated data. Primarily, this data can be
categorized as configuration data, session runtime data, or general non-session data. Configuration data is typically
user-addressable (via editing the corresponding .yaml file) and is used to configure the runtime managed by the library.
This includes data managed by the ProjectConfiguration and ExperimentConfiguration classes, for example. Session runtime
data is generated during runtime and specifically excludes animal-generated behavior data (which is managed by the
DataLogger and other Ataraxis classes). Examples of this data can be found in the ZaberPositions or MesoscopePosition
classes. General non-runtime data currently includes animal surgery data (and, in the future, other 'metadata'). This
data is stored in SurgeryData class instance. Regardless of data-type, all classes and methods from this module are not
intended to be called by users directly."""

import re
from pathlib import Path
import warnings
from dataclasses import field, dataclass

import appdirs
from ataraxis_base_utilities import LogLevel, console, ensure_directory_exists
from ataraxis_data_structures import YamlConfig
from ataraxis_time.time_helpers import get_timestamp


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
    # Resolves the path to the static .txt file used to store the local path to the root directory
    app_dir = Path(appdirs.user_data_dir(appname="sl_experiment", appauthor="sun_lab"))
    path_file = app_dir.joinpath("root_path.txt")

    # In case this function is called before the app directory is created, ensures the app directory exists
    ensure_directory_exists(path_file)

    # Ensures that the input root directory exists
    ensure_directory_exists(path)

    # Replaces the contents of the root_path.txt file with the provided path
    with open(path_file, "w") as f:
        f.write(str(path))


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

    surgery_sheet_id: str = ""
    """The ID for the Google Sheet file that stores surgery information for the animal whose data is managed by this 
    instance. This is used to parse and write the surgery data for each managed animal into its 'metadata' folder, so 
    that the surgery data is always kept together with the rest of the training and experiment data."""
    water_log_sheet_id: str = ""
    """The ID for the Google Sheet file that stores water restriction information for the animal whose data is managed 
    by this instance. This is used to synchronize the information inside the water restriction log with the state of 
    the animal at the end of each training or experiment runtime.
    """
    credentials_path: str | Path = Path("/media/Data/Experiments/sl-surgery-log-0f651e492767.json")
    """
    The path to the locally stored .JSON file that contains the service account credentials used to read and write 
    Google Sheet data. This is used to access and work with the surgery log and the water restriction log. Usually, the 
    same service account is used across all projects.
    """
    local_root_directory: str | Path = Path("/media/Data/Experiments")
    """The path to the root directory where all projects are stored on the host-machine (VRPC). Note, this is always 
    written automatically when the class is saved to disk or loaded from disk, so manually writing this value is 
    pointless."""
    server_root_directory: str | Path = Path("/media/cbsuwsun/storage/sun_data")
    """The path to the root directory where all projects are stored on the BioHPC server machine."""
    nas_root_directory: str | Path = Path("/home/cybermouse/nas/rawdata")
    """The path to the root directory where all projects are stored on the Synology NAS."""
    mesoscope_root_directory: str | Path = Path("/home/cybermouse/scanimage/mesodata")
    """The path to the root directory used to store all mesoscope-acquired data on the PC that manages the 
    mesoscope (ScanImagePC)."""
    face_camera_index: int = 0
    """The index of the face camera in the list of all available Harvester-managed cameras."""
    left_camera_index: int = 0
    """The index of the left body camera in the list of all available OpenCV-managed cameras."""
    right_camera_index: int = 2
    """The index of the right body camera in the list of all available OpenCV-managed cameras."""
    harvesters_cti_path: str | Path = Path("/opt/mvIMPACT_Acquire/lib/x86_64/mvGenTLProducer.cti")
    """The path to the GeniCam CTI file used to connect to Harvesters-managed cameras. Currently, this is only used by 
    the face camera."""
    actor_port: str = "/dev/ttyACM0"
    """The USB port used by the Actor Microcontroller."""
    sensor_port: str = "/dev/ttyACM1"
    """The USB port used by the Sensor Microcontroller."""
    encoder_port: str = "/dev/ttyACM2"
    """The USB port used by the Encoder Microcontroller."""
    headbar_port: str = "/dev/ttyUSB0"
    """The USB port used by the HeadBar Zaber motor controllers (devices)."""
    lickport_port: str = "/dev/ttyUSB1"
    """The USB port used by the LickPort Zaber motor controllers (devices)."""
    unity_ip: str = "127.0.0.1"
    """The IP address of the MQTT broker used to communicate with the Unity game engine. Note, this is only used during 
    experiment runtimes. Training runtimes ignore this parameter."""
    unity_port: int = 1883
    """The port number of the MQTT broker used to communicate with the Unity game engine. Note, this is only used during
    experiment runtimes. Training runtimes ignore this parameter."""
    valve_calibration_data: dict[int | float, int | float] | tuple[tuple[int | float, int | float], ...] = (
        (15000, 1.8556),
        (30000, 3.4844),
        (45000, 7.1846),
        (60000, 10.0854),
    )
    """A dictionary or tuple of tuples that maps valve open times, in microseconds, to the dispensed volume of water, 
    in microliters. During runtime, this data is used by the ValveModule to translate the requested reward volumes into
    times the valve needs to be open to deliver the desired volume.
    """

    @classmethod
    def load(cls, project_name: str) -> "ProjectConfiguration":
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

        # Ensures console is enabled
        if not console.enabled:
            console.enable()

        # Uses appdirs to localize default user data directory. This serves as a static pointer to the storage directory
        # where the path to the 'root' experiment directory can be safely stored between runtimes. This way, we can set
        # the path once and reuse it for all future projects and runtimes.
        app_dir = Path(appdirs.user_data_dir(appname="sl_experiment", appauthor="sun_lab"))
        path_file = app_dir.joinpath("root_path.txt")

        # If the .txt file that stores the local root path does not exist, prompts the user to provide the path to the
        # local root directory and creates the root_path.txt file
        if not path_file.exists():
            # Gets the path to the local root directory from the user via command line input
            message = (
                "Unable to resolve the local root directory automatically. Provide the absolute path to the local "
                "directory that stores all project-specific directories."
            )
            console.echo(message=message, level=LogLevel.WARNING)
            root_path_str = input("Local root path: ")
            root_path = Path(root_path_str)

            # If necessary, generates the local root directory
            ensure_directory_exists(root_path)

            # Also ensures that the app directory exists, so that the path_file can be created below.
            ensure_directory_exists(path_file)

            # Saves the root path to the file
            with open(path_file, "w") as f:
                f.write(str(root_path))

        # Otherwise, uses the root path and the project name to resolve the path to the project configuration directory
        # and load the project configuration data.
        else:
            # Reads the root path from the file
            with open(path_file, "r") as f:
                root_path = Path(f.read().strip())

        # Uses the root experiment directory path to generate the path to the target project's configuration file.
        config_path = root_path.joinpath(project_name, "configuration", "project_configuration.yaml")
        ensure_directory_exists(config_path)  # Ensures the directory tree for the config path exists.
        if not config_path.exists():
            message = (
                f"Unable to load project configuration data from disk as no 'project_configuration.yaml' file found at "
                f"the provided project path. Generating a precursor (default) configuration file at the "
                f"specified path. Edit the file to specify project configuration before proceeding further to avoid "
                f"runtime errors."
            )
            console.echo(message=message, level=LogLevel.WARNING)

            # Generates the default configuration instance and dumps it as a .yaml file. Note, as part of this process,
            # the class generates the correct 'local_root_path' based on the path provided by the user.
            precursor = ProjectConfiguration(local_root_directory=root_path)
            precursor._to_path(path=config_path)

            # Waits for the user to manually configure the newly created file.
            input(f"Enter anything to continue: ")

        # Loads the data from the YAML file and initializes the class instance.
        instance: ProjectConfiguration = cls.from_yaml(file_path=config_path)  # type: ignore

        # Converts all paths loaded as strings to Path objects used inside the library
        instance.mesoscope_root_directory = Path(instance.mesoscope_root_directory)
        instance.nas_root_directory = Path(instance.nas_root_directory)
        instance.server_root_directory = Path(instance.server_root_directory)
        instance.credentials_path = Path(instance.credentials_path)
        instance.harvesters_cti_path = Path(instance.harvesters_cti_path)

        # Local root path is always re-computed using the data stored in the user data directory.
        instance.local_root_directory = root_path

        # Converts valve_calibration data from dictionary to a tuple of tuples format
        if not isinstance(instance.valve_calibration_data, tuple):
            instance.valve_calibration_data = tuple((k, v) for k, v in instance.valve_calibration_data.items())

        # Partially verifies the loaded data. Most importantly, this step does not allow proceeding if the user did not
        # replace the surgery log adn water restriction log placeholders with valid ID values.
        instance._verify_data()

        # Returns the initialized class instance to caller
        return instance

    def _to_path(self, path: Path) -> None:
        """Saves the instance data to disk as a .yaml file.

        This method is automatically called when the project is created. All future runtimes use the from_path() method
        to load and reuse the configuration data saved to the .yaml file.

        Args:
            path: The path to the .yaml file to save the data to.
        """

        # Converts all Path objects to strings before dumping the data, as .yaml encoder does not properly recognize
        # Path objects
        self.local_root_directory = str(self.local_root_directory)
        self.mesoscope_root_directory = str(self.mesoscope_root_directory)
        self.nas_root_directory = str(self.nas_root_directory)
        self.server_root_directory = str(self.server_root_directory)
        self.credentials_path = str(self.credentials_path)
        self.harvesters_cti_path = str(self.harvesters_cti_path)

        # Converts valve calibration data into dictionary format
        if isinstance(self.valve_calibration_data, tuple):
            self.valve_calibration_data = {k: v for k, v in self.valve_calibration_data}

        # Saves the data to the YAML file
        self.to_yaml(file_path=path)

        # As part of this runtime, also generates and dumps the 'default' experiment configuration. The user can then
        # use the default file as an example to write their own experiment configurations.
        example_experiment = ExperimentConfiguration()
        example_experiment.to_yaml(path.parent.joinpath("default_experiment.yaml"))

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

        # Verifies Google Sheet ID formatting. Google Sheet IDs are usually 44 characters long, containing letters,
        # numbers, hyphens, and underscores
        pattern = r"^[a-zA-Z0-9_-]{44}$"
        if not re.match(pattern, self.surgery_sheet_id):
            message = (
                f"Unable to verify the surgery_sheet_id field loaded from the 'project_configuration.yaml' file. "
                f"Expected a string with 44 characters, using letters, numbers, hyphens, and underscores, but found: "
                f"{self.surgery_sheet_id}."
            )
            console.error(message=message, error=ValueError)
        if not re.match(pattern, self.water_log_sheet_id):
            message = (
                f"Unable to verify the surgery_sheet_id field loaded from the 'project_configuration.yaml' file. "
                f"Expected a string with 44 characters, using letters, numbers, hyphens, and underscores, but found: "
                f"{self.water_log_sheet_id}."
            )
            console.error(message=message, error=ValueError)

        # Verifies the path to the credentials' file and the path to the Harvesters CTI file:
        if isinstance(self.credentials_path, Path) and (
            not self.credentials_path.exists() or self.credentials_path.suffix != ".json"
        ):
            message = (
                f"Unable to verify the credentials_path field loaded from the 'project_configuration.yaml' file. "
                f"Expected a path to an existing .json file that stores the Google API service account credentials, "
                f"but instead encountered the path to a non-json or non-existing file: {self.credentials_path}."
            )
            console.error(message=message, error=ValueError)
        if isinstance(self.harvesters_cti_path, Path) and (
            not self.harvesters_cti_path.exists() or self.harvesters_cti_path.suffix != ".cti"
        ):
            message = (
                f"Unable to verify the harvesters_cti_path field loaded from the 'project_configuration.yaml' file. "
                f"Expected a path to an existing .cti file that can be used to access a Genicam-compatible camera, "
                f"but instead encountered the path to a non-cti or non-existing file: {self.harvesters_cti_path}."
            )
            console.error(message=message, error=ValueError)

        # Verifies directory paths:
        if isinstance(self.mesoscope_root_directory, Path) and (
            not self.mesoscope_root_directory.is_dir() or not self.mesoscope_root_directory.exists()
        ):
            message = (
                f"Unable to verify the mesoscope_root_directory field loaded from the 'project_configuration.yaml' "
                f"file. Expected a path to an existing filesystem-mounted directory, but instead encountered the path "
                f"to a non-directory or non-existing directory: {self.mesoscope_root_directory}."
            )
            console.error(message=message, error=ValueError)
        if isinstance(self.nas_root_directory, Path) and (
            not self.nas_root_directory.is_dir() or not self.nas_root_directory.exists()
        ):
            message = (
                f"Unable to verify the nas_root_directory field loaded from the 'project_configuration.yaml' file. "
                f"Expected a path to an existing filesystem-mounted directory, but instead encountered the path to a "
                f"non-directory or non-existing directory: {self.nas_root_directory}."
            )
            console.error(message=message, error=ValueError)
        if isinstance(self.server_root_directory, Path) and (
            not self.server_root_directory.is_dir() or not self.server_root_directory.exists()
        ):
            message = (
                f"Unable to verify the server_root_directory field loaded from the 'project_configuration.yaml' file. "
                f"Expected a path to an existing filesystem-mounted directory, but instead encountered the path to a "
                f"non-directory or non-existing directory: {self.server_root_directory}."
            )
            console.error(message=message, error=ValueError)


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
    """The integer code of the experiment state. Experiment states do not have a predefined meaning, Instead, each 
    project is expected to define and follow its own experiment state code mapping. Typically, the experiment state 
    code is used to denote major experiment stages, such as 'baseline', 'task', 'cooldown', etc. Note, the same 
    experiment state code can be used by multiple sequential ExperimentState instances to change the VR system states 
    while maintaining the same experiment state."""
    vr_state_code: int
    """One of the supported VR system state-codes. Currently, the Mesoscope-VR system supports two state codes. State 
    code '1' denotes 'REST' state and code '2' denotes 'RUN' state. Note, multiple consecutive ExperimentState 
    instances with different experiment state codes can reuse the same VR state code."""
    state_duration_s: float
    """The time, in seconds, to maintain the current combination of the experiment and VR states."""


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

    cue_map: dict[int, float] = field(default_factory=lambda: {0: 30.0, 1: 30.0, 2: 30.0, 3: 30.0, 4: 30.0})
    """A dictionary that maps each integer-code associated with a wall cue used in the Virtual Reality experiment 
    environment to its length in real-world centimeters. It is used to map each VR cue to the distance the animal needs
    to travel to fully traverse the wall cue region from start to end."""
    experiment_states: dict[str, ExperimentState] = field(
        default_factory=lambda: {
            "baseline": ExperimentState(experiment_state_code=1, vr_state_code=1, state_duration_s=30),
            "experiment": ExperimentState(experiment_state_code=2, vr_state_code=2, state_duration_s=120),
            "cooldown": ExperimentState(experiment_state_code=3, vr_state_code=1, state_duration_s=15),
        }
    )
    """A dictionary that uses human-readable state-names as keys and ExperimentState instances as values. Each 
    ExperimentState instance represents a phase of the experiment."""


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
    valve_scale_coefficient: float | None = None
    """ValveInterface instance property."""
    valve_nonlinearity_exponent: float | None = None
    """ValveInterface instance property."""
    torque_per_adc_unit: float | None = None
    """TorqueInterface instance property."""
    screens_initially_on: bool | None = None
    """ScreenInterface instance property."""
    recorded_mesoscope_ttl: bool | None = None
    """TTLInterface instance property."""


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
    """The ID of the experimenter running the session."""
    mouse_weight_g: float
    """The weight of the animal, in grams, at the beginning of the session."""
    dispensed_water_volume_ml: float
    """Stores the total water volume, in milliliters, dispensed during runtime."""
    minimum_reward_delay: int
    """Stores the minimum delay, in seconds, that can separate the delivery of two consecutive water rewards."""
    maximum_reward_delay_s: int
    """Stores the maximum delay, in seconds, that can separate the delivery of two consecutive water rewards."""
    maximum_water_volume_ml: float
    """Stores the maximum volume of water the system is allowed to dispense during training."""
    maximum_training_time_m: int
    """Stores the maximum time, in minutes, the system is allowed to run the training for."""
    experimenter_notes: str = "Replace this with your notes."
    """This field is not set during runtime. It is expected that each experimenter replaces this field with their 
    notes made during runtime."""
    experimenter_given_water_volume_ml: float = 0.0
    """The additional volume of water, in milliliters, administered by the experimenter to the animal after the session.
    """


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
    """The ID of the experimenter running the session."""
    mouse_weight_g: float
    """The weight of the animal, in grams, at the beginning of the session."""
    dispensed_water_volume_ml: float
    """Stores the total water volume, in milliliters, dispensed during runtime."""
    final_run_speed_threshold_cm_s: float
    """Stores the final running speed threshold, in centimeters per second, that was active at the end of training."""
    final_run_duration_threshold_s: float
    """Stores the final running duration threshold, in seconds, that was active at the end of training."""
    initial_run_speed_threshold_cm_s: float
    """Stores the initial running speed threshold, in centimeters per second, used during training."""
    initial_run_duration_threshold_s: float
    """Stores the initial above-threshold running duration, in seconds, used during training."""
    increase_threshold_ml: float
    """Stores the volume of water delivered to the animal, in milliliters, that triggerred the increase in the running 
    speed and duration thresholds."""
    run_speed_increase_step_cm_s: float
    """Stores the value, in centimeters per second, used by the system to increment the running speed threshold each 
    time the animal receives 'increase_threshold' volume of water."""
    run_duration_increase_step_s: float
    """Stores the value, in seconds, used by the system to increment the duration threshold each time the animal 
    receives 'increase_threshold' volume of water."""
    maximum_water_volume_ml: float
    """Stores the maximum volume of water the system is allowed to dispensed during training."""
    maximum_training_time_m: int
    """Stores the maximum time, in minutes, the system is allowed to run the training for."""
    experimenter_notes: str = "Replace this with your notes."
    """This field is not set during runtime. It is expected that each experimenter will replace this field with their 
    notes made during runtime."""
    experimenter_given_water_volume_ml: float = 0.0
    """The additional volume of water, in milliliters, administered by the experimenter to the animal after the session.
    """


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
    """The ID of the experimenter running the session."""
    mouse_weight_g: float
    """The weight of the animal, in grams, at the beginning of the session."""
    dispensed_water_volume_ml: float
    """Stores the total water volume, in milliliters, dispensed during runtime."""
    experimenter_notes: str = "Replace this with your notes."
    """This field is not set during runtime. It is expected that each experimenter will replace this field with their 
    notes made during runtime."""
    experimenter_given_water_volume_ml: float = 0.0
    """The additional volume of water, in milliliters, administered by the experimenter to the animal after the session.
    """
    mesoscope_x_position: float = 0.0
    """The X-axis position, in centimeters, of the Mesoscope objective used during session runtime."""
    mesoscope_y_position: float = 0.0
    """The Y-axis position, in centimeters, of the Mesoscope objective used during session runtime."""
    mesoscope_roll_position: float = 0.0
    """The Roll-axis position, in degrees, of the Mesoscope objective used during session runtime."""
    mesoscope_z_position: float = 0.0
    """The Z-axis position, in centimeters, of the Mesoscope objective used during session runtime."""
    mesoscope_fast_z_position: float = 0.0
    """The Fast-Z-axis position, in micrometers, of the Mesoscope objective used during session runtime."""
    mesoscope_tip_position: float = 0.0
    """The Tilt-axis position, in degrees, of the Mesoscope objective used during session runtime."""
    mesoscope_tilt_position: float = 0.0
    """The Tip-axis position, in degrees, of the Mesoscope objective used during session runtime."""


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
class MesoscopePositions(YamlConfig):
    """This class is used to save the Mesoscope position as a .yaml file to reuse it between experiment sessions.

    Primarily, the class is used to help the experimenter to position the Mesoscope at the same position across
    multiple imaging sessions.

    Notes:
        The same information as in this class is stored in the MesoscopeExperimentDescriptor class. The key difference
        is that the data from this class is kept in the 'persistent' VRPC directory and updated with each session, while
        each descriptor is permanently stored in each session raw_data directory.
    """

    mesoscope_x_position: float = 0.0
    """The X-axis position, in centimeters, of the Mesoscope objective used during session runtime."""
    mesoscope_y_position: float = 0.0
    """The Y-axis position, in centimeters, of the Mesoscope objective used during session runtime."""
    mesoscope_roll_position: float = 0.0
    """The Roll-axis position, in degrees, of the Mesoscope objective used during session runtime."""
    mesoscope_z_position: float = 0.0
    """The Z-axis position, in centimeters, of the Mesoscope objective used during session runtime."""
    mesoscope_fast_z_position: float = 0.0
    """The Fast-Z-axis position, in micrometers, of the Mesoscope objective used during session runtime."""
    mesoscope_tip_position: float = 0.0
    """The Tilt-axis position, in degrees, of the Mesoscope objective used during session runtime."""
    mesoscope_tilt_position: float = 0.0
    """The Tip-axis position, in degrees, of the Mesoscope objective used during session runtime."""


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

    # Main attributes that are expected to be provided by the user during class initialization
    project_name: str
    """The name of the project for which the data is acquired."""
    animal_id: str
    """The ID code of the animal for which the data is acquired."""
    surgery_sheet_id: str
    """The ID for the Google Sheet file that stores surgery information for the animal whose data is managed by this 
    instance."""
    water_log_sheet_id: str
    """The ID for the Google Sheet file that stores water restriction information for the animal whose data is managed 
    by this instance.
    """
    session_type: str
    """Stores the type of the session. Primarily, this determines how to read the session_descriptor.yaml file. Has 
    to be set to one of the three supported types: 'Lick training', 'Run training' or 'Experiment'.
    """
    credentials_path: str | Path
    """
    The path to the locally stored .JSON file that stores the service account credentials used to read and write Google 
    Sheet data. This is used to access and work with the surgery log and the water restriction log.
    """
    local_root_directory: str | Path
    """The path to the root directory where all projects are stored on the host-machine (VRPC)."""
    server_root_directory: str | Path
    """The path to the root directory where all projects are stored on the BioHPC server machine."""
    nas_root_directory: str | Path
    """The path to the root directory where all projects are stored on the Synology NAS."""
    mesoscope_root_directory: str | Path
    """The path to the root directory used to store all mesoscope-acquired data on the ScanImagePC."""
    session_name: str = "None"
    """Stores the name of the session for which the data is acquired. This name is generated at class initialization 
    based on the current microsecond-accurate timestamp. Do NOT manually provide this name at class initialization.
    Use 'from_path' class method to initialize a SessionData instance for an already existing session data directory.
    """
    experiment_name: str | None = None
    """Stores the name of the experiment configuration file. If the session_name attribute is 'experiment', this filed
    is used to communicate the specific experiment configuration used by the session. During runtime, this is 
    used to load the experiment configuration (to run the experiment) and to save the experiment configuration to the
    session raw_data folder. If the session is not an experiment session, this is statically set to None."""

    def __post_init__(self) -> None:
        """Generates the session name and creates the session directory structure on all involved PCs."""

        # If the session name is provided, ends the runtime early. This supports initializing the
        # SessionData class from the path to the root directory of a previous created session, which is used during
        # runtime-independent data preprocessing.
        if "None" not in self.session_name:
            return

        # Acquires the UTC timestamp to use as the session name
        self.session_name = str(get_timestamp(time_separator="-"))

        # Ensures all root directory paths are stored as Path objects.
        self.local_root_directory = Path(self.local_root_directory)
        self.server_root_directory = Path(self.server_root_directory)
        self.nas_root_directory = Path(self.nas_root_directory)
        self.mesoscope_root_directory = Path(self.mesoscope_root_directory)

        # Constructs the session directory path and generates the directory
        raw_session_path = self.local_root_directory.joinpath(self.project_name, self.animal_id, self.session_name)

        # Handles potential session name conflicts
        counter = 0
        while raw_session_path.exists():
            counter += 1
            new_session_name = f"{self.session_name}_{counter}"
            raw_session_path = self.local_root_directory.joinpath(self.project_name, self.animal_id, new_session_name)

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
            self.local_root_directory.joinpath(self.project_name, self.animal_id, self.session_name, "raw_data")
        )
        ensure_directory_exists(
            self.local_root_directory.joinpath(
                self.project_name, self.animal_id, self.session_name, "raw_data", "camera_frames"
            )
        )
        ensure_directory_exists(
            self.local_root_directory.joinpath(
                self.project_name, self.animal_id, self.session_name, "raw_data", "mesoscope_frames"
            )
        )
        ensure_directory_exists(
            self.local_root_directory.joinpath(
                self.project_name, self.animal_id, self.session_name, "raw_data", "behavior_data_log"
            )
        )

        ensure_directory_exists(
            self.local_root_directory.joinpath(self.project_name, self.animal_id, "persistent_data")
        )
        ensure_directory_exists(self.nas_root_directory.joinpath(self.project_name, self.animal_id, self.session_name))
        ensure_directory_exists(
            self.server_root_directory.joinpath(self.project_name, self.animal_id, self.session_name)
        )
        ensure_directory_exists(self.local_root_directory.joinpath(self.project_name, self.animal_id, "metadata"))
        ensure_directory_exists(self.server_root_directory.joinpath(self.project_name, self.animal_id, "metadata"))
        ensure_directory_exists(self.nas_root_directory.joinpath(self.project_name, self.animal_id, "metadata"))
        ensure_directory_exists(self.mesoscope_root_directory.joinpath("mesoscope_frames"))
        ensure_directory_exists(
            self.mesoscope_root_directory.joinpath("persistent_data", self.project_name, self.animal_id)
        )

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

        # Loads class data
        instance: SessionData = cls.from_yaml(file_path=path)  # type: ignore

        # Ensures all loaded paths are stored as Path objects.
        instance.local_root_directory = Path(instance.local_root_directory)
        instance.mesoscope_root_directory = Path(instance.mesoscope_root_directory)
        instance.nas_root_directory = Path(instance.nas_root_directory)
        instance.server_root_directory = Path(instance.server_root_directory)
        instance.credentials_path = Path(instance.credentials_path)

        # Returns the instance to caller
        return instance

    def to_path(self) -> None:
        """Saves the data of the instance to the 'raw_data' directory of the managed session as a 'session_data.yaml'
        file.

        This is used to save the data stored in the instance to disk, so that it can be reused during preprocessing or
        data processing. This also serves as the repository for the identification information about the project,
        animal, and session that generated the data.
        """

        # Converts all Paths objects to strings before dumping the data to YAML.
        self.local_root_directory = str(self.local_root_directory)
        self.mesoscope_root_directory = str(self.mesoscope_root_directory)
        self.nas_root_directory = str(self.nas_root_directory)
        self.server_root_directory = str(self.server_root_directory)
        self.credentials_path = str(self.credentials_path)

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

    @property
    def experiment_configuration_path(self) -> Path:
        """Returns the path to the .yaml file that stores the configuration of the experiment runtime for the managed
        session.

        This information is used during experiment runtimes to determine how to run the experiment.
        """
        local_root_directory = Path(self.local_root_directory)
        return local_root_directory.joinpath(self.project_name, "configuration", f"{self.experiment_name}.yaml")

    @property
    def local_experiment_configuration_path(self) -> Path:
        """Returns the path to the .yaml file used to save the managed session's experiment configuration.

        This is used to preserve the experiment configuration inside the raw_data directory of the managed session.
        """
        return self.raw_data_path.joinpath(f"{self.experiment_name}.yaml")

    @property
    def previous_mesoscope_positions_path(self) -> Path:
        """Returns the path to the 'mesoscope_positions.yaml' file of the previous session.

        The file is stored inside the 'persistent_data' directory of the managed animal and is used to help restore the
        Mesoscope to the same position during following session(s).
        """
        local_root_directory = Path(self.local_root_directory)
        return local_root_directory.joinpath(
            self.project_name, self.animal_id, "persistent_data", "mesoscope_positions.yaml"
        )


@dataclass()
class SubjectData:
    """Stores the surgical procedure subject (mouse) ID information."""

    id: int
    """Stores the unique ID (name) of the subject. Assumes all mice are given a numeric ID, rather than a string name.
    """
    ear_punch: str
    """Stores the ear tag location of the subject."""
    sex: str
    """Stores the gender of the subject."""
    genotype: str
    """Stores the genotype of the subject."""
    date_of_birth_us: int
    """Stores the date of birth of the subject as the number of microseconds elapsed since UTC epoch onset."""
    weight_g: float
    """Stores the weight of the subject pre-surgery, in grams."""
    cage: int
    """Stores the number of the latest cage used to house the subject."""
    location_housed: str
    """Stores the latest location used to house the subject after the surgery."""
    status: str
    """Stores the latest status of the subject (alive / deceased)."""


@dataclass()
class ProcedureData:
    """Stores the general information about the surgical procedure."""

    surgery_start_us: int
    """Stores the date and time when the surgery was started as microseconds elapsed since UTC epoch onset."""
    surgery_end_us: int
    """Stores the date and time when the surgery has ended as microseconds elapsed since UTC epoch onset."""
    surgeon: str
    """Stores the name or ID of the surgeon. If the intervention was carrie out by multiple surgeon, all participating
    surgeon names and IDs are stored as part of the same string."""
    protocol: str
    """Stores the experiment protocol number (ID) used during the surgery."""
    surgery_notes: str
    """Stores surgeon's notes taken during the surgery."""
    post_op_notes: str
    """Stores surgeon's notes taken during the post-surgery recovery period."""


@dataclass
class ImplantData:
    """Stores the information about a single implantation performed during surgery.

    Multiple ImplantData instances are used at the same time if the surgery involved multiple implants.
    """

    implant: str
    """The descriptive name of the implant."""
    implant_target: str
    """The name of the brain region or cranium section targeted by the implant."""
    implant_code: int
    """The manufacturer code or internal reference code for the implant. This code is used to identify the implant in 
    additional datasheets and lab ordering documents."""
    implant_ap_coordinate_mm: float
    """Stores implant's antero-posterior stereotactic coordinate, in millimeters, relative to bregma."""
    implant_ml_coordinate_mm: float
    """Stores implant's medial-lateral stereotactic coordinate, in millimeters, relative to bregma."""
    implant_dv_coordinate_mm: float
    """Stores implant's dorsal-ventral stereotactic coordinate, in millimeters, relative to bregma."""


@dataclass
class InjectionData:
    """Stores the information about a single injection performed during surgery.

    Multiple InjectionData instances are used at the same time if the surgery involved multiple injections.
    """

    injection: str
    """The descriptive name of the injection."""
    injection_target: str
    """The name of the brain region targeted by the injection."""
    injection_volume_nl: float
    """The volume of substance, in nanoliters, delivered during the injection."""
    injection_code: int
    """The manufacturer code or internal reference code for the injected substance. This code is used to identify the 
    substance in additional datasheets and lab ordering documents."""
    injection_ap_coordinate_mm: float
    """Stores injection's antero-posterior stereotactic coordinate, in millimeters, relative to bregma."""
    injection_ml_coordinate_mm: float
    """Stores injection's medial-lateral stereotactic coordinate, in millimeters, relative to bregma."""
    injection_dv_coordinate_mm: float
    """Stores injection's dorsal-ventral stereotactic coordinate, in millimeters, relative to bregma."""


@dataclass
class DrugData:
    """Stores the information about all drugs administered to the subject before, during, and immediately after the
    surgical procedure.
    """

    lactated_ringers_solution_volume_ml: float
    """Stores the volume of Lactated Ringer's Solution (LRS) administered during surgery, in ml."""
    lactated_ringers_solution_code: int
    """Stores the manufacturer code or internal reference code for Lactated Ringer's Solution (LRS). This code is used 
    to identify the LRS batch in additional datasheets and lab ordering documents."""
    ketoprofen_volume_ml: float
    """Stores the volume of ketoprofen administered during surgery, in ml."""
    ketoprofen_code: int
    """Stores the manufacturer code or internal reference code for ketoprofen. This code is used to identify the 
    ketoprofen batch in additional datasheets and lab ordering documents."""
    buprenorphine_volume_ml: float
    """Stores the volume of buprenorphine administered during surgery, in ml."""
    buprenorphine_code: int
    """Stores the manufacturer code or internal reference code for buprenorphine. This code is used to identify the 
    buprenorphine batch in additional datasheets and lab ordering documents."""
    dexamethasone_volume_ml: float
    """Stores the volume of dexamethasone administered during surgery, in ml."""
    dexamethasone_code: int
    """Stores the manufacturer code or internal reference code for dexamethasone. This code is used to identify the 
    dexamethasone batch in additional datasheets and lab ordering documents."""


@dataclass
class SurgeryData(YamlConfig):
    """Aggregates all data for a single mouse surgery procedure.

    This class aggregates other dataclass instances that store specific data about the surgical procedure. Primarily, it
    is used to save the data as a .yaml file to the metadata directory of each animal used in every lab project.
    This way, the surgery data is always stored alongside the behavior and brain activity data collected during training
    and experiment runtimes.
    """

    subject: SubjectData
    """Stores the ID information about the subject (mouse)."""
    procedure: ProcedureData
    """Stores general data about the surgical procedure."""
    drugs: DrugData
    """Stores the data about the substances subcutaneously injected into the subject before, during and immediately 
    after the surgical intervention."""
    implants: list[ImplantData]
    """Stores the data for all cranial and transcranial implants introduced to the subject during the procedure."""
    injections: list[InjectionData]
    """Stores the data about all substances infused into the brain of the subject during the surgery."""
