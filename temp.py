@dataclass()
class DeepLabCutData:
    """Stores the paths to the directories and files that make up the 'deeplabcut' project-specific directory.

    DeepLabCut (DLC) is used to track animal body parts and poses in video data acquired during experiment and training
    sessions. Since DLC is designed to work with projects, rather than single animals or sessions, each Sun lab
    project data hierarchy contains a dedicated 'deeplabcut' directory under the root project directory. The contents of
    that directory are largely managed by the DLC itself. Therefore, each session of a given project refers to and
    uses the same 'deeplabcut' directory.
    """

    deeplabcut_path: Path = Path()
    """Stores the path to the project-specific DeepLabCut directory. This folder stores all DeepLabCut data specific to
    a single project, which is reused during the processing of all sessions of the project."""

    def resolve_paths(self, root_directory_path: Path) -> None:
        """Resolves all paths managed by the class instance based on the input root directory path.

        This method is called each time the class is instantiated to regenerate the managed path hierarchy on any
        machine that instantiates the class.

        Args:
            root_directory_path: The path to the top-level directory of the local hierarchy. Depending on the managed
                hierarchy, this has to point to a directory under the main /session, /animal, or /project directory of
                the managed session.
        """

        # Generates the managed paths
        self.deeplabcut_path = root_directory_path

    def make_directories(self) -> None:
        """Ensures that all major subdirectories and the root directory exist."""
        ensure_directory_exists(self.deeplabcut_path)


@dataclass()
class ConfigurationData:
    """Stores the paths to the directories and files that make up the 'configuration' project-specific directory.

    The configuration directory contains various configuration files and settings used by data acquisition,
    preprocessing, and processing pipelines in the lab. Generally, all configuration settings are defined once for each
    project and are reused for every session within the project. Therefore, this directory is created under each main
    project directory.

    Notes:
        Some attribute names inside this section match the names in the RawData section. This is intentional, as some
        configuration files are copied into the raw_data session directories to allow reinstating the session data
        hierarchy across machines.
    """

    configuration_path: Path = Path()
    """Stores the path to the project-specific configuration directory. This directory is used by all animals 
    and sessions of the project to store all pan-project configuration files. The configuration data is reused by all
    sessions in the project."""
    experiment_configuration_path: Path = Path()
    """Stores the path to the experiment_configuration.yaml file. This file contains the snapshot of the 
    experiment runtime configuration used by the session. This file is only created for experiment session. It does not
    exist for behavior training sessions."""
    project_configuration_path: Path = Path()
    """Stores the path to the project_configuration.yaml file. This file contains the snapshot of the configuration 
    parameters for the session's project."""
    single_day_s2p_configuration_path: Path = Path()
    """Stores the path to the single_day_s2p_configuration.yaml file stored inside the project's 'configuration' 
    directory on the fast BioHPC server volume. This configuration file specifies the parameters for the 'single day' 
    suite2p registration pipeline, which is applied to each session that generates brain activity data."""
    multi_day_s2p_configuration_path: Path = Path()
    """Stores the path to the multi_day_s2p_configuration.yaml file stored inside the project's 'configuration' 
    directory on the fast BioHPC server volume. This configuration file specifies the parameters for the 'multiday' 
    sl-suite2p-based registration pipelines used tot rack brain cells across multiple sessions."""

    def resolve_paths(self, root_directory_path: Path, experiment_name: str | None = None) -> None:
        """Resolves all paths managed by the class instance based on the input root directory path.

        This method is called each time the class is instantiated to regenerate the managed path hierarchy on any
        machine that instantiates the class.

        Args:
            root_directory_path: The path to the top-level directory of the local hierarchy. Depending on the managed
                hierarchy, this has to point to a directory under the main /session, /animal, or /project directory of
                the managed session.
            experiment_name: Optionally specifies the name of the experiment executed as part of the managed session's
                runtime. This is used to correctly configure the path to the specific ExperimentConfiguration data file.
                If the managed session is not an Experiment session, this parameter should be set to None.
        """

        # Generates the managed paths
        self.configuration_path = root_directory_path
        if experiment_name is None:
            self.experiment_configuration_path = self.configuration_path.joinpath("null")
        else:
            self.experiment_configuration_path = self.configuration_path.joinpath(f"{experiment_name}.yaml")
        self.project_configuration_path = self.configuration_path.joinpath("project_configuration.yaml")
        self.single_day_s2p_configuration_path = self.configuration_path.joinpath("single_day_s2p_configuration.yaml")
        self.multi_day_s2p_configuration_path = self.configuration_path.joinpath("multi_day_s2p_configuration.yaml")

    def make_directories(self) -> None:
        """Ensures that all major subdirectories and the root directory exist."""
        ensure_directory_exists(self.configuration_path)


@dataclass()
class VRPCPersistentData:
    """Stores the paths to the directories and files that make up the 'persistent_data' directory on the VRPC.

    Persistent data directories are only used during data acquisition. Therefore, unlike most other directories, they
    are purposefully designed for specific PCs that participate in data acquisition. This section manages the
    animal-specific persistent_data directory stored on the VRPC.

    VRPC persistent data directory is used to preserve configuration data, such as the positions of Zaber motors and
    Meososcope objective, so that they can be reused across sessions of the same animals. The data in this directory
    is read at the beginning of each session and replaced at the end of each session.
    """

    persistent_data_path: Path = Path()
    """Stores the path to the project and animal specific 'persistent_data' directory to which the managed session 
    belongs, relative to the VRPC root. This directory is exclusively used on the VRPC."""
    zaber_positions_path: Path = Path()
    """Stores the path to the Zaber motor positions snapshot generated at the end of the previous session runtime. This 
    is used to automatically restore all Zaber motors to the same position across all sessions."""
    mesoscope_positions_path: Path = Path()
    """Stores the path to the Mesoscope positions snapshot generated at the end of the previous session runtime. This 
    is used to help the user to (manually) restore the Mesoscope to the same position across all sessions."""

    def resolve_paths(self, root_directory_path: Path) -> None:
        """Resolves all paths managed by the class instance based on the input root directory path.

        This method is called each time the class is instantiated to regenerate the managed path hierarchy on any
        machine that instantiates the class.

        Args:
            root_directory_path: The path to the top-level directory of the local hierarchy. Depending on the managed
                hierarchy, this has to point to a directory under the main /session, /animal, or /project directory of
                the managed session.
        """

        # Generates the managed paths
        self.persistent_data_path = root_directory_path
        self.zaber_positions_path = self.persistent_data_path.joinpath("zaber_positions.yaml")
        self.mesoscope_positions_path = self.persistent_data_path.joinpath("mesoscope_positions.yaml")

    def make_directories(self) -> None:
        """Ensures that all major subdirectories and the root directory exist."""

        ensure_directory_exists(self.persistent_data_path)


@dataclass()
class ScanImagePCPersistentData:
    """Stores the paths to the directories and files that make up the 'persistent_data' directory on the ScanImagePC.

    Persistent data directories are only used during data acquisition. Therefore, unlike most other directories, they
    are purposefully designed for specific PCs that participate in data acquisition. This section manages the
    animal-specific persistent_data directory stored on the ScanImagePC (Mesoscope PC).

    ScanImagePC persistent data directory is used to preserve the motion estimation snapshot, generated during the first
    experiment session. This is necessary to align the brain recording field of view across sessions. In turn, this
    is used to carry out 'online' motion and z-drift correction, improving the accuracy of across-day (multi-day)
    cell tracking.
    """

    persistent_data_path: Path = Path()
    """Stores the path to the project and animal specific 'persistent_data' directory to which the managed session 
    belongs, relative to the ScanImagePC root. This directory is exclusively used on the ScanImagePC (Mesoscope PC)."""
    motion_estimator_path: Path = Path()
    """Stores the 'reference' motion estimator file generated during the first experiment session of each animal. This 
    file is kept on the ScanImagePC to image the same population of cells across all experiment sessions."""

    def resolve_paths(self, root_directory_path: Path) -> None:
        """Resolves all paths managed by the class instance based on the input root directory path.

        This method is called each time the class is instantiated to regenerate the managed path hierarchy on any
        machine that instantiates the class.

        Args:
            root_directory_path: The path to the top-level directory of the local hierarchy. Depending on the managed
                hierarchy, this has to point to a directory under the main /session, /animal, or /project directory of
                the managed session.
        """

        # Generates the managed paths
        self.persistent_data_path = root_directory_path
        self.motion_estimator_path = self.persistent_data_path.joinpath("MotionEstimator.me")

    def make_directories(self) -> None:
        """Ensures that all major subdirectories and the root directory exist."""

        ensure_directory_exists(self.persistent_data_path)


@dataclass()
class MesoscopeData:
    """Stores the paths to the directories and files that make up the 'meso_data' directory on the ScanImagePC.

    The meso_data directory is the root directory where all mesoscope-generated data is stored on the ScanImagePC. The
    path to this directory should be given relative to the VRPC root and be mounted to the VRPC filesystem via the
    SMB or equivalent protocol.

    During runtime, the ScanImagePC should organize all collected data under this root directory. During preprocessing,
    the VRPC uses SMB to access the data in this directory and merge it into the 'raw_data' session directory. The paths
    in this section, therefore, are specific to the VRPC and are not used on other PCs.
    """

    meso_data_path: Path = Path()
    """Stores the path to the root ScanImagePC data directory, mounted to the VRPC filesystem via the SMB or equivalent 
    protocol. All mesoscope-generated data is stored under this root directory before it is merged into the VRPC-managed
    raw_data directory of each session."""
    mesoscope_data_path: Path = Path()
    """Stores the path to the 'default' mesoscope_data directory. All experiment sessions across all animals and 
    projects use the same mesoscope_data directory to save the data generated by the mesoscope via ScanImage 
    software. This simplifies ScanImagePC configuration process during runtime, as all data is always saved in the same
    directory. During preprocessing, the data is moved from the default directory first into a session-specific 
    ScanImagePC directory and then into the VRPC raw_data session directory."""
    session_specific_path: Path = Path()
    """Stores the path to the session-specific data directory. This directory is generated at the end of each experiment
    runtime to prepare mesoscope data for being moved to the VRPC-managed raw_data directory and to reset the 'default' 
    mesoscope_data directory for the next session's runtime."""
    ubiquitin_path: Path = Path()
    """Stores the path to the 'ubiquitin.bin' file. This file is automatically generated inside the session-specific 
    data directory after its contents are safely transferred to the VRPC as part of preprocessing. During redundant data
    removal step of preprocessing, the VRPC searches for directories marked with ubiquitin.bin and deletes them from the
    ScanImagePC filesystem."""

    def resolve_paths(self, root_mesoscope_path: Path, session_name: str) -> None:
        """Resolves all paths managed by the class instance based on the input root directory path.

        This method is called each time the class is instantiated to regenerate the managed path hierarchy on any
        machine that instantiates the class.

        Args:
            root_mesoscope_path: The path to the top-level directory of the ScanImagePC data hierarchy mounted to the
                VRPC via the SMB or equivalent protocol.
            session_name: The name of the session for which this subclass is initialized.
        """

        # Generates the managed paths
        self.meso_data_path = root_mesoscope_path
        self.session_specific_path = self.meso_data_path.joinpath(session_name)
        self.ubiquitin_path = self.session_specific_path.joinpath("ubiquitin.bin")
        self.mesoscope_data_path = self.meso_data_path.joinpath("mesoscope_data")

    def make_directories(self) -> None:
        """Ensures that all major subdirectories and the root directory exist."""

        ensure_directory_exists(self.meso_data_path)


@dataclass()
class VRPCDestinations:
    """Stores the paths to the VRPC filesystem-mounted directories of the Synology NAS and BioHPC server.

    The paths from this section are primarily used to transfer preprocessed data to the long-term storage destinations.
    Additionally, they allow VRPC to interface with the configuration directory of the BioHPC server to start data
    processing jobs and to read the data from the processed_data directory to remove redundant data from the VRPC
    filesystem.

    Overall, this section is intended solely for the VRPC and should not be used on other PCs.
    """

    nas_raw_data_path: Path = Path()
    """Stores the path to the session's raw_data directory on the Synology NAS, which is mounted to the VRPC via the 
    SMB or equivalent protocol."""
    server_raw_data_path: Path = Path()
    """Stores the path to the session's raw_data directory on the BioHPC server, which is mounted to the VRPC via the 
    SMB or equivalent protocol."""
    server_processed_data_path: Path = Path()
    """Stores the path to the session's processed_data directory on the BioHPC server, which is mounted to the VRPC via 
    the SMB or equivalent protocol."""
    server_configuration_path: Path = Path()
    """Stores the path to the project-specific 'configuration' directory on the BioHPC server, which is mounted to the 
    VRPC via the SMB or equivalent protocol."""
    telomere_path: Path = Path()
    """Stores the path to the session's telomere.bin marker. This marker is generated as part of data processing on the 
    BioHPC server to notify the VRPC that the server received preprocessed data intact. The presence of this marker is 
    used by the VRPC to determine which locally stored raw_data is safe to delete from the filesystem."""
    suite2p_configuration_path: Path = Path()
    """Stores the path to the suite2p_configuration.yaml file stored inside the project's 'configuration' directory on
    the BioHPC server. This configuration file specifies the parameters for the 'single day' sl-suite2p registration 
    pipeline, which is applied to each session that generates brain activity data."""
    processing_tracker_path: Path = Path()
    """Stores the path to the processing_tracker.yaml file stored inside the sessions' root processed_data directory on 
    the BioHPC server. This file tracks which processing pipelines need to be applied the target session and the status 
    (success / failure) of each applied pipeline.
    """
    multiday_configuration_path: Path = Path()
    """Stores the path to the multiday_configuration.yaml file stored inside the project's 'configuration' directory 
    on the BioHPC server. This configuration file specifies the parameters for the 'multiday' sl-suite2p registration 
    pipeline used to track brain cells across multiple sessions."""

    def resolve_paths(
        self,
        nas_raw_data_path: Path,
        server_raw_data_path: Path,
        server_processed_data_path: Path,
        server_configuration_path: Path,
    ) -> None:
        """Resolves all paths managed by the class instance based on the input root directory paths.

        This method is called each time the class is instantiated to regenerate the managed path hierarchy on any
        machine that instantiates the class.

        Args:
            nas_raw_data_path: The path to the session's raw_data directory on the Synology NAS, relative to the VRPC
                filesystem root.
            server_raw_data_path: The path to the session's raw_data directory on the BioHPC server, relative to the
                VRPC filesystem root.
            server_processed_data_path: The path to the session's processed_data directory on the BioHPC server,
                relative to the VRPC filesystem root.
            server_configuration_path: The path to the project-specific 'configuration' directory on the BioHPC server,
                relative to the VRPC filesystem root.
        """

        # Generates the managed paths
        self.nas_raw_data_path = nas_raw_data_path
        self.server_raw_data_path = server_raw_data_path
        self.server_processed_data_path = server_processed_data_path
        self.server_configuration_path = server_configuration_path
        self.telomere_path = self.server_raw_data_path.joinpath("telomere.bin")
        self.suite2p_configuration_path = self.server_configuration_path.joinpath("suite2p_configuration.yaml")
        self.processing_tracker_path = self.server_processed_data_path.joinpath("processing_tracker.yaml")
        self.multiday_configuration_path = self.server_configuration_path.joinpath("multiday_configuration.yaml")

    def make_directories(self) -> None:
        """Ensures that all major subdirectories and the root directory exist."""
        ensure_directory_exists(self.nas_raw_data_path)
        ensure_directory_exists(self.server_raw_data_path)
        ensure_directory_exists(self.server_configuration_path)
        ensure_directory_exists(self.server_processed_data_path)