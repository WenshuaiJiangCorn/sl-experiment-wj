"""This module provides additional tools and classes used by other modules of the mesoscope_vr package. Primarily, this
includes various dataclasses specific to the Mesoscope-VR systems and utility functions used from other modules. The
contents of this module are not intended to be used outside the mesoscope_vr package."""

from pathlib import Path
from dataclasses import field, dataclass

from sl_shared_assets import SessionData, MesoscopeSystemConfiguration, get_system_configuration_data
from ataraxis_base_utilities import console, ensure_directory_exists


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
class VRPCPersistentData:
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

    def __post_init__(self):
        # Resolves paths that can be derived from the root path.
        self.zaber_positions_path = self.persistent_data_path.joinpath("zaber_positions.yaml")
        self.mesoscope_positions_path = self.persistent_data_path.joinpath("mesoscope_positions.yaml")

        # Resolves the session descriptor path based on the session type.
        if self.session_type == "lick training":
            self.session_descriptor_path = self.persistent_data_path.joinpath(f"lick_training_session_descriptor.yaml")
        if self.session_type == "run training":
            self.session_descriptor_path = self.persistent_data_path.joinpath(f"run_training_session_descriptor.yaml")
        if self.session_type == "mesoscope experiment":
            self.session_descriptor_path = self.persistent_data_path.joinpath(
                f"mesoscope_experiment_session_descriptor.yaml"
            )
        else:
            message = (
                f"Unsupported session type: {self.session_type} encountered when initializing additional path "
                f"dataclasses for the Mesoscope-VR data acquisition system. Supported session types are "
                f"'lick training', 'run training', and 'mesoscope experiment'."
            )
            console.error(message, error=ValueError)

        # Ensures that the target persistent directory exists
        ensure_directory_exists(self.persistent_data_path)


@dataclass()
class ScanImagePCData:
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
    root directory ('meso-data' directory).
    """
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
    """Stores the 'reference' motion estimator file generated during the first experiment session of each animal. This 
    file is kept on the ScanImagePC to image the same population of cells across all experiment sessions."""

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
class VRPCDestinations:
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

    def __post_init__(self):
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
        self.vrpc_persistent_data = VRPCPersistentData(
            session_type=session_data.session_type,
            persistent_data_path=system_configuration.paths.root_directory.joinpath(project, animal, "persistent_data"),
        )

        self.scanimagepc_data = ScanImagePCData(
            session_name=session,
            meso_data_path=system_configuration.paths.mesoscope_directory,
            persistent_data_path=system_configuration.paths.mesoscope_directory.joinpath(
                project, animal, "persistent_data"
            ),
        )

        self.destinations = VRPCDestinations(
            nas_raw_data_path=system_configuration.paths.nas_directory.joinpath(project, animal, session, "raw_data"),
            server_raw_data_path=system_configuration.paths.server_storage_directory.joinpath(
                project, animal, session, "raw_data"
            ),
            server_processed_data_path=system_configuration.paths.server_working_directory.joinpath(
                project, animal, session, "processed_data"
            ),
        )
