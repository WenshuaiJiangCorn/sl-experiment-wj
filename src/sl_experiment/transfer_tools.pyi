from pathlib import Path

from .packaging_tools import calculate_directory_checksum as calculate_directory_checksum

def _transfer_file(src_file: Path, source_directory: Path, dest_directory: Path) -> None:
    """Copies the input file from the source directory to the destination directory while preserving the file metadata.

    This is a worker method used by the transfer_directory() method to move multiple files in parallel.

    Notes:
        If the file is found under a hierarchy of subdirectories inside the input source_directory, that hierarchy will
        be preserved in the destination directory.

    Args:
        src_file: The file to be copied.
        source_directory: The root directory where the file is located.
        dest_directory: The destination directory where to move the file.
    """

def transfer_directory(source: Path, destination: Path, num_threads: int = 1, verify_integrity: bool = True) -> None:
    """Copies the contents of the input directory tree from source to destination while preserving the folder
    structure.

    This function is used to assemble the experimental data from all remote machines used in the acquisition process.
    It is also used to transfer the preprocessed raw data from the local machine to the NAS and the Sun lab BioHPC
    cluster.

    Notes:
        This method recreates the moved directory hierarchy on the destination if the hierarchy does not exist. This is
        done before copying the files.

        The method executes a multithreading copy operation. It does not clean up the source files. That job is handed
        to the SessionData class, which also executes the copy operation simultaneously for multiple destinations.

        If the method is configured to verify transferred file integrity, it reruns the xxHash3-128 checksum calculation
        and compares the returned checksum to the one stored in the source directory. The method assumes that all input
        directories contain the 'ax_checksum.txt' file that stores the 'source' directory checksum at the highest level
        of the input directory tree.

    Args:
        source: The path to the directory that needs to be moved.
        destination: The path to the destination directory where to move the contents of the source directory.
        num_threads: The number of threads to use for parallel file transfer. This number should be set depending on the
            type of transfer (local or remote) and is not guaranteed to provide improved transfer performance. For local
            transfers, setting this number above 1 will likely provide a performance boost. For remote transfers using
            a single TCP / IP socket (such as non-multichannel SMB protocol), the number should be set to 1.
        verify_integrity: Determines whether to perform integrity verification for the transferred files. Note,
            integrity verification is a time-consuming process and generally would not be a concern for most runtimes.
            Therefore, it is often fine to disable this option to optimize method runtime speed.

    Raises:
        RuntimeError: If the transferred files do not pass the xxHas3-128 checksum integrity verification.
    """
