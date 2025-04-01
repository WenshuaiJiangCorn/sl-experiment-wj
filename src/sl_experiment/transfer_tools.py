"""This module provides methods for moving data between the local machine, the ScanImage (Mesoscope) PC, the Synology
NAS drive, and the lab BioHPC cluster. All methods in this module expect that the destinations and sources are mounted
on the host file-system and use the os tools for moving the data.
"""

import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm
from ataraxis_base_utilities import console, ensure_directory_exists

from .packaging_tools import calculate_directory_checksum


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
    relative = src_file.relative_to(source_directory)
    dest_file = dest_directory / relative
    shutil.copy2(src_file, dest_file)


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
    if not source.exists():
        message = f"Unable to move the directory {source}, as it does not exist."
        console.error(message=message, error=FileNotFoundError)

    # Ensures the destination root directory exists.
    ensure_directory_exists(destination)

    # Collects all items (files and directories) in the source directory.
    all_items = tuple(source.rglob("*"))

    # Loops over all items (files and directories). Adds files to the file_list variable. Uses directories to reinstate
    # the source subdirectory hierarchy in the destination directory.
    file_list = []
    for item in sorted(all_items, key=lambda x: len(x.relative_to(source).parts)):
        # Recreates directory structure on destination
        if item.is_dir():
            dest_dir = destination / item.relative_to(source)
            dest_dir.mkdir(parents=True, exist_ok=True)
        # Also builds the list of files to be moved
        else:  # is_file()
            file_list.append(item)

    # Copies the data to the destination. For parallel workflows, the method uses the ThreadPoolExecutor to move
    # multiple files at the same time. Since I/O operations do not hold GIL, we do not need to parallelize with
    # Processes here.
    if num_threads > 1:
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = {executor.submit(_transfer_file, file, source, destination): file for file in file_list}
            for future in tqdm(
                as_completed(futures),
                total=len(file_list),
                desc=f"Transferring files to {Path(*destination.parts[-6:])}",
                unit="file",
            ):
                # Propagates any exceptions from the file transfer.
                future.result()
    else:
        for file in tqdm(file_list, desc=f"Transferring files to {Path(*destination.parts[-6:])}", unit="file"):
            _transfer_file(file, source, destination)

    # Verifies the integrity of the transferred directory by rerunning xxHash3-128 calculation.
    if verify_integrity:
        destination_checksum = calculate_directory_checksum(directory=destination, batch=False, save_checksum=False)
        with open(file=source.joinpath("ax_checksum.txt"), mode="r") as local_checksum:
            message = (
                f"Checksum mismatch detected when transferring {Path(*source.parts[-6:])} to "
                f"{Path(*destination.parts[-6:])}! The data was likely corrupted in transmission. User intervention "
                f"required."
            )
            if not destination_checksum == local_checksum.readline().strip():
                console.error(message=message, error=RuntimeError)
