"""This module provides methods for moving data between the local machine, the NAS drive (SMB protocol) and the Sun lab
BioHPC cluster (SFTP protocol).
"""

import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from .packaging_tools import calculate_directory_checksum
from ataraxis_base_utilities import console, ensure_directory_exists


def _transfer_file(src_file: Path, source_directory: Path, dest_directory: Path) -> None:
    relative = src_file.relative_to(source_directory)
    dest_file = dest_directory / relative
    shutil.copy2(src_file, dest_file)


def transfer_directory(
    source: Path,
    destination: Path,
    num_processes: int = 1,
    remove_sources: bool = False,
    verify_integrity: bool = True,
) -> None:
    """ Transfers an entire directory tree from source to destination while preserving the folder structure.

    Args:
        source: The path to the directory that needs to be moved.
        destination: The path to the destination directory where to move the source directory.
        parallel: If True, parallelized the transfer process by moving multiple files at the same time.

    Notes:
        This method recreates the moved directory hierarchy on the destination if the hierarchy does not exist. This is
        done before moving the files.

        If source directory does not have an ax_checksum file that stores the xxHash-128 checksum at the highest level,
        generates directory checksum before moving any files. If the file does have a checksum, uses the existing
        checksum for transfer verification (if enabled).
    """
    if not source.exists():
        message = f"Unable to move the directory {source}, as it does not exist."
        console.error(message=message, error=FileNotFoundError)

    # Ensures the destination root directory exists.
    ensure_directory_exists(destination)

    # Collects all items (files and directories) in the source directory.
    all_items = tuple(source.rglob("*"))

    # Recreates directory structure on destination
    for item in sorted(all_items, key=lambda x: len(x.relative_to(source).parts)):
        if item.is_dir():
            dest_dir = destination / item.relative_to(source)
            dest_dir.mkdir(parents=True, exist_ok=True)

    file_list = [item for item in all_items if item.is_file()]
    if num_processes > 1:
        with ThreadPoolExecutor(max_workers=num_processes) as executor:
            futures = {executor.submit(_transfer_file, file, source, destination): file for file in file_list}
            for future in as_completed(futures):
                # Propagate any exceptions from the file transfer.
                future.result()
    else:
        for file in tqdm(file_list, desc="Transferring files", unit="file"):
            _transfer_file(file, source, destination)

    # Verifies the integrity of the transferred directory by rerunning xxHash-129 calculation.
    destination_checksum = calculate_directory_checksum(directory=destination, batch=False, save_checksum=False)
    source_checksum = str("")  # TODO: Add ax_checksum loading

    # If checksums match, removes the source directory if this was requested via arguments
    if remove_sources:
        if verify_integrity and (source_checksum == destination_checksum):
            # Delete the source directory
            shutil.rmtree(source)
            print("Transfer successful. Source directory removed.")
            return
        else:
            print("Checksum mismatch! Transfer may be corrupted. Source directory was NOT removed.")


# --- Example usage ---
if __name__ == "__main__":
    transfer_directory(
        source=Path("/path/to/source_session"),
        destination=Path("/path/to/destination_session"),
        parallel=True,  # Set to True to transfer files concurrently.
    )
