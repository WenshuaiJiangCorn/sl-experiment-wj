"""This module provides methods for packaging session runtime data for transmission over the network. The methods from
this module work in tandem with methods offered by transfer_tools.py to ensure the integrity of the transferred data.
"""

import os
from pathlib import Path
from functools import partial
from concurrent.futures import ProcessPoolExecutor, as_completed

from tqdm import tqdm
import xxhash


def _calculate_file_checksum(base_directory: Path, file_path: Path) -> tuple[str, bytes]:
    """Calculates xxHash3-128 checksum for a single file and its path relative to the base directory.

    This function is passed to parallel workers used by the calculate_directory_hash() method that iteratively
    calculates the checksum for all files inside a directory. Each call to this function returns the checksum for the
    target file, which includes both the contents of the file and its path relative to the base directory.

    Args:
        base_directory: The path to the base (root) directory for which.
        file_path: Absolute path to the target file.

    Returns:
        A tuple with two elements. The first element is the path to the file relative to the base directory. The second
        element is the xxhash3-128 checksum that covers the relative path and the contents of the file.
    """
    # Initializes the hashsum object.
    checksum = xxhash.xxh3_128()

    # Encodes the relative path and appends it to the checksum. This ensures that the hashsum reflects both the state
    # of individual files and the layout of the overall encoded directory structure.
    relative_path = str(file_path.relative_to(base_directory))
    checksum.update(relative_path.encode())

    # Extends the checksum to reflect the file data state. Uses 8 MB chunks to avoid excessive RAM hogging at the cost
    # of slightly reduced throughput.
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024 * 8), b""):
            checksum.update(chunk)

    # Returns both path and file checksum. Although the relative path information is already encoded in the hashsum, the
    # relative path information is re-encoded at the directory level to protect against future changes to the per-file
    # hashsum calculation logic. It is extra work, but it improves the overall checksum security.
    return relative_path, checksum.digest()


def calculate_directory_checksum(
    directory: Path, num_processes: int | None = None, batch: bool = False, save_checksum: bool = True
) -> str:
    """Calculates xxHash3-128 checksum for the input directory, which includes the data of all contained files and
    the directory structure information.

    This function is used to generate a checksum for each experimental session directory. Checksums are used to
    verify the session data integrity during transmission between the PC that acquired the data and longer term storage
    locations, such as the NAS or the lab processing server. The function can be configured to write the generated
    checksum as a hexadecimal string to the ax_checksum.txt file stored at the highest level of the input directory.

    Note:
        All data transfer methods from this library automatically verify data integrity when it is uploaded to or
        downloaded from long-term storage. We assume that long-term storage comes with the necessary parity and data
        integrity verification methods to minimize data corruption risks.

        This method uses multiprocessing to efficiently parallelize checksum calculation for multiple files. In
        combination with xxHash3, this achieves a significant speedup over more common checksums, such as MD5 and
        SHA256. Note that xxHash3 is not suitable for security purposes and is only used to ensure data integrity.

        The method notifies the user about the checksum calculation process via the terminal.

        The returned checksum accounts for both the contents of each file and the layout of the input directory
        structure.

    Args:
        directory: The Path to the directory to be checksummed.
        num_processes: The number of CPU processes to use for parallelizing checksum calculation. If set to None, the
            function defaults to using (logical CPU count - 4).
        batch: Determines whether the function is called as part of batch-processing multiple directories. This is used
            to optimize progress reporting to avoid cluttering the terminal.
        save_checksum: Determines whether the checksum should be saved (written to) a .txt file.

    Returns:
        The xxHash3-128 checksum for the input directory as a hexadecimal string.
    """
    # Determines the number of parallel processes to use.
    if num_processes is None:
        num_processes = max(1, os.cpu_count() - 4)  # type: ignore

    # Determines the path to each file inside the input directory structure and sorts them for consistency
    path: Path
    files = sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.stem != "ax_checksum" and path.suffix != ".txt"  # Excludes checksum files
    )

    # Precreates the directory checksum
    checksum = xxhash.xxh3_128()

    # Process files in parallel
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        # Creates the partial function with fixed base_directory (the first argument of _calculate_file_hash())
        process_file = partial(_calculate_file_checksum, directory)

        # Submits all tasks to be executed in parallel
        future_to_path = {executor.submit(process_file, file): file for file in files}

        # Collects results as they complete
        results = []
        if not batch:
            with tqdm(
                total=len(files), desc=f"Calculating checksum for {Path(*directory.parts[-6:])}", unit="files"
            ) as pbar:
                for future in as_completed(future_to_path):
                    results.append(future.result())
                    pbar.update(1)
        else:
            # For batch mode, uses a direct list comprehension with as_completed. This avoids the overhead of progress
            # tracking while maintaining parallel processing, avoiding terminal clutter in batched contexts.
            results = [future.result() for future in as_completed(future_to_path)]

        # Sorts results for consistency and combines them into the final checksum
        for file_path, file_checksum in sorted(results):
            checksum.update(file_path.encode())
            checksum.update(file_checksum)

    checksum_hexstr = checksum.hexdigest()

    # Writes the hash to ax_checksum.txt in the root directory
    if save_checksum:
        checksum_path = directory / "ax_checksum.txt"
        with open(checksum_path, "w") as f:
            f.write(checksum_hexstr)

    return checksum_hexstr
