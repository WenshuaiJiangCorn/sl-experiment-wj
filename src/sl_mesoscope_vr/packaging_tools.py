"""This module provides methods for packaging experimental session data for transmission to the NAS or Sun lab
data cluster. The methods from this module work in tandem with methods offered by transfer_tools.py."""

import xxhash
from ataraxis_time import PrecisionTimer
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from pathlib import Path
from tqdm import tqdm
import os


def calculate_file_hash(base_dir: Path, filepath: Path) -> tuple[str, bytes]:
    """Calculates xxHash3-64 for a single file and its path.

    Args:
        base_dir: Base directory for calculating relative paths
        filepath: Path to the file to hash

    Returns:
        Tuple of (relative path, file's xxHash digest)
    """
    xxh = xxhash.xxh3_128()

    # Add the relative path
    rel_path = str(filepath.relative_to(base_dir))
    xxh.update(rel_path.encode())

    # Add file contents
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024 * 128), b''):
            xxh.update(chunk)

    return (rel_path, xxh.digest())


def calculate_directory_hash(directory: Path, num_processes: int = None) -> str:
    """Calculates xxHash64 checksum of all files in the directory and subdirectories.

    Uses multiprocessing to parallelize file reading and hashing while maintaining
    consistent directory structure checksumming.

    Args:
        directory: Path to the directory to be checksummed
        num_processes: Number of processes to use (default: CPU count - 4)

    Returns:
        The human-readable hexadecimal string representation of the xxHash.
    """
    if num_processes is None:
        num_processes = max(1, os.cpu_count() - 4)

    # Get all files first and sort them for consistency
    files = sorted(p for p in directory.rglob('*') if p.is_file())

    # Create the final hash
    final_hash = xxhash.xxh3_128()

    # Process files in parallel
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        # Create partial function with fixed base_dir
        process_file = partial(calculate_file_hash, directory)

        # Submit all tasks
        future_to_path = {
            executor.submit(process_file, file): file
            for file in files
        }

        # Collect results as they complete
        results = []
        with tqdm(total=len(files), desc="Calculating directory checksum", unit="files") as pbar:
            for future in as_completed(future_to_path):
                results.append(future.result())
                pbar.update(1)

        # Sort results for consistency and combine
        for rel_path, file_digest in sorted(results):
            final_hash.update(rel_path.encode())
            final_hash.update(file_digest)

    return final_hash.hexdigest()


if __name__ == "__main__":
    timer = PrecisionTimer('s')

    input_path = Path('/media/Data/2022_01_25/1')
    output_path = Path('/media/Data/2022_01_25/1_7z')

    timer.reset()
    hashsum = calculate_directory_hash(input_path)
    elapsed = timer.elapsed
    print(f"Hashsum {hashsum} took {elapsed} seconds.")
