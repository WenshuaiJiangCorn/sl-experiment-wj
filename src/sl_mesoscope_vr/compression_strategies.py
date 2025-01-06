import tifffile
import dask.bag as db
from dask.diagnostics import ProgressBar
from pathlib import Path
from ataraxis_base_utilities import LogLevel, console, ensure_directory_exists
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from typing import Any
import hashlib
import zipfile
from tqdm import tqdm

def extract_frames_from_tiff_stack(image_directory: Path, stack_size: int = 500, remove_sources: bool = False) -> None:
    """Loops over all multi-frame .tiff files in the input directory and extracts all individual frames as .tiff files.

    This function is used as a preprocessing step for mesoscope=acquired data, preparing it to be further processed
    via suite2p. In addition to extracting data as single frame tiffs, the function also compresses each tiff with an
    efficient 'lerc' scheme and removes unnecessary tiff metadata.

    Notes:
        This function is specifically calibrated to work with .tiff stacks produced by the scanimage matlab software.
        Critically, these stacks are named using '__' to separate session and stack number from the rest of the
        file name, and the stack number is always found last, e.g.: 'Tyche-A7_2022_01_25_1__00001_00067.tif'. If the
        input tiff files do not follow this naming convention, the function will not work as expected.

        This function assumes that scanimage buffers frames until the stack_size number of frames is available and then
        saves the frames as a .tiff stack. Therefore, all processed stacks from the same session have to be full, except
        for the last stack of the session.

        This function uses dask to efficiently process multiple stacks in-parallel.

    Args:
        image_directory: The directory containing the multi-frame '.tiff' files.
        stack_size: The number of frames stored inside each tiff stack, assuming the stack was filled.
        remove_sources: Determines whether to remove the original '.tiff' files after they have been processed.

    Raises:
        ValueError: If no '.tiff' or '.tif' files are found in the input directory.
    """

    # Generates a new 'mesoscope_frames' directory, to store extracted .tiff frame files.
    output_dir = image_directory.joinpath("mesoscope_frames")
    ensure_directory_exists(output_dir)

    # Finds all .tif files in the input directory (non-recursive).
    tiff_files = list(image_directory.glob("*.tif"))
    if not tiff_files:  # Uses .tiff extension as fall-back
        tiff_files = list(image_directory.glob("*.tiff"))

    # Aborts with an error if no '.tiff' files are found.
    if not tiff_files:
        message = (
            f"No TIFF files found in the input directory {image_directory}."
        )
        console.error(message=message, error=ValueError)

    def _process_file(tiff_path: Path) -> str:
        """Reads a TIFF stack and writes frames as uncompressed TIFF frames."""

        # Loads the stack into RAM
        img = tifffile.imread(str(tiff_path))

        # Uses the base stack name to compute the numeric ID for each frame. Specifically, uses the stack number and
        # the stack size parameters to determine the exact number of each extracted frame in the overall sequence saved
        # over multiple stacks.
        base_name = tiff_path.stem
        session_sequence: str = base_name.split("__")[-1]
        _, stack_number = session_sequence.split("_")
        previous_frames = (int(stack_number) - 1) * stack_size

        # Sequentially loops over each frame and extracts them as Tiffs.
        frame: np.ndarray[Any]
        for i, frame in enumerate(img, start=previous_frames + 1):
            output_path = output_dir.joinpath(f"{i:012d}.tiff")
            tifffile.imwrite(
                output_path,
                frame,
                compression='lerc',
                compressionargs={'level': 0.0},  # Lossless
                predictor=True,
                resolutionunit='NONE'  # Remove unnecessary metadata
            )

        if remove_sources:
            tiff_path.unlink()  # Removes the original tiff file after it has been processed

        # Notifies the user that the stack has been processed
        return f"Completed: {tiff_path.stem}."

    # Creates a Dask bag from the list of files
    bag = db.from_sequence(tiff_files)

    # Maps the processing function across the bag
    results = bag.map(_process_file)

    # Computes with a progress bar
    with ProgressBar():
        results = results.compute(scheduler='processes')

    # Prints results
    for res in results:
        console.echo(res, level=LogLevel.SUCCESS)


def process_file(file: Path) -> int:
    """Reads a TIFF file and returns its stack size (if it's a 3D stack)."""
    img: np.ndarray = tifffile.imread(str(file))

    # If the file is a stack (3D), return the stack size (the first dimension of the shape)
    if len(img.shape) == 3:
        return img.shape[0]
    return 0


def find_max_stack_size(root_directory: Path) -> int:
    # Recursively find all subdirectories
    subdirectories = [d for d in root_directory.rglob('*') if d.is_dir()]

    tiff_files = []

    # Collect up to 3 .tif files from each subdirectory
    for subdirectory in subdirectories:
        tiff_files_in_subdir = list(subdirectory.glob("*.tif"))[:3]
        tiff_files.extend(tiff_files_in_subdir)

    console.echo(message=f"Found {len(tiff_files)} candidate .tif files.", level=LogLevel.INFO)

    # Use ThreadPoolExecutor to process all the TIFF files in parallel
    with ThreadPoolExecutor() as executor:
        # Use tqdm to track the progress in parallel processing
        stack_sizes = list(
            tqdm(executor.map(process_file, tiff_files), total=len(tiff_files), desc="Processing files", unit="file"))

    # Get the maximum stack size from all the results
    max_stack_size = max(stack_sizes)

    # Returns the maximum size found
    return max_stack_size


def calculate_directory_md5(directory: Path) -> str:
    """Calculate MD5 hash of all files in directory and subdirectories."""
    md5_hash = hashlib.md5()

    # Walk through all files in sorted order for consistency
    for filepath in sorted(directory.rglob('*')):
        if filepath.is_file():
            # Add relative path to hash for structure awareness
            md5_hash.update(str(filepath.relative_to(directory)).encode())

            # Add file contents to hash
            with open(filepath, 'rb') as f:
                for chunk in iter(lambda: f.read(8192), b''):
                    md5_hash.update(chunk)

    return md5_hash.hexdigest()


def calculate_zip_md5(zip_path: Path) -> str:
    """Calculate MD5 hash of a zip file."""
    md5_hash = hashlib.md5()

    with open(zip_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            md5_hash.update(chunk)

    return md5_hash.hexdigest()


def zip_directory(directory: Path, output_zip: Path) -> None:
    """Zip a directory with all its contents."""
    directory = Path(directory)
    output_zip = Path(output_zip)

    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zf:
        # Walk through directory
        for filepath in directory.rglob('*'):
            if filepath.is_file():
                # Add file to zip with relative path
                zf.write(filepath, filepath.relative_to(directory))


def unzip_directory(zip_path: Path, output_dir: Path) -> None:
    """Unzip a directory to specified location."""
    zip_path = Path(zip_path)
    output_dir = Path(output_dir)

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(output_dir)


if __name__ == "__main__":
    console.enable()
    print()
    print(find_max_stack_size(Path("/mnt/nearline/Tyche")))
    # input_dir = Path("/home/cybermouse/Desktop/Data/2022_01_25/1")
    #
    # extract_frames_from_tiff_stack(input_dir, stack_size=500, remove_sources=True)
