from pathlib import Path

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
