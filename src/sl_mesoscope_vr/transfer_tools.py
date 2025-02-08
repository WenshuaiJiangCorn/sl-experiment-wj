"""This module provides methods for moving data between the local machine, the NAS drive (SMB protocol) and the Sun lab
BioHPC cluster (SFTP protocol).
"""
import os
import shutil
import hashlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from src.sl_mesoscope_vr.packaging_tools import calculate_directory_checksum
import xxhash

def calculate_local_checksum(path: Path) -> str:
    """Uses the faster xxHash3-128 checksum method from packaging_tools.py"""
    if path.is_dir():
        return calculate_directory_checksum(path)
    elif path.is_file():
        # ✅ 파일 단위로 xxHash 체크섬 계산
        with open(path, "rb") as f:
            hasher = xxhash.xxh3_128()
            for chunk in iter(lambda: f.read(1024 * 1024 * 8), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    else:
        raise ValueError(f"Invalid path: {path}")

# --- Main transfer function ---
def transfer_directory(source: Path,
                       destination: Path,
                       transfer_type: str = 'local',
                       parallel: bool = False,
                       chunk_size: int = 1024 * 1024 * 32,
                       sftp_client=None) -> None:
    """
    Transfers an entire directory tree from source to destination while preserving the folder structure.

    Parameters:
      - source (Path): Local source directory.
      - destination (Path): Destination directory. For NAS this is the remote path; for local transfer it is a local path.
      - transfer_type (str): One of:
            'local'  - regular filesystem copy;
            'nas'    - use the NAS upload function (requires that you have already set up your SMB connection);
            'sftp'   - use an SFTP client (you must supply an sftp_client).
      - parallel (bool): If True, copies files concurrently.
      - chunk_size (int): The size (in bytes) of file chunks when reading/writing large files.
      - sftp_client: A paramiko SFTPClient instance (required if transfer_type=='sftp').

    Process:
      1. Recreates the directory structure at the destination.
      2. Transfers each file (using the appropriate method).
      3. Calculates checksums on both source and destination.
      4. If the checksums match, writes an 'ax_checksum.txt' file in the destination
         (containing metadata and the checksum) and then removes the source directory.
    """
    source = Path(source)
    destination = Path(destination)

    if not source.exists():
        raise FileNotFoundError(f"Source directory {source} does not exist.")

    # Ensure the destination root exists.
    destination.mkdir(parents=True, exist_ok=True)

    # List all items (files and directories) in the source.
    all_items = list(source.rglob('*'))

    # --- Step 1. Recreate directory structure on destination ---
    for item in sorted(all_items, key=lambda x: len(x.relative_to(source).parts)):
        if item.is_dir():
            dest_dir = destination / item.relative_to(source)
            dest_dir.mkdir(parents=True, exist_ok=True)

    # --- Step 2. Transfer all files ---
    def transfer_file(src_file: Path):
        relative = src_file.relative_to(source)
        dest_file = destination / relative
        if transfer_type == 'local':
            shutil.copy2(src_file, dest_file)
        elif transfer_type == 'nas':
            # Import the NAS uploader (make sure your SMB session is set up first)
            from transfer_tools import upload_to_nas
            upload_to_nas(src_file, dest_file, chunk_size)
        elif transfer_type == 'sftp':
            if sftp_client is None:
                raise ValueError("sftp_client must be provided for sftp transfer_type")
            # For simplicity, using sftp_client.put (for large files you may want to implement chunking)
            sftp_client.put(str(src_file), str(dest_file))
        else:
            raise ValueError(f"Unknown transfer type: {transfer_type}")

    file_list = [item for item in all_items if item.is_file()]
    if parallel:
        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(transfer_file, file): file for file in file_list}
            for future in as_completed(futures):
                # Propagate any exceptions from the file transfer.
                future.result()
    else:
        for file in tqdm(file_list, desc="Transferring files", unit="file"):
            transfer_file(file)

    # --- Step 3. Verify transfer by calculating checksums ---
    print("Calculating source checksum...")
    src_checksum = calculate_local_checksum(source)


    print("Calculating destination checksum...")
    if transfer_type == 'local':
        dest_checksum = calculate_local_checksum(destination)
    elif transfer_type == 'nas':
        from transfer_tools import calculate_nas_checksum
        dest_checksum = calculate_nas_checksum(destination)
    elif transfer_type == 'sftp':
        # For SFTP you would need to implement a remote checksum calculation
        raise NotImplementedError("Checksum calculation for sftp transfer_type is not implemented.")
    else:
        raise ValueError(f"Unknown transfer type: {transfer_type}")

    print(f"Source checksum:      {src_checksum}")
    print(f"Destination checksum: {dest_checksum}")

    # --- Step 4. If checksums match, write ax_checksum.txt and remove the source ---
    if src_checksum == dest_checksum:
        checksum_file = destination / "ax_checksum.txt"
        with open(checksum_file, "w") as f:
            f.write(f"Checksum: {dest_checksum}\n")
            f.write(f"Source: {source}\n")
            f.write(f"Destination: {destination}\n")
        # Delete the source directory
        shutil.rmtree(source)
        print("Transfer successful. Source directory removed.")
    else:
        print("Checksum mismatch! Transfer may be corrupted. Source directory was NOT removed.")


# --- Example usage ---
if __name__ == '__main__':
    # For a local-to-local copy:
    transfer_directory(
        source=Path("/path/to/source_session"),
        destination=Path("/path/to/destination_session"),
        transfer_type='local',
        parallel=True  # Set to True to transfer files concurrently.
    )

    # For NAS transfers, you must first set up the SMB connection:
    #   from transfer_tools import setup_smb_connection
    #   setup_smb_connection(server="YOUR_NAS_IP", username="your_username", password="your_password")
    # Then call transfer_directory() with transfer_type='nas'

    # For SFTP transfers, you must create an SFTP client:
    #   sftp = setup_ssh_client(hostname="host", username="user", password="pass").open_sftp()
    # Then call transfer_directory(..., transfer_type='sftp', sftp_client=sftp)
