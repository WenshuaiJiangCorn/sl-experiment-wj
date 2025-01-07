"""This module provides methods for moving data between the local machine, the NAS drive (SMB protocol) and the Sun lab
BioHPC cluster (SFTP protocol)."""

import paramiko
from smbclient import register_session, scandir, mkdir, open_file, shutil
from pathlib import Path
from tqdm import tqdm
import hashlib


def setup_ssh_client(hostname: str, username: str, password: str = None,
                     key_filename: str = None) -> paramiko.SSHClient:
    """Setup SSH client with either password or key authentication"""
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(hostname, username=username, password=password,
                   key_filename=key_filename)
    return client


def sftp_operations(hostname: str, username: str, password: str = None):
    with setup_ssh_client(hostname, username, password) as ssh:
        sftp = ssh.open_sftp()

        # Upload
        sftp.put('local_file.txt', '/remote/path/file.txt')

        # Download
        sftp.get('/remote/path/file.txt', 'local_file.txt')

        # List directory
        for entry in sftp.listdir_attr('/remote/path'):
            print(f"{entry.filename}: {entry.st_size} bytes")

        # Remove file
        sftp.remove('/remote/path/file.txt')

        sftp.close()


def setup_smb_connection(server: str, username: str, password: str) -> None:
    """Establishes the connection to the NAS over the SMB protocol.

    Call this function before calling other NAS-manipulation functions. This function registers the local client with
    the NAS authentication server, which enables other NAS-related functions to access the data stored on the NAS.

    Args:
        server: The IP address or hostname of the NAS.
        username: The username for authentication.
        password: The password for authentication.

    """
    register_session(server, username=username, password=password, encrypt=False)


def download_from_nas(remote_path: Path, local_path: Path, chunk_size: int = 1024 * 1024 * 32) -> None:
    """Downloads a file or directory from NAS to the local machine.

    The function downloads individual files < 1 GB as a single operation and breaks larger files into chunks.
    For directories, it maintains the folder structure and recursively downloads all contents.

    Args:
        remote_path: The path of the file/directory on the NAS.
        local_path: The path where to download on local machine.
        chunk_size: Size of chunks to read/write in bytes. Chunking is only applied to files larger than 1 GB.
    """
    # Checks if remote_path is a directory
    try:
        # This raises an error if remote_path is not a directory
        entries = list(scandir(remote_path))

        # Creates local directory if it doesn't exist
        local_dir = Path(local_path)
        local_dir.mkdir(parents=True, exist_ok=True)

        # Recursively downloads all contents from the source directory
        for entry in entries:
            remote_item_path = remote_path.joinpath(entry.name)
            local_item_path = local_dir.joinpath(entry.name)

            if entry.is_file():
                download_from_nas(remote_item_path, local_item_path, chunk_size)
            else:
                # Recursively handles subdirectories
                download_from_nas(remote_item_path, local_item_path, chunk_size)

    except NotADirectoryError:
        # Files are downloaded either as chunks or via one operation (for small files)
        file_size = Path(remote_path).stat().st_size

        # Files less than 1GB are downloaded in one go
        if file_size < 1024 * 1024 * 1024:  # 1GB
            with open_file(remote_path, mode='rb') as remote_file:
                with open(local_path, 'wb') as local_file:
                    local_file.write(remote_file.read())
            return

        # Larger files are broken into chunks
        with open_file(remote_path, mode='rb') as remote_file:
            with open(local_path, 'wb') as local_file:
                with tqdm(total=file_size, unit='B', unit_scale=True,
                          desc=f"Downloading {Path(remote_path).name}:") as pbar:
                    while True:
                        chunk = remote_file.read(chunk_size)
                        if not chunk:
                            break
                        local_file.write(chunk)
                        pbar.update(len(chunk))


def upload_to_nas(local_path: Path, remote_path: Path, chunk_size: int = 1024 * 1024 * 32) -> None:
    """Uploads a file or directory from the local machine to NAS.

    The function uploads individual files < 1 GB as a single operation and breaks larger files into chunks.
    For directories, it maintains the folder structure and recursively uploads all contents.

    Args:
        local_path: The path of the file/directory on local machine.
        remote_path: The path where to upload on NAS.
        chunk_size: Size of chunks to read/write in bytes (default 32MB). Chunking is only applied to files larger
            than 1 GB.
    """
    local_path = Path(local_path)

    # If the input is a directory, recurses over its contents and uploads them to NAS.
    if local_path.is_dir():
        # Creates remote directory if it doesn't exist
        try:
            mkdir(remote_path)
        except FileExistsError:
            pass

        # Recursively uploads all contents
        for entry in local_path.iterdir():
            remote_item_path = remote_path.joinpath(entry.name)
            local_item_path = Path(entry)

            if entry.is_file():
                upload_to_nas(local_item_path, remote_item_path, chunk_size)
            else:
                # Recursively handles subdirectories
                upload_to_nas(local_item_path, remote_item_path, chunk_size)

    # File processing
    else:
        file_size = local_path.stat().st_size

        # Files less than 1GB are uploaded in one go
        if file_size < 1024 * 1024 * 1024:  # 1GB
            with open(local_path, 'rb') as local_file:
                with open_file(remote_path, mode='wb') as remote_file:
                    remote_file.write(local_file.read())
            return

        # Larger files are broken into chunks
        with open(local_path, 'rb') as local_file:
            with open_file(remote_path, mode='wb') as remote_file:
                with tqdm(total=file_size, unit='B', unit_scale=True, desc=f"Uploading {local_path.name}:") as pbar:
                    while True:
                        chunk = local_file.read(chunk_size)
                        if not chunk:
                            break
                        remote_file.write(chunk)
                        pbar.update(len(chunk))


def remove_from_nas(remote_path: Path) -> None:
    """Removes a file or directory from NAS.

    For directories, recursively removes all contents before removing the directory itself.
    For files, removes them directly.

    Args:
       remote_path: The path of the file or directory on the NAS to be removed.
    """
    try:
        # Checks if it's a directory by attempting to list contents
        _ = list(scandir(remote_path))
        shutil.rmtree(remote_path)  # Removes directory recursively
    except NotADirectoryError:
        shutil.remove(remote_path)  # If the path points to a file, directly removes the file
    except FileNotFoundError:
        return  # If the file or directory does not exist, no action is taken


def calculate_nas_checksum(remote_path: Path, chunk_size: int = 1024 * 1024 * 32) -> str:
    """Calculates MD5 checksum of a file or directory on NAS.

    For directories, calculates checksum based on both structure and content.
    For zip files or single files, calculates straightforward MD5.

    Args:
        remote_path: Path to file or directory on NAS.
        chunk_size: Size of chunks to read (default 32MB).

    Returns:
        MD5 checksum has.
    """

    # Pre-initializes the checksum
    md5_hash = hashlib.md5()

    # Directory processing
    try:
        # Checks if the path points to a directory as this raises errors for Files
        entries = list(scandir(remote_path))

        # Recursively traverses the sorted directory and calculates the checksum for each discovered file
        for entry in sorted(entries, key=lambda e: e.name):
            full_path = remote_path.joinpath(entry.name)

            if entry.is_file():
                # Adds the relative path to hash
                md5_hash.update(str(entry.name).encode())

                # Adds file contents to hash
                with open_file(full_path, 'rb') as f:
                    while True:
                        chunk = f.read(chunk_size)
                        if not chunk:
                            break
                        md5_hash.update(chunk)
            else:
                # Recursively processes subdirectories
                subdir_hash = calculate_nas_checksum(full_path, chunk_size)
                md5_hash.update(subdir_hash.encode())

    # For file inputs, directly calculates the checksum for the file.
    except NotADirectoryError:
        with open_file(remote_path, 'rb') as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                md5_hash.update(chunk)

    return md5_hash.hexdigest()

# Example usage:
# server = "//YOUR_NAS_IP"  # e.g., "//192.168.1.100"
# username = "your_username"
# password = "your_password"

# Setup connection
# setup_smb_connection(server, username, password)

# Examples of operations
# Download
# download_from_nas("//server/share/remote_file.txt", "local_file.txt")
#
# # Upload
# upload_to_nas("local_file.txt", "//server/share/remote_file.txt")
#
# # Remove
# remove_from_nas("//server/share/remote_file.txt")
