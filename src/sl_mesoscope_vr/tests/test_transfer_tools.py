import tempfile
import shutil
from pathlib import Path
import pytest
from src.sl_mesoscope_vr.transfer_tools import transfer_directory, calculate_local_checksum


@pytest.fixture
def setup_test_dirs():
    """Creates a temporary source and destination directory for testing."""
    temp_src = tempfile.TemporaryDirectory()
    temp_dst = tempfile.TemporaryDirectory()
    src_path = Path(temp_src.name) / "test_session"
    dst_path = Path(temp_dst.name) / "test_session_copy"
    src_path.mkdir(parents=True, exist_ok=True)

    # Create sample subdirectories and files
    (src_path / "subdir1").mkdir()
    (src_path / "subdir2").mkdir()

    (src_path / "file1.txt").write_text("This is a test file 1.")
    (src_path / "subdir1" / "file2.txt").write_text("This is file 2 in subdir1.")
    (src_path / "subdir2" / "file3.txt").write_text("This is file 3 in subdir2.")

    yield src_path, dst_path

    # Cleanup
    temp_src.cleanup()
    temp_dst.cleanup()


def test_transfer_local(setup_test_dirs):
    """Tests transferring a directory using the 'local' transfer type."""
    src_path, dst_path = setup_test_dirs

    # Compute and print initial checksum
    src_checksum = calculate_local_checksum(src_path)
    print(f"Source checksum: {src_checksum}")

    # Print per-file checksums before transfer
    for file in src_path.rglob('*'):
        if file.is_file():
            print(f"Source file {file}: {calculate_local_checksum(file)}")

    # Perform transfer
    transfer_directory(source=src_path, destination=dst_path, transfer_type='local', parallel=False)

    # Check if source is removed
    assert not src_path.exists(), "Source directory was not deleted after transfer."

    # Compute and print destination checksum
    checksum_file = dst_path / "ax_checksum.txt"
    if checksum_file.exists():
        checksum_file.unlink()


    dest_checksum = calculate_local_checksum(dst_path, ignore_files=["ax_checksum.txt"])
    print(f"Destination checksum: {dest_checksum}")

    # Print per-file checksums after transfer
    for file in dst_path.rglob('*'):
        if file.is_file():
            print(f"Destination file {file}: {calculate_local_checksum(file)}")

    # Verify checksum matches
    assert src_checksum == dest_checksum, "Checksums do not match after transfer."
