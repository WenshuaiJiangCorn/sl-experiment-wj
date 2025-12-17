import time
import subprocess

_CONTROLLER_PORT = "COM3"  # Update this to your controller's port


def complete_reset():
    """Completely resets the connection."""
    # 1. Kill Python processes (Windows)
    subprocess.run(["taskkill", "/F", "/IM", "python.exe"], check=False, stderr=subprocess.DEVNULL)
    time.sleep(1)

    # 2. Reset via serial
    try:
        import serial

        ser = serial.Serial(_CONTROLLER_PORT, 115200)
        ser.setDTR(False)
        time.sleep(0.5)
        ser.setDTR(True)
        ser.close()
    except:
        pass

    time.sleep(3)

    print("Reset complete. Try running your test now.")
