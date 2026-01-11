from rplidar import RPLidar
import time

# Use the UART port!
PORT = '/dev/ttyTHS1'
BAUD = 115200  # Standard for RPLIDAR A1/A2

try:
    lidar = RPLidar(PORT, baudrate=BAUD)
    print("Connected! Device info:")
    print(lidar.get_info())
    print("\nHealth:")
    print(lidar.get_health())

    print("\nScanning... (should see measurements)")
    for i, scan in enumerate(lidar.iter_scans(max_buf_meas=500)):
        print(f"Scan {i+1}: {len(scan)} points")
        if i >= 5: break  # Just first few for test
        time.sleep(0.5)

except Exception as e:
    print("Error:", e)

finally:
    lidar.stop()
    lidar.stop_motor()
    lidar.disconnect()
