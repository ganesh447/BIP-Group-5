# lidar.py
from pyrplidar import PyRPlidar

class Lidar:
    def __init__(self, port="/dev/ttyACM1", baud=115200):
        self.lidar = PyRPlidar()
        self.port = port
        self.baud = baud

    def start(self):
        try:
            self.lidar.connect(port=self.port, baudrate=self.baud, timeout=3)
            self.lidar.start_scan()
            self.lidar.set_motor_pwm(600)
            print("LIDAR started")
        except Exception as e:
            print("LIDAR error:", e)

    def get_scan(self):
        try:
            return next(self.lidar.iter_scans())
        except:
            return None

    def stop(self):
        self.lidar.stop()
        self.lidar.disconnect()
        print("LIDAR stopped")