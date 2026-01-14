# main.py
from lidar import Lidar
from detection import find_obstacle
from fsm import SimpleFSM
from lane_placeholder import get_steering
from jetbot import Robot
import time

def main():
    robot = Robot()
    lidar = Lidar()
    fsm = SimpleFSM(robot)

    lidar.start()

    print("Starting... Press Ctrl+C to stop")

    try:
        while True:
            scan = lidar.get_scan()
            has_obs, where = find_obstacle(scan)

            steering = get_steering()          # ← will be your model later

            fsm.update(has_obs, where, steering)

            time.sleep(0.07)  # ≈ 14 times per second

    except KeyboardInterrupt:
        print("\nStopped")
    finally:
        lidar.stop()
        robot.stop()

if __name__ == "__main__":
    main()