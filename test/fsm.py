# fsm.py
from jetbot import Robot

class SimpleFSM:
    def __init__(self, robot):
        self.robot = robot
        self.state = "NORMAL"
        self.last_avoid_time = time.time()

    def update(self, has_obstacle, obstacle_location, lane_steering=0.0):
        if self.state == "NORMAL":
            if has_obstacle:
                print(f"Obstacle! ({obstacle_location}) → avoiding")
                self.state = "AVOID"
            else:
                # Normal driving - use lane following when ready
                speed = 0.30
                turn = lane_steering * 0.4
                self.robot.set_motors(speed + turn, speed - turn)

        elif self.state == "AVOID":
            self.robot.forward(0.20)
            if obstacle_location in ['front', 'left']:
                self.robot.right(0.40)    # turn right to avoid
            else:
                self.robot.left(0.40)

            # Simple timeout - go back to normal after 2.5 seconds
            # (you can make this more sophisticated later)
            import time
            if time.time() - self.last_avoid_time > 2.5:
                self.state = "NORMAL"
                print("→ Back to normal driving")
            self.last_avoid_time = time.time()  # reset timer

    def stop(self):
        self.robot.stop()