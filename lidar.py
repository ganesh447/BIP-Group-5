#!/usr/bin/env python
import rospy
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
import numpy as np

class LidarObstacleNode:
    def __init__(self):
        rospy.init_node('lidar_obstacle_node')

        # Publisher for obstacle state
        self.pub_state = rospy.Publisher('/obstacle/state', String, queue_size=10)

        # Publisher for RViz markers
        self.pub_marker = rospy.Publisher('/obstacle/markers', Marker, queue_size=10)

        rospy.Subscriber('/scan', LaserScan, self.scan_callback)

        self.stop_dist = 0.30
        self.avoid_dist = 0.50

    def scan_callback(self, scan):
        state, sector_points = self.process_scan(scan)
        self.pub_state.publish(String(data=state))

        # Publish RViz marker
        marker = Marker()
        marker.header.frame_id = scan.header.frame_id
        marker.header.stamp = rospy.Time.now()
        marker.ns = "obstacle_sectors"
        marker.id = 0
        marker.type = Marker.POINTS
        marker.action = Marker.ADD
        marker.scale.x = 0.05
        marker.scale.y = 0.05
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0

        for p in sector_points:
            pt = Point()
            pt.x = p[0]
            pt.y = p[1]
            pt.z = 0
            marker.points.append(pt)

        self.pub_marker.publish(marker)

        rospy.loginfo_throttle(1.0, f"Obstacle state: {state}")

    def process_scan(self, scan):
        ranges = np.array(scan.ranges)
        angles = np.linspace(scan.angle_min, scan.angle_max, len(ranges))
        ranges = np.nan_to_num(ranges, nan=np.inf, posinf=np.inf)

        # Select sectors: front, left, right
        front_min, left_min, right_min = self.min_dist(ranges, angles, -0.2, 0.2), \
                                         self.min_dist(ranges, angles, -0.6, -0.2), \
                                         self.min_dist(ranges, angles, 0.2, 0.6)

        # Convert to xy points for RViz
        sector_points = []
        for sector, a_min, a_max in zip([front_min, left_min, right_min],
                                        [-0.2, -0.6, 0.2],
                                        [0.2, -0.2, 0.6]):
            idx = (angles >= a_min) & (angles <= a_max)
            for r, a in zip(ranges[idx], angles[idx]):
                x = r * np.cos(a)
                y = r * np.sin(a)
                sector_points.append((x, y))

        if front_min < self.stop_dist:
            state = "STOP"
        elif left_min < self.avoid_dist:
            state = "AVOID_RIGHT"
        elif right_min < self.avoid_dist:
            state = "AVOID_LEFT"
        else:
            state = "CLEAR"

        return state, sector_points

    def min_dist(self, ranges, angles, a_min, a_max):
        mask = (angles >= a_min) & (angles <= a_max)
        if not np.any(mask):
            return np.inf
        return np.min(ranges[mask])

if __name__ == '__main__':
    node = LidarObstacleNode()
    rospy.spin()
