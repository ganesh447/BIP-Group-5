# lidar/obstacle_detector.py
def detect_obstacles(scan_points, 
                     min_dist=0.50, 
                     front_cone_deg=60, 
                     side_cone_deg=90):
    """
    Returns tuple: (has_obstacle: bool, direction: str)
    direction can be: 'none', 'left', 'center', 'right'
    """
    left = []
    center = []
    right = []

    for quality, angle_deg, distance_mm in scan_points or []:
        distance = distance_mm / 1000.0
        if distance <= 0 or distance > 5.0:
            continue

        abs_angle = abs(angle_deg)
        if abs_angle <= front_cone_deg / 2:
            center.append(distance)
        elif angle_deg < -side_cone_deg:
            left.append(distance)
        elif angle_deg > side_cone_deg:
            right.append(distance)

    min_center = min(center) if center else float('inf')
    min_left   = min(left)   if left   else float('inf')
    min_right  = min(right)  if right  else float('inf')

    if min_center < min_dist:
        return True, "center"
    if min_left < min_dist:
        return True, "left"
    if min_right < min_dist:
        return True, "right"
    
    return False, "none"