# Simple waypoint navigation with color detection
import threading
import sys, time
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from rclpy.action import ActionClient
from nav2_msgs.action import NavigateToPose
import signal


class RobotController(Node):
    def __init__(self):
        super().__init__('robot_controller')
        
        # Initialize CvBridge
        self.bridge = CvBridge()
        
        # Subscribe to camera feed
        self.image_sub = self.create_subscription(
            Image, 
            '/camera/image_raw', 
            self.image_callback, 
            10)
        
        # Publisher for velocity commands
        self.vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # Action client for navigation
        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        
        # Color detection flags
        self.red_detected = False
        self.green_detected = False
        self.blue_detected = False
        
        # List of detected colors
        self.detected_colors = []
        
        # Navigation state
        self.navigating = False
        self.mission_complete = False
        self.goal_handle = None
        self.all_waypoints_visited = False
        self.waypoints_visited_count = 0
        
        # Waypoints
        self.waypoints = [
            (-1.0, -5.0, 0),
            (4.94, -6.26, 0),
            (-3.6, 3.8, 0),
            (-5.14, -1.05, 0),
            (-4.18, -8.86, 0) # Blue
        ]
        self.current_waypoint = 0
        
        self.get_logger().info('Robot controller initialized')
    
    def image_callback(self, data):
        try:
            # Convert ROS Image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(data, 'bgr8')
            
            # Convert to HSV for color filtering
            hsv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
            
            # Create color masks
            # Red (needs two ranges because it wraps in HSV)
            lower_red1 = np.array([0, 100, 100])
            upper_red1 = np.array([10, 255, 255])
            lower_red2 = np.array([160, 100, 100])
            upper_red2 = np.array([179, 255, 255])
            red_mask1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
            red_mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
            red_mask = cv2.bitwise_or(red_mask1, red_mask2)
            
            # Green
            lower_green = np.array([40, 100, 100])
            upper_green = np.array([80, 255, 255])
            green_mask = cv2.inRange(hsv_image, lower_green, upper_green)
            
            # Blue
            lower_blue = np.array([100, 100, 100])
            upper_blue = np.array([140, 255, 255])
            blue_mask = cv2.inRange(hsv_image, lower_blue, upper_blue)
            
            # Reset current detection flags
            self.red_detected = False
            self.green_detected = False
            self.blue_detected = False
            
            # Process each color for detection
            self.red_detected = self.detect_color(cv_image, red_mask, (0, 0, 255), "Red")
            self.green_detected = self.detect_color(cv_image, green_mask, (0, 255, 0), "Green")
            self.blue_detected = self.detect_color(cv_image, blue_mask, (255, 0, 0), "Blue")
            
            # Update detected colors list (for historical tracking)
            if self.red_detected and "Red" not in self.detected_colors:
                self.detected_colors.append("Red")
                self.get_logger().info('Red box detected!')
            
            if self.green_detected and "Green" not in self.detected_colors:
                self.detected_colors.append("Green")
                self.get_logger().info('Green box detected!')
            
            if self.blue_detected and "Blue" not in self.detected_colors:
                self.detected_colors.append("Blue")
                self.get_logger().info('Blue box detected!')
            
            # Add live color detection status to the image
            self.add_status_overlay(cv_image)
            
            # Display results
            cv2.imshow('Color Detection', cv_image)
            cv2.waitKey(1)
            
        except Exception as e:
            self.get_logger().error(f'Error processing image: {str(e)}')
    
    def add_status_overlay(self, image):
        # Create a semi-transparent overlay for text
        h, w = image.shape[:2]
        overlay = image.copy()
        
        # Draw background rectangle for text
        cv2.rectangle(overlay, (10, h - 110), (250, h - 10), (0, 0, 0), -1)
        
        # Add transparency
        alpha = 0.7
        cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
        
        # Add status text
        red_status = "RED: DETECTED" if self.red_detected else "RED: Not Detected"
        green_status = "GREEN: DETECTED" if self.green_detected else "GREEN: Not Detected"
        blue_status = "BLUE: DETECTED" if self.blue_detected else "BLUE: Not Detected"
        
        # Show text with color coding
        cv2.putText(image, red_status, (20, h - 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                    (0, 0, 255) if self.red_detected else (150, 150, 150), 2)
        
        cv2.putText(image, green_status, (20, h - 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                    (0, 255, 0) if self.green_detected else (150, 150, 150), 2)
        
        cv2.putText(image, blue_status, (20, h - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                    (255, 0, 0) if self.blue_detected else (150, 150, 150), 2)
        
        # Add state info
        if self.mission_complete:
            state_text = "BLUE BOX FOUND - MISSION COMPLETE"
        elif self.all_waypoints_visited:
            state_text = "ALL WAYPOINTS VISITED"
        else:
            state_text = f"WAYPOINT: {self.current_waypoint+1}/{len(self.waypoints)}"
        
        cv2.putText(image, state_text, (w - 240, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    def detect_color(self, image, mask, color, label):
        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Find the largest contour
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            
            # Only process if contour is large enough
            if area > 500:
                # Draw a rectangle around the contour
                x, y, w, h = cv2.boundingRect(largest_contour)
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                
                # Add text label
                cv2.putText(image, f'{label}: {area:.0f}', (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                return True
        
        return False
    
    def navigate_to_waypoint(self):
        if self.navigating or self.mission_complete or self.all_waypoints_visited:
            return False
        
        # Get current waypoint coordinates
        x, y, yaw = self.waypoints[self.current_waypoint]
        self.get_logger().info(f'Navigating to waypoint {self.current_waypoint+1}: ({x}, {y}, {yaw})')
        
        # Create navigation goal with higher speed parameters
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.pose.position.x = x
        goal_msg.pose.pose.position.y = y
        
        # Simplify orientation - just use upright orientation (no specific yaw)
        goal_msg.pose.pose.orientation.z = 0.0
        goal_msg.pose.pose.orientation.w = 1.0
        
        # Send goal
        try:
            self.nav_client.wait_for_server(timeout_sec=1.0)
            self.navigating = True
            send_goal_future = self.nav_client.send_goal_async(goal_msg)
            send_goal_future.add_done_callback(self.goal_response_callback)
            
            # Also send a direct velocity command to increase speed
            self.increase_speed()
            
            return True
        except Exception as e:
            self.get_logger().error(f'Navigation server error: {str(e)}')
            return False
    
    def increase_speed(self):
        """Send a direct velocity command to increase speed"""
        cmd = Twist()
        cmd.linear.x = 0.5  # Increased from default 0.26 m/s
        self.vel_pub.publish(cmd)
    
    def goal_response_callback(self, future):
        goal_handle = future.result()
        
        if not goal_handle.accepted:
            self.get_logger().warn('Goal rejected')
            self.navigating = False
            self.current_waypoint = (self.current_waypoint + 1) % len(self.waypoints)
            self.waypoints_visited_count += 1
            return
        
        self.goal_handle = goal_handle
        self.get_logger().info('Goal accepted')
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.goal_result_callback)
    
    def goal_result_callback(self, future):
        self.navigating = False
        
        # After reaching a waypoint, check if we see blue
        if self.blue_detected:
            self.get_logger().info('Blue box detected at waypoint, stopping!')
            self.mission_complete = True
            
            # Send a stop command to ensure the robot stops
            cmd = Twist()
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
            self.vel_pub.publish(cmd)
            return
            
        # Increment the waypoint visited counter
        self.waypoints_visited_count += 1
        
        # Check if we've visited all waypoints
        if self.waypoints_visited_count >= len(self.waypoints):
            self.get_logger().info('All waypoints visited')
            self.all_waypoints_visited = True
            
            # Send a stop command
            cmd = Twist()
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
            self.vel_pub.publish(cmd)
        else:
            # Move to next waypoint
            self.current_waypoint = (self.current_waypoint + 1) % len(self.waypoints)
            self.get_logger().info(f'Moving to next waypoint: {self.current_waypoint+1}')
    
    def stop_robot(self):
        """Send zero velocity command to stop the robot"""
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0
        self.vel_pub.publish(cmd)


def main(args=None):
    # Initialize ROS
    rclpy.init(args=args)
    
    # Create node
    robot = RobotController()
    
    # Set up signal handling for cleanup
    def signal_handler(sig, frame):
        robot.stop_robot()
        cv2.destroyAllWindows()
        rclpy.shutdown()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # Start node in separate thread
    thread = threading.Thread(target=rclpy.spin, args=(robot,), daemon=True)
    thread.start()
    
    # Give time for initialization
    time.sleep(2.0)
    
    # Main control loop
    last_navigation_time = 0
    
    try:
        while rclpy.ok():
            current_time = time.time()
            
            # If mission is complete or all waypoints visited, just continue monitoring
            if robot.mission_complete:
                robot.get_logger().info('Mission complete - blue box found. Staying in position.')
                time.sleep(5.0)  # Print message every 5 seconds
                continue
                
            if robot.all_waypoints_visited:
                robot.get_logger().info('All waypoints visited without finding blue box. Exploration complete.')
                time.sleep(5.0)  # Print message every 5 seconds
                continue
                
            # Navigate to next waypoint if not currently navigating
            if not robot.navigating:
                if current_time - last_navigation_time > 1.0:  # Reduced delay between navigation attempts
                    robot.navigate_to_waypoint()
                    last_navigation_time = current_time
            
            # If navigating, periodically boost speed
            if robot.navigating and int(current_time) % 3 == 0:
                robot.increase_speed()
            
            # Show live color detection in each frame
            current_colors = []
            if robot.red_detected:
                current_colors.append("Red")
            if robot.green_detected:
                current_colors.append("Green")
            if robot.blue_detected:
                current_colors.append("Blue")
                
            if current_colors and int(current_time) % 2 == 0:  # Only log every 2 seconds to reduce spam
                robot.get_logger().info(f'Currently detecting: {", ".join(current_colors)}')
            
            # Small delay
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        pass
    
    # Clean up
    robot.stop_robot()
    cv2.destroyAllWindows()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
