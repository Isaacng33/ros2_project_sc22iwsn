# Color detection for RGB boxes

import threading
import sys, time
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from rclpy.exceptions import ROSInterruptException
import signal


class colourIdentifier(Node):
    def __init__(self):
        super().__init__('cI')
        
        # Initialize CvBridge
        self.bridge = CvBridge()
        
        # Color detection sensitivity
        self.sensitivity = 15  # Slightly increased for simulation
        
        # Subscribe to camera feed
        self.image_sub = self.create_subscription(
            Image, 
            '/camera/image_raw', 
            self.image_callback, 
            10)
        
        # Create windows for displaying images
        cv2.namedWindow('Camera Feed', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Color Detection', cv2.WINDOW_NORMAL)
        
        self.get_logger().info('Color detector initialized')
    
    def image_callback(self, data):
        try:
            # Convert ROS Image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(data, 'bgr8')
            
            # Create a copy for drawing detections
            detection_image = cv_image.copy()
            
            # Convert to HSV (better for color filtering)
            hsv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
            
            # Define color ranges in HSV
            # Red color (note: red wraps around in HSV, might need two ranges)
            lower_red1 = np.array([0, 100, 100])
            upper_red1 = np.array([10, 255, 255])
            lower_red2 = np.array([160, 100, 100])
            upper_red2 = np.array([179, 255, 255])
            
            # Green color
            lower_green = np.array([40, 100, 100])
            upper_green = np.array([80, 255, 255])
            
            # Blue color
            lower_blue = np.array([100, 100, 100])
            upper_blue = np.array([140, 255, 255])
            
            # Create masks for each color
            red_mask1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
            red_mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
            red_mask = cv2.bitwise_or(red_mask1, red_mask2)
            
            green_mask = cv2.inRange(hsv_image, lower_green, upper_green)
            blue_mask = cv2.inRange(hsv_image, lower_blue, upper_blue)
            
            # Combine all color masks
            combined_mask = cv2.bitwise_or(red_mask, green_mask)
            combined_mask = cv2.bitwise_or(combined_mask, blue_mask)
            
            # Apply mask to original image
            color_detected = cv2.bitwise_and(cv_image, cv_image, mask=combined_mask)
            
            # Find contours for each color and draw bounding boxes
            self.find_and_draw_contours(detection_image, red_mask, (0, 0, 255), "Red")
            self.find_and_draw_contours(detection_image, green_mask, (0, 255, 0), "Green")
            self.find_and_draw_contours(detection_image, blue_mask, (255, 0, 0), "Blue")
            
            # Display results
            cv2.imshow('Camera Feed', cv_image)
            cv2.imshow('Color Detection', detection_image)
            cv2.waitKey(1)
            
        except Exception as e:
            self.get_logger().error(f'Error processing image: {str(e)}')
    
    def find_and_draw_contours(self, image, mask, color, label):
        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Find the largest contour
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            
            # Only process if contour is large enough (filter noise)
            if area > 500:
                # Draw a rectangle around the contour
                x, y, w, h = cv2.boundingRect(largest_contour)
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                
                # Add text label with area
                cv2.putText(image, f'{label}: {area:.0f}', (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                self.get_logger().info(f'Detected {label} box, area: {area:.0f}')


def main(args=None):
    # Initialize ROS
    rclpy.init(args=args)
    
    # Create node
    color_detector = colourIdentifier()
    
    # Handle keyboard interrupt
    def signal_handler(sig, frame):
        color_detector.get_logger().info('Shutting down...')
        cv2.destroyAllWindows()
        rclpy.shutdown()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # Start node in separate thread
    thread = threading.Thread(target=rclpy.spin, args=(color_detector,), daemon=True)
    thread.start()
    
    try:
        # Keep main thread alive
        while rclpy.ok():
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    
    # Clean up
    cv2.destroyAllWindows()
    rclpy.shutdown()


if __name__ == '__main__':
    main()