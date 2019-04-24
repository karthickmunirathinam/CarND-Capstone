from styx_msgs.msg import TrafficLight
import cv2
import numpy as np
import rospy


class TLClassifier(object):
    def __init__(self):
        #TODO load classifier
        pass

    def cv_based_red_color_classifier(self,image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        red1_lower = np.array([0, 100, 100])
        red1_upper = np.array([10, 255,255])
        red2_lower = np.array([160,100,100])
        red2_upper = np.array([179,255,255])

        mask1 = cv2.inRange(hsv, red1_lower, red1_upper)
        mask2 = cv2.inRange(hsv, red2_lower, red2_upper)
        red_img = cv2.addWeighted(mask1,1.0,mask2,1.0,0)

        im, contours, hierarchy = cv2.findContours(red_img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        red_count = 0
        for x,contour in enumerate(contours):
            contourarea = cv2.contourArea(contour) #get area of contour
            if 15 < contourarea < 700: #Discard contours with a too large area as this may just be noise
                arclength = cv2.arcLength(contour, True)
                approxcontour = cv2.approxPolyDP(contour, 0.01 * arclength, True)
                #Find the coordinates of the polygon with respect to he camera frame in pixels
                rect_cordi = cv2.minAreaRect(contour)
                obj_x = int(rect_cordi[0][0])
                obj_y = int(rect_cordi[0][1])
                #Check for Square
                if len(approxcontour)>5:
                    red_count += 1

        # rospy.loginfo("red :  %d",red_count)
        if red_count > 0:
            return TrafficLight.RED

        return TrafficLight.UNKNOWN

    def carla_real_data_classifier(self,image):
        return TrafficLight.UNKNOWN

    def dl_based_classifier(self,image):
        return TrafficLight.UNKNOWN

    def get_classification(self, image, method):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction
        if(method == "opencv"):
            return self.cv_based_red_color_classifier(image)
        elif(method == "carla"):
            return self.carla_real_data_classifier(image)

        return self.dl_based_classifier(image)

