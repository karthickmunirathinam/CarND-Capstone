#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Int32
from styx_msgs.msg import Lane, Waypoint
from scipy.spatial import KDTree
import numpy as np

import math

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 100 # Number of waypoints we will publish. You can change this number
MAX_DECEL = 1.0 # TODO update to higher value
FREQUENCY = 10
STOP_EARLY_POINTS = 4


class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)
        #rospy.Subscriber('/obstacle_waypoint', Waypoint, self.obstacle_cb)

        self.final_waypoints_pub = rospy.Publisher('/final_waypoints', Lane, queue_size=1)

        # member variables
        self.pose = None
        self.base_waypoints = None
        self.waypoints_2d = None
        self.waypoint_tree = None
        self.stopline_wp_idx = -1
        rospy.loginfo("Initialization waypoint node")
        self.loop()

    def loop(self):
        rate = rospy.Rate(FREQUENCY)
        while not rospy.is_shutdown():
            if not (None in (self.pose, self.waypoint_tree)):
                self.publish_waypoints()
            rate.sleep()
            # rospy.loginfo("Waypoint updater loop")

    def get_closest_waypoint_idx(self):
        x = self.pose.pose.position.x
        y = self.pose.pose.position.y
        closest_idx = self.waypoint_tree.query([x,y],1)[1]

        #check if the closest is ahead or behind ego
        closest_coord = self.waypoints_2d[closest_idx]
        prev_coord = self.waypoints_2d[closest_idx-1]

        #equation for hyperplane through closest coordi
        cl_vect = np.array(closest_coord)
        prev_vect = np.array(prev_coord)
        pos_vect = np.array([x,y])
        #determines if the both vectors are in same direction or in different direction
        val = np.dot(cl_vect-prev_vect, pos_vect-cl_vect)
        if val>0:
            closest_idx = (closest_idx+1)%len(self.waypoints_2d)

        return closest_idx

    def publish_waypoints(self):
        final_lane = self.generate_lane()
        self.final_waypoints_pub.publish(final_lane)

    def generate_lane(self):
        lane = Lane()
        closest_idx = self.get_closest_waypoint_idx()
        farthest_idx = closest_idx + LOOKAHEAD_WPS
        sliced_base_waypoints = self.base_waypoints.waypoints[closest_idx:farthest_idx]

        if (self.stopline_wp_idx == -1) or (self.stopline_wp_idx >= farthest_idx):
            lane.waypoints = sliced_base_waypoints
        else:
            lane.waypoints = self.decelerate_waypoints(sliced_base_waypoints,closest_idx)

        return lane

    def decelerate_waypoints(self,waypoints, closest_idx):
        temp = []
        #stop few points early such that whole car stays behind the stop line
        stop_idx = max(self.stopline_wp_idx - closest_idx - STOP_EARLY_POINTS, 0)
        for i,wp in enumerate(waypoints):
            p= Waypoint()
            p.pose = wp.pose
            dist = self.distance(waypoints,i,stop_idx)
            # TODO check a way to reduce velocity even smoothly
            vel = math.sqrt(2*MAX_DECEL*dist)
            if vel < 1.0:
                vel = 0

            #limit speed to the
            p.twist.twist.linear.x = min(vel,wp.twist.twist.linear.x)
            temp.append(p)

        return temp


    def pose_cb(self, msg):
        self.pose = msg


    def waypoints_cb(self, waypoints):
        self.base_waypoints = waypoints
        if self.waypoints_2d is None:
            self.waypoints_2d = [[waypoint.pose.pose.position.x,waypoint.pose.pose.position.y] for waypoint in waypoints.waypoints]
            self.waypoint_tree = KDTree(self.waypoints_2d)
            print ("waypoints initialized")

    def traffic_cb(self, msg):
        # TODO: Callback for /traffic_waypoint message. Implement
        self.stopline_wp_idx = msg.data

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
