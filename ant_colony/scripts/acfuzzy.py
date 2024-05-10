#!/usr/bin/env python3

import rospy
from std_msgs.msg import Float64
import skfuzzy as fuzz
import numpy as np
from time import sleep
import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal

class FuzzyTSPController:
    def __init__(self):
        rospy.init_node('fuzzy_tsp_controller')
        self.rate = rospy.Rate(10)  # 10 Hz
        self.pub_best_solution = rospy.Publisher('/best_solution', Float64, queue_size=10)
        
        # Input variables
        self.pheromone_weight = None
        self.number_of_ants = None
        self.iterations = None
        self.destinations = []

        # Initialize subscriber for input variables
        rospy.Subscriber('/pheromone_weight', Float64, self.pheromone_weight_callback)
        rospy.Subscriber('/number_of_ants', Float64, self.number_of_ants_callback)
        rospy.Subscriber('/iterations', Float64, self.iterations_callback)

        # Input destination points
        self.input_destinations()

        # Fuzzy membership functions
        self.pheromone_weight_mfs = {
            'Low_Density': [0, 1, 2],
            'Medium_Density': [1.5, 2.5, 3.5],
            'High_Density': [3, 4, 5]
        }

        self.number_of_ants_mfs = {
            'Small': [0, 3, 6],
            'Medium': [4, 7, 10],
            'High': [8, 10, 10]
        }

        self.iterations_mfs = {
            'Small': [0, 30, 60],
            'Medium': [40, 70, 100],
            'Large': [80, 110, 145]
        }

    def pheromone_weight_callback(self, msg):
        self.pheromone_weight = msg.data

    def number_of_ants_callback(self, msg):
        self.number_of_ants = msg.data

    def iterations_callback(self, msg):
        self.iterations = msg.data

    def input_destinations(self):
        tp_number = int(input("Enter the number of points to go? :"))
        
        print("Enter destination points to go as x and y respectively.")
        for i in range(tp_number):
            if i < 1:
                print("****First enter the robot's starting location on the map?****")
            else:
                print("\n***********\n {}. Point: Enter Destination\n***********".format(i))
            x_co = float(input("{}. Target Point = x[{}][0] = ".format(i, i)))
            y_co = float(input("{}. Target Point = y[{}][1] = ".format(i, i)))
            self.destinations.append([x_co, y_co])

    def calculate_best_solution(self):
        # Fuzzification
        pheromone_membership = self.fuzzify(self.pheromone_weight, self.pheromone_weight_mfs)
        ants_membership = self.fuzzify(self.number_of_ants, self.number_of_ants_mfs)
        iterations_membership = self.fuzzify(self.iterations, self.iterations_mfs)

        # Rule Evaluation
        rule_outputs = self.evaluate_rules(pheromone_membership, ants_membership, iterations_membership)

        # Aggregation
        aggregated = self.aggregate_outputs(rule_outputs)

        # Defuzzification
        best_solution = fuzz.defuzz(np.array([0, 1]), aggregated, 'centroid')

        self.pub_best_solution.publish(best_solution)

    def fuzzify(self, value, mfs):
        membership = {}
        for key, mf_range in mfs.items():
            membership[key] = fuzz.interp_membership(np.array(mf_range), fuzz.trimf(np.array(mf_range), mf_range), value)
        return membership

    def evaluate_rules(self, pheromone_membership, ants_membership, iterations_membership):
        rule_outputs = np.zeros(len(pheromone_membership['Low_Density']))
        # Assuming rules remain the same
        rules = [
            (3, 3, 3), (3, 2, 3), (3, 1, 3), (3, 3, 2), (3, 2, 2), (3, 1, 2), (3, 3, 1), (3, 2, 1), (3, 1, 1),
            (1, 3, 3), (1, 3, 2), (1, 3, 1), (1, 2, 3), (1, 2, 2), (1, 2, 1), (1, 1, 3), (1, 1, 2), (1, 1, 1),
            (2, 3, 3), (2, 3, 2), (2, 3, 1), (2, 2, 3), (2, 2, 2), (2, 2, 1), (2, 1, 3), (2, 1, 2), (2, 1, 1)
        ]
        for rule in rules:
            antecedent = np.min([pheromone_membership['Low_Density'][rule[0]-1], ants_membership['Small'][rule[1]-1], iterations_membership['Small'][rule[2]-1]])
            rule_outputs[rule[0]-1] = max(rule_outputs[rule[0]-1], antecedent)
        return rule_outputs

    def aggregate_outputs(self, rule_outputs):
        return np.max(rule_outputs)

    def move_to_destination(self, x, y):
        rospy.loginfo("Moving to destination: x={}, y={}".format(x, y))
        # Buat dan kirim tujuan untuk gerakan robot
        client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        client.wait_for_server()
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = "map"
        goal.target_pose.pose.position.x = x
        goal.target_pose.pose.position.y = y
        goal.target_pose.pose.orientation.w = 1.0
        client.send_goal(goal)
        # Tunggu hingga tugas selesai
        client.wait_for_result()
        rospy.loginfo("Reached destination: x={}, y={}".format(x, y))

    def run(self):
        for i, destination in enumerate(self.destinations):
            x, y = destination
            if i == 0:
                rospy.loginfo("Starting from initial position: x={}, y={}".format(x, y))
            else:
                self.move_to_destination(x, y)
        rospy.loginfo("All destinations reached. Task completed.")

if __name__ == '__main__':
    controller = FuzzyTSPController()
    try:
        controller.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("ROS node has been shut down.")