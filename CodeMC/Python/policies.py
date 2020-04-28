"""
Date: 13/04/2020
Last Modification: 13/04/2020
Author: Maxime Cauté (maxime.caute@epfl.ch)
---
The functions in this file are used to define policies to be learnt.
"""
import numpy as np
import data_saver

def nearest_neighbour_policy(situation):
    request = situation[0]

    x_pickup, y_pickup = request[0], request[1]

    vehicles = situation[1:]

    min_distance = 10000
    best_location = (0,0)
    for vehicle in vehicles:
        x_vehicle, y_vehicle = vehicle[0], vehicle[1]

        distance = np.sqrt( (x_vehicle - x_pickup)**2 + (y_vehicle - y_pickup)**2 )

        if distance < min_distance:
            min_distance = distance
            best_location = (x_vehicle, y_vehicle)

    return best_location

def top_corner_policy(situation):
    x_corner, y_corner = 0,0

    vehicles = situation[1:]

    min_distance = 10000
    best_location = (0,0)
    for vehicle in vehicles:
        x_vehicle, y_vehicle = vehicle[0], vehicle[1]

        distance = np.sqrt( (x_vehicle - x_corner)**2 + (y_vehicle - y_corner)**2 )

        if distance < min_distance:
            min_distance = distance
            best_location = (x_vehicle, y_vehicle)

    return best_location


### Utils ###

def assignements_from_table(table, policy):
    assignements_table = [policy(situation) for situation in table]
    return assignements_table

def generate_assignements_from_indices(indices, policy):
    for i in indices:
        situations_table = data_saver.load_table(i, folder = "situations_tables")
        assignements_table = assignements_from_table(situations_table,policy)
        data_saver.save_table(assignements_table, i, folder = "assignements_tables")

def test_policy_tables(table, assignements_table, policy):
    for i, situation in enumerate(table):
        policy_choice = policy(situation)
        assignement = assignements_table[i]

        if policy_choice != assignement:
            print("Wrong choice!", situation, assignement, policy_choice)

if __name__ == "__main__":
    generate_assignements_from_indices(range(2000), top_corner_policy)
