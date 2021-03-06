"""
Date: 13/04/2020
Last Modification: 18/06/2020
Author: Maxime Cauté (maxime.caute@epfl.ch)
---
The functions in this file are used to define policies to be learnt
and handle their assignement tables.
---
Upon being executed, this file generates assignement tables
"""
import numpy as np
import data_saver
import data_generator

def nearest_neighbour_policy(situation):
    """
    This function defines the nearest neighbour policy.
    That is, the selected vehicle is the closest one to the request.
    ---
    Input:
        - situation: int list list.
            A representation of the situation.
                The first element is the request information.
                    Its two first values are the pickup location (x,y) indices.
                    Its two following values are the dropoff location indices.
                The following elements are the vehicles information.
                    Their two fist values are their current position indices.
    Output:
        - best_location: int*int.
            The indices of the position of the vehicle closest to the request.
    """
    request = situation[0]
    pickup = request[:2]
    vehicles = situation[1:]

    # Setting minimal distance as an absurdly high value at beginning.
    min_distance = 10000
    best_location = (0,0)
    for vehicle in vehicles:
        vehicle_position = vehicle[:2]
        distance = data_generator.distance(vehicle_position, pickup)

        if distance < min_distance:
            min_distance = distance
            best_location = tuple(vehicle_position)
    return best_location

def top_corner_policy(situation):
    """
    This function defines the top corner policy.
    That is, the selected vehicle is the closest one to the top left corner.
    ---
    Input:
        - situation: int list list.
            A representation of the situation.
                The first element is the request information.
                    Its two first values are the pickup location (x,y) indices.
                    Its two following values are the dropoff location indices.
                The following elements are the vehicles information.
                    Their two fist values are their current position indices.
    Output:
        - best_location: int*int.
            The indices of the position of the vehicle closest to
            the top left corner.
    """
    top_left_corner=(0,0)

    vehicles = situation[1:]

    min_distance = 10000
    best_location = (0,0)
    for vehicle in vehicles:
        vehicle_position = vehicle[:2]
        distance = data_generator.distance(vehicle_position, top_left_corner)

        if distance < min_distance:
            min_distance = distance
            best_location = tuple(vehicle_position)

    return best_location


### Utils ###

def assignements_from_table(table, policy):
    """
    This function creates an assignement table from a situation_table.
    ---
    Input:
        - table: int list list list.
            The situation table from which the policy assignements are generated.
                The elements of this list are situations (see l.18).
        - policy: int list list -> int*int.
            The policy to create assignements from.
    Output:
        - assignements_table: int list list.
            The assignement table to be generated.
                It is a position (x,y) list,
                shaped as a list list for convenience.
    """
    assignements_table = [policy(situation) for situation in table]
    return assignements_table

def generate_assignements_from_indices(indices, policy, load_path = ""):
    """
    This function generates assignement table files
    from situation table files.
    ---
    Input:
        - indices: int list.
            The list of indices of the tables to process.
        - policy: int list list -> int*int.
            The policy to create assignements from.
    Parameters:
        # TODO Rename ?
        - load_path: string.
            The path where tables are located.
                Defaults to local folder.
    Side effects:
        - Generates an assignement file in folder located by load_path.
    """
    for i in indices:
        situations_table = data_saver.load_table(i, folder = "situations_tables", relative_path = load_path)
        assignements_table = assignements_from_table(situations_table,policy)
        data_saver.save_table(assignements_table, i, folder = "assignements_tables", relative_path = load_path)

def test_policy_tables(table, assignements_table, policy):
    """
    This function evaluates policy table.
    Its main purpose is to act as a sanity check.
    ---
    Input:
        - table: int list list list.
            The situation table from which the policy assignements
            are generated.
                The elements of this list are situations (see l.18).
        - assignements_table: int list list.
            The assignement table to be evaluated.
                It is a position (x,y) list.
        - policy: int list list -> int*int.
            The policy to create assignements from.
    Side effects:
        - Prints out wrong assignements according to policy.
    """
    for i, situation in enumerate(table):
        policy_choice = tuple(policy(situation))
        assignement = tuple(assignements_table[i])

        if policy_choice != assignement:
            print("Wrong choice!", situation, assignement, policy_choice)

if __name__ == "__main__":
    """
    Upon being executed, these file generates assignements from given indices.
    ---
    Several arguments can be added, see 'generate_assignements_from_indices'
    description for more precision.

    Call Example:
        python3 policies.py 10 -p "TestTrashDir/"
    """
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("tables", type=int,
                        help="Number of tables to be generated.")
    parser.add_argument("-pol", "--policy", type=str,
                        default="NN",
                        choices =["NN","TC"],
                        help="The policy to use for assignements."+
                            "Possibles choices are "+
                            "Nearest Neighbour (NN), and "+
                            "Top Corner (TC) policies."+
                            "Defaults to NN.")
    parser.add_argument("-p", "--path", type=str,
                        default="",
                        help="The path where to save files. Defaults to local.")
    args = parser.parse_args()

    policy = (  nearest_neighbour_policy if args.policy == "NN" else
                top_corner_policy )# if args.policy == "TC"

    generate_assignements_from_indices(
            range(args.tables), policy, load_path=args.path
                )
