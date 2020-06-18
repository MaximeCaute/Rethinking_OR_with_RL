"""
Date: 13/04/2020
Last Modification: 18/06/2020
Author: Maxime Cauté (maxime.caute@epfl.ch)
---
The functions in this file are used to generate the data for learning.
Upon being executed, it generates a set of data_tables.
"""

import numpy as np
import numpy.random as rdm
import pandas as pd

import data_saver

def distance(source,target):
    """
    This function computes the euclidian distance
    between two sets of coordinates.
    ---
    Input:
        - source : int*int.
            The source coordinates in the (x,y) plan.
        - target : int*int.
            The target coordinates in the (x,y) plan.
    """
    x, y = source[0], source[1]
    target_x, target_y = target[0], target[1]
    dist = np.sqrt( (x-target_x)**2 + (y-target_y)**2 )
    return dist

def random_pixel(image_size, forbidden_pixels = []):
    """
    This function selects a random pixel.
    By default, this is uniformly chosen, without any pixel restriction.
    ---
    Input:
        - image_size: int.
            The size of the image, in pixels.
    Parameters:
        - forbidden_pixels: int couple list.
            The list of the forbidden pixels, by location.
                Defaults to no pixel.
    Output:
        - pixel: int couple.
            The chosen pixel, as its coordinates.
    """
    pixel = (rdm.randint(image_size), rdm.randint(image_size))
    while pixel in forbidden_pixels:
        pixel = (rdm.randint(image_size), rdm.randint(image_size))
    return pixel


def generate_pickup_and_dropoff(image_size, forbidden_pixels = []):
    """
    This functions generate a pickup and dropoff location for requests.
    ---
    Input:
        - image_size: int.
            The size of the image, in pixels.
    Parameters:
        - forbidden_pixels: int couple list.
            The list of the forbidden pixels, by location.
                Defaults to no pixel.
    Output:
        - pickup_and_dropoff: int quadruple.
            The pickup and dropoff, as two concatenated locations.
    """
    pickup = random_pixel(image_size, forbidden_pixels)
    dropoff = random_pixel(image_size, forbidden_pixels)
    pickup_and_dropoff = np.concatenate([pickup, dropoff])
    return pickup_and_dropoff

def generate_vehicle(image_size, forbidden_pixels = []):
    """
    This functions generates information on a vehicle.
    By default, only its position is generated.
    ---
    Input:
        - image_size: int. The size of the image, in pixels.
    Parameters:
        - forbidden_pixels: int couple list.
            The list of the forbidden pixels, by location.
                Defaults to no pixel.
    Output:
        - vehicle: int list.
            The vehicle information.
    """
    vehicle_position = random_pixel(image_size, forbidden_pixels)
    vehicle = np.array(vehicle_position)
    return vehicle

def generate_vehicles(fleet_size, image_size, overlaps = True, multiple_nearest = True, reference_for_nearest = (0,0)):
    """
    This functions generates informations on all the vehicles.
    By default, only its position, with allowed overlapsing.
    ---
    Input:
        - fleet_size: int.
            The size of the fleet, i.e. the number of vehicles.
        - image_size: int.
            The size of the image, in pixels.
    Parameters:
        - overlaps: boolean.
            Toggles vehicle overlaps.
                Defaults to allowed.
        - multiple_nearest: boolean.
            Toggles multiple nearest vehicles to the reference.
                Used to generated Nearest Neighbor tables
                    with one single solution.
                Defaults to allowed.
        - reference_for_nearest: int*int.
            If multiple nearest is not allowed, the reference for
            nearest vehicle computation.
                Defaults to top left corner.
    Output:
        - vehicles. int list list.
            The list of the vehicles, containing information for each one.
    """
    vehicles = []
    vehicles_positions = []
    nearest_distance = image_size**2

    for n in range(fleet_size):
        #Checking overlaps
        if not overlaps:
            forbidden_pixels = vehicles_positions
        else:
            forbidden_pixels = []

        vehicle = generate_vehicle(image_size, forbidden_pixels = forbidden_pixels)
        vehicle_position = tuple(vehicle[0:2])
        dist = distance(vehicle_position, reference_for_nearest)

        while ( not multiple_nearest and dist == nearest_distance  ):
                vehicle = generate_vehicle(image_size, forbidden_pixels = forbidden_pixels)
                vehicle_position = tuple(vehicle[0:2])
                dist = distance(vehicle_position, reference_for_nearest)


        if dist < nearest_distance:
            nearest_distance = dist

        vehicles.append(vehicle)
        vehicles_positions.append(vehicle_position)

    return vehicles

def generate_situation(image_size = 30, fleet_size = 30,
                       vehicle_overlaps = True,
                       multiple_nearest = True, reference_for_nearest = (-1,-1)):
    """
    This function generates a DaRP situation, i.e. a pickup and delivery request,
    and a fleet of agents.
    ---
    Parameters:
        - image_size: int.
            The size of the image, in pixels.
                Defaults to 30.
        - fleet_size: int.
            The size of the fleet, i.e. the number of vehicles.
                Defaults to 30.
        - vehicle_overlaps: boolean.
            Toggles vehicle overlaps.
                Defaults to allowed.
        - multiple_nearest: boolean.
            Toggles multiple nearest vehicles to the reference.
                Used to generated Nearest Neighbor tables
                    with one single solution.
                Defaults to allowed.
        - reference_for_nearest: int*int.
            If multiple nearest is not allowed, the reference for
            nearest vehicle computation.
                Defaults to pickup location.
    Output:
        - situation: int list list.
            The information about the situation, as a list.
                The first element is the request information.
                The following elements are the vehicles information.
    """
    request = generate_pickup_and_dropoff(image_size)
    pickup = tuple(request[0:2])

    if reference_for_nearest == (-1,-1):
        reference_for_nearest = pickup

    vehicles = generate_vehicles(fleet_size, image_size,
                                 overlaps = vehicle_overlaps,
                                 multiple_nearest = multiple_nearest,
                                 reference_for_nearest = reference_for_nearest)


    situation = [request] + vehicles
    return situation

def generate_situations_table(requests_amount = 100, image_size = 30, fleet_size = 30,
                              vehicle_overlaps = True,
                              multiple_nearest = True, reference_for_nearest = (-1,-1)):
    """
    This functions generates a table with a given amount of situations.
    ---
    Parameters:
        - requests_amount: int.
            The number of requests that will be in the table
                Defaults to 100.
        - image_size: int.
            The size of the image, in pixels.
                Defaults to 30.
        - fleet_size: int.
            The size of the fleet, i.e. the number of vehicles.
                Defaults to 30.
        - vehicle_overlaps: boolean.
            Toggles vehicle overlaps.
                Defaults to allowed.
        - multiple_nearest: boolean.
            Toggles multiple nearest vehicles to the reference.
                Used to generated Nearest Neighbor tables
                    with one single solution.
                Defaults to allowed.
        - reference_for_nearest: int*int.
            If multiple nearest is not allowed, the reference for
            nearest vehicle computation.
                Defaults to pickup location.
    Output:
        - table: int list list list.
            A table of situations in the shape of a list.
                Each element is the information about the situation,
                as a list in which:
                    the first element is the request information;
                    the following elements are the vehicles information.
    """
    table = []
    for n in range(requests_amount):
        situation = generate_situation(image_size = image_size, fleet_size = fleet_size,
                                       vehicle_overlaps = vehicle_overlaps,
                                       multiple_nearest = multiple_nearest, reference_for_nearest = reference_for_nearest)

        table.append(situation)
    return table


def generate_tables(tables_amount, requests_amount = 100, image_size = 30, fleet_size = 30,
                    vehicle_overlaps = True,
                    multiple_nearest = True, reference_for_nearest = (-1,-1),
                    save = False, save_path = ""):
    """
    This function generates a set of situation tables that can be saved.
    ---
    Input:
        - tables_amount: int.
            The amount of tables to generate.
    Parameters:
        - requests_amount: int.
            The amount of requests to be generated in a table.
                Defaults to 100.
        - image_size : int.
            The size of the image, in pixels.
                Defaults to 30.
        - fleet_size: int.
            The size of the fleet, i.e. the number of vehicles.
                Defaults to 30.
        - vehicle_overlaps: boolean.
            Toggles vehicle overlaps.
                Defaults to allowed.
        - multiple_nearest: boolean.
            Toggles multiple nearest vehicles to the reference.
                Used to generated Nearest Neighbor tables
                    with one single solution.
                Defaults to allowed.
        - reference_for_nearest: int*int.
            If multiple nearest is not allowed, the reference for
            nearest vehicle computation.
                Defaults to pickup location.
        - save: boolean.
            Toggles table saving.
                Defaults to activated.
        - save_path: string.
            The path where the data is to be saved.
                Defaults to local folder.
    Output:
        - tables: table list.
            The list of the tables.
                See generate_situations_table (function above) for more info.
    """
    tables = []
    for n in range(tables_amount):
        table = generate_situations_table(requests_amount = requests_amount,
                                          image_size = image_size, fleet_size = fleet_size,
                                          vehicle_overlaps = vehicle_overlaps,
                                          multiple_nearest = multiple_nearest, reference_for_nearest = reference_for_nearest)

        if save:
            data_saver.save_table(table, n, folder = 'situations_tables', relative_path = save_path)
        else:
            tables.append(table)
    return tables

# Doubled
def dist(s,t):
    return np.sqrt( (s[0] -t[0])**2 + (s[1]-t[1])**2 )


def test(table, a_table, multiple_nns = False, fleet_size = 30):
    """
    This function executes a sanity check on a table.
    ---
    Input:
        - table: int list list list.
            The situation table to be evaluated.
                Each element is the information about the situation,
                as a list in which:
                    the first element is the request information;
                    the following elements are the vehicles information.
        - a_table: int list list.
            The assignement table, to evaluate multiple nearest neighbours.
    Parameters:
        - multiple_nns: boolean.
            Toggles multiple nearest neighbours to the pickup request.
                Defaults to allowed.
        - fleet_size: int.
            The size of the leet to be compared with agents in the tables.
                Defaults to 30
    Side effect:
        - Prints the amount of wrong tables.
    """
    errs = 0
    tot = 0
    for i, r in enumerate(table):
        err= False
        tot+=1
        assi = a_table[i]
        opti_dist = dist(r[0], assi )
        opti_pos = [vhc for vhc in r[1:] if dist(vhc, r[0]) == opti_dist]
        if len(opti_pos) != 1 and not multiple_nns:
            print("Too many optimal solutions!", opti_pos)
            err = True
        if len(r) != fleet_size+1:
            print("Too many vehicles!", len(r))
            err = True

        if err:
            errs+=1
    print("Error rate:", float(errs)/tot)



if __name__ == "__main__":
    """
    Upon being executed, generates a set of tables.
    Several arguments can be added, see 'generate_tables' description
    for more precision.

    Call example:
        python3 data_generator.py 5 -p "TestTableDir/"
    """
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("tables", type=int,
                        help="Number of tables to be generated.")
    ### optional arguments
    parser.add_argument("-r", "--requests_amount", type=int, default=500,
                        help="Number of requests per table.")
    parser.add_argument("-i", "--image_size", type=int, default=30,
                        help="Size of the images to be generated, in pixels.")
    parser.add_argument("-f", "--fleet_size", type=int, default=10,
                        help="Size of the fleet to be generated.")
    parser.add_argument("-vo", "--vehicles_overlaps", action="store_true",
                        default=False,
                        help="Allows vehicle overlap.")
    parser.add_argument("-mn", "--multiple_nearest", action="store_true",
                        default=False,
                        help="Allows multiple nearest neighbours to reference.")
    parser.add_argument("-ref", "--reference_for_nearest", type=int, nargs=2,
                        default = [-1, -1],
                        help="Sets reference position for nearest neighbour.")
    parser.add_argument("-p", "--path", type = str, default="",
                        help="The path where to save files. Defaults to local.")
    args = parser.parse_args()

    generate_tables(args.tables,
                    requests_amount = args.requests_amount,
                    image_size = args.image_size, fleet_size = args.fleet_size,
                    vehicle_overlaps = args.vehicles_overlaps,
                    multiple_nearest = args.multiple_nearest,
                    reference_for_nearest = args.reference_for_nearest,
                    save = True, save_path = args.path)#"NN_tables30/"
