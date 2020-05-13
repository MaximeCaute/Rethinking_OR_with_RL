"""
Date: 13/04/2020
Last Modification: 13/04/2020
Author: Maxime Cauté (maxime.caute@epfl.ch)
---
The functions in this file are used to generate the data for learning.
"""

import numpy as np
import numpy.random as rdm
import pandas as pd

import data_saver

def distance(source,target):
    x, y = source[0], source[1]
    target_x, target_y = target[0], target[1]
    dist = np.sqrt( (x-target_x)**2 + (y-target_y)**2 )
    return dist

def random_pixel(image_size, forbidden_pixels = []):
    """
    This function selects a random pixel. By default, this is uniformly chosen, without any pixel restriction.
    ---
    Input:
        - image_size: int. The size of the image, in pixels.
    Parameters:
        - forbidden_pixels: int couple list. The list of the forbidden pixels, by location. By default, every pixel is allowed.
    Output:
        - pixel: int couple. The chosen pixel, as its coordinates.
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
        - image_size: int. The size of the image, in pixels.
    Parameters:
        - forbidden_pixels: int couple list. The list of the forbidden pixels, by location. By default, every pixel is allowed.
    Output:
        - pickup_and_dropoff: int quadruple. The pickup and dropoff, as two concatenated locations.
    """
    pickup = random_pixel(image_size, forbidden_pixels)
    dropoff = random_pixel(image_size, forbidden_pixels)
    pickup_and_dropoff = np.concatenate([pickup, dropoff])
    return pickup_and_dropoff

def generate_vehicle(image_size, forbidden_pixels = []):
    """
    This functions generates informations on a vehicle.  By default, only its position.
    ---
    Input:
        - image_size: int. The size of the image, in pixels.
    Parameters:
        - forbidden_pixels: int couple list. The list of the forbidden pixels, by location. By default, every pixel is allowed.
    Output:
        - vehicle: int list. The vehicle informations.
    """
    vehicle_position = random_pixel(image_size, forbidden_pixels)
    vehicle = np.array(vehicle_position)
    return vehicle

def generate_vehicles(fleet_size, image_size, overlaps = True, multiple_nearest = True, reference_for_nearest = (0,0)):
    """
    This functions generates informations on all the vehicles.  By default, only its position, with allowed overlapsing.
    ---
    Input:
        - fleet_size: int. The size of the fleet, i.e. the number of vehicles.
        - image_size: int. The size of the image, in pixels.
    Parameters:
        - overlaps: boolean. Whether or not vehicle overlaps is allowed. By default, overlaps is allowed.
        TODO
    Output:
        - vehicles.
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
    TODO
    This functions generates table with a given amount of requests, i.e couple of locations (respectively pickup and dropoff), in the shape  of a quadruple.
    ---
    Input:
        - requests_amount: int. The number of requests that will be in the table
    Parameters:
        - image_size: int. The size of the image, in pixels.
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
    tables = []
    for n in range(tables_amount):
        table = generate_situations_table(requests_amount = requests_amount,
                                          image_size = image_size, fleet_size = fleet_size,
                                          vehicle_overlaps = vehicle_overlaps,
                                          multiple_nearest = multiple_nearest, reference_for_nearest = reference_for_nearest)

        if save:
            data_saver.save_table(table, 2000+n, folder = 'situations_tables', relative_path = save_path)
        else:
            tables.append(table)
    return tables

def dist(s,t):
    return np.sqrt( (s[0] -t[0])**2 + (s[1]-t[1])**2 )


def test(table, a_table, multiple_nns = False, fleet_size = 30):
    errs = 0
    tot = 0
    for i, r in enumerate(table):
        err= False
        tot+=1
        assi = a_table[i]
        opti_dist = dist(r[0], assi )
        opti_pos = [vhc for vhc in r[1:] if dist(vhc, r[0]) == opti_dist]
        if len(opti_pos) != 1 and not multiple_nns:
            print("Too opti!", opti_pos)
            err = True
        if len(r) != fleet_size+1:
            print("Too much vehicles!", len(r))
            err = True

        if err:
            errs+=1
    print("Error rate:", float(errs)/tot)



if __name__ == "__main__":
    generate_tables(2000, requests_amount = 500, image_size = 30, fleet_size = 10,
                    vehicle_overlaps = False,
                    multiple_nearest = False, reference_for_nearest = (-1,-1),
                    save = True, save_path = "")
