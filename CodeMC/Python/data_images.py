"""
Date: 13/04/2020
Last Modification: 13/04/2020
Author: Maxime Cauté (maxime.caute@epfl.ch)
---
The functions in this file are used to manipulate the data as images.
"""

import numpy as np
import matplotlib.pyplot as plt

def request_image(situation, image_size):
    """
    Creates the image of the request map out of a situation.
    This image contains a 1 for the pickup, -1 for the other cases.
    ---
    Input:
        - situation:  int list list. The description of the situation,
            with the request as a first element, and the vehicles locations afterwards.
    Parameters:
        - image_size: int. The size of the image.
    Output:
        - image: int 2D-matrix. The map of the request: 1 for the pickup, -1 for the other cases.
    """
    image = - np.ones((image_size, image_size))

    request = situation[0]

    x_pickup, y_pickup = request[0], request[1]
    x_dropoff, y_dropoff = request[2], request[3]

    image[x_pickup, y_pickup] = 1
    #image[x_dropoff, y_dropoff] = -2

    return image

def vehicles_image(situation, image_size):
    """
    Creates the image of the vehicles map out of a situation.
    This image contains a 1 for the occupied cells, -1 for the others.
    ---
    Input:
        - situation:  int list list. The description of the situation,
            with the request as a first element, and the vehicles locations afterwards.
    Parameters:
        - image_size: int. The size of the image.
    Output:
        - image: int 2D-matrix. The map of the vehicles: 1 for the occupied cells, -1 for the others.
    """
    image = -np.ones((image_size, image_size))

    vehicles =  situation[1:]

    for vehicle in vehicles:
        x_vehicle, y_vehicle = vehicle[0], vehicle[1]

        image[x_vehicle, y_vehicle] = 1

    return image

def situation_to_2_channels(situation, image_size):
    """
    Creates a 2-channels image set to describe a situation.
    The first image is the request map, the second the vehicles map
    ---
    Input:
        - situation:  int list list. The description of the situation,
            with the request as a first element, and the vehicles locations afterwards.
    Parameters:
        - image_size: int. The size of the image.
    Output:
        - data: int 3D-matrix. The 2-channels image set to describe the situation.
    """
    data = []
    request = request_image(situation, image_size)
    data.append(request)

    vehicles = vehicles_image(situation, image_size)
    data.append(vehicles)

    return data

def situation_table_to_2_channels(table, image_size):
    """
    Creates a table of 2-channels image sets from a situation table.
    ---
    Input:
        - table:  int list list list. A table containing situations description.
    Parameters:
        - image_size: int. The size of the image.
    Output:
        - data_table: int 3D-matrix list. The table containing the 2-channels image sets
        from the situations descriptions in table.
    """
    data_table = [situation_to_2_channels(situation, image_size) for situation in table]

    return data_table

def assignement_image(assignement, image_size):
    """
    Creates the image of the assignement map out of an assignement.
    This image contains a 1 for the assigned cell, -1 for the others.
    ---
    Input:
        -assignement:  int list list. The description of the assignement, i.e. its coordinates.
    Parameters:
        - image_size: int. The size of the image.
    Output:
        - image: int 2D-matrix. The map of the assignement: 1 for the occupied cells, -1 for the others.
    """
    image = np.zeros((image_size, image_size))

    x_assignement, y_assignement = assignement[0], assignement[1]

    image[x_assignement, y_assignement] = 1

    return image

def assignement_table_to_image(table, image_size):
    """
    Creates a table of assignement images from an assignement table.
    ---
    Input:
        - table:  int list list. A table containing assignements description.
    Parameters:
        - image_size: int. The size of the image.
    Output:
        - data: int 3D-matrix. The table containing the assignement images
        from the assignements description in table.
    """
    data_table = [assignement_image(assignement, image_size) for assignement in table]
    return data_table

def visualize_image(image , title = ""):
    """
    Displays an image.
    ---
    Input:
        - image: int 2D-matrix. the image to be displayed.
    Parameters:
        - title: string. An eventual title to be displayed.
    """
    plt.matshow(image, cmap = 'coolwarm')
    plt.title(title)
