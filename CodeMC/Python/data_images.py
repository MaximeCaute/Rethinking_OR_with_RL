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
    image = np.zeros((image_size, image_size))

    request = situation[0]

    x_pickup, y_pickup = request[0], request[1]
    x_dropoff, y_dropoff = request[2], request[3]

    image[x_pickup, y_pickup] = 1
    image[x_dropoff, y_dropoff] = -1

    return image

def vehicles_image(situation, image_size):
    image = np.zeros((image_size, image_size))

    vehicles =  situation[1:]

    for vehicle in vehicles:
        x_vehicle, y_vehicle = vehicle[0], vehicle[1]

        image[x_vehicle, y_vehicle]+= 1

    return image

def situation_to_2_channels(situation, image_size):
    data = []
    request = request_image(situation, image_size)
    data.append(request)

    vehicles = vehicles_image(situation, image_size)
    data.append(vehicles)

    return data

def situation_table_to_2_channels(table, image_size):
    data_table = [situation_to_2_channels(situation, image_size) for situation in table]

    return data_table

def assignement_image(assignement, image_size):
    image = np.zeros((image_size, image_size))

    x_assignement, y_assignement = assignement[0], assignement[1]

    image[x_assignement, y_assignement] = 1

    return image

def assignement_table_to_image(table, image_size):
    data_table = [assignement_image(assignement, image_size) for assignement in table]
    return data_table

def visualize_image(image , title = ""):
    plt.matshow(image, cmap = 'coolwarm')
    plt.title(title)
