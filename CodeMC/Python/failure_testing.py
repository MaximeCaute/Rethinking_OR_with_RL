"""
Date: 23/05/2020
Last Modification: 18/06/2020
Author: Maxime Cauté (maxime.caute@epfl.ch)
---
The functions in this file are used to evaluate the failure of a network.
"""
import numpy as np
import data_generator
import data_saver
import data_images
import data2tensor
import torch

def compute_accuracy(network_output_tensor, expert_output_tensor):
    """
    This functions compute the accuracy from an output tensor
    to a comparison tensor. This accuracy is the percentage of the time in which
    the matching cell is selected.
    ---
    Input:
        - network_output_tensor: float 3D tensor.
            The output tensor from the network.
                The first dimension is the batch dimension.
                The two other dimensions are the dimensions of the actual map.
        - expert_output_tensor: float 3D tensor.
            The ouptut tensor from the network.
                Dimensions akin to network_output_tensor.
    Output:
        - accuracy: float.
            The accuracy of the network.
    """
    network_chosen_positions = data2tensor.get_chosen_positions(network_output_tensor)
    expert_chosen_positions =  data2tensor.get_chosen_positions(expert_output_tensor)

    matches = (network_chosen_positions == expert_chosen_positions)
    accuracy = np.mean(matches)
    return accuracy

def compute_vehicle_selection_accuracy(network_output_tensor, input_tensor):
    """
    This function computes the accuracy of vehicle selection. In other words,
        the number of assignements that are actually a real vehicle.
    ---
    Input:
        - network_output_tensor: float 3D tensor. The output tensor from a network
            (or any assignement images batch) describing the assignements.
                The first dimension is the batch dimension.
                The second dimension is the indices dimension (x,y) on the map.
        - input_tensor: float 4D tensor. The input tensor to the network
            (or any situation images batch) describing the input situation.
                The first dimension is the batch dimension.
                The second dimension is the channels dimension
                The third and fourth dimensions reflect the actual 2D map.
    """
    situations_amount = network_output_tensor.shape[0]
    accuracies = []
    for i in range(situations_amount):
        # Tensors are detached due to gradient protection.
        situation = input_tensor[i].detach()
        output = network_output_tensor[i].detach()

        # isn't it better to reuse data2tensor function?
        chosen_cell_index = np.argmax(output)
        # argmax index is here 1D index so we flatten situation
        vehicle_cell_content = int(situation[1].view((-1))[chosen_cell_index])
        accuracies.append( max(vehicle_cell_content, 0) )

    accuracy = np.mean(accuracies)
    return accuracy



def evaluate_vehicle_selection(network, testing_indices, data_path = "", minibatch_size = 50, data_norm = False):
    """
    This functions evaluates the vehicle selection accuracy of a network on a set of testing indices.
    ---
    Input:
        - network: PyTorch Module. The network to be evaluated.
        - testing_indices: int list. The set of table testing indices on which the network will be evaluated.
    Parameters:
        - data_path: string. The path at which the table data can be found.
                Defaults to local folder.
        - minibatch_size: int. The size of a minibatch.
                These are used to limit the amount of memory used in a single run over a batch.
                Defaults to 50.
        - data_norm: boolean. Regulates data normalization when creating input tensor.
    Output:
        - accuracy: float. The network accuracy on assigning a proper vehicle.
    """
    for i in testing_indices:

        situations_table = data_saver.load_table(i, folder = "situations_tables", relative_path = data_path)
        situations_images = data_images.situation_table_to_2_channels(situations_table, network.image_size)

        assignements_table = data_saver.load_table(i, folder = "assignements_tables", relative_path = data_path)
        assignements_images = data_images.assignement_table_to_image(assignements_table, network.image_size)

        batch_size = len(situations_images)
        batch_indices = range(batch_size)

        accuracies = []

        for b in range(0, batch_size, minibatch_size):
            minibatch_indices = batch_indices[b:b+minibatch_size]

            input_tensor = data2tensor.get_input_data_tensor(situations_images, minibatch_indices, data_norm = data_norm)

            network_output_tensor = network(input_tensor)
            expert_output_tensor = data2tensor.get_output_data_tensor(assignements_images, minibatch_indices)

            minibatch_accuracy = compute_vehicle_selection_accuracy(network_output_tensor, input_tensor)

            accuracies.append(minibatch_accuracy)
    accuracy = np.mean(accuracies)
    return accuracy

def compute_distance_to_choice(network_output_tensor, expert_output_tensor):
    """
    This functions compute the mean distance between network assignements and expert assignements.
    ---
    Input:
        - network_output_tensor: float 3D array. The assignements from the network.
                The first dimension is the batch dimension.
                The second and third dimensions reflect the actual 2D map.
        - expert_output_tensor: float 3D array. The assignements from the expert.
                Dimensions akin to network_output_tensor.
    Output:
        - mean_distance: float. The mean distance between assignements.
    """
    situations_amount = network_output_tensor.shape[0]
    distances = []
    for i in range(situations_amount):
        # Tensors detached due to gradient protections.
        expert_choice = expert_output_tensor[i].detach()
        network_choice = network_output_tensor[i].detach()

        # Perhaps use dedicated function?
        network_chosen_cell_indices = np.unravel_index(
                                            np.argmax(network_choice, axis=None), network_choice.shape
                                            )

        expert_chosen_cell_indices = np.unravel_index(
                                            np.argmax(expert_choice, axis=None), expert_choice.shape
                                            )

        distance = data_generator.distance(network_chosen_cell_indices, expert_chosen_cell_indices)
        distances.append(distance)

    mean_distance = np.mean(distances)
    return mean_distance

def evaluate_vehicle_distance(network, testing_indices, data_path = "", minibatch_size = 50, data_norm = False):
    # TODO Rename ?
    """
    This function evaluates the distance to the expert choice of a network's assignements.
    ---
    Input:
        - network: PyTorch Module. The network to be evaluated.
        - testing_indices: int list. The set of table testing indices on which the network will be evaluated.
    Parameters:
        - data_path: string. The path at which the table data can be found.
                Defaults to local folder.
        - minibatch_size: int. The size of a minibatch.
                These are used to limit the amount of memory used in a single run over a batch.
                Defaults to 50.
        - data_norm: boolean. Regulates data normalization when creating input tensor.
    Output:
        - distance: float. The mean distance to the expert assignement.
    """
    for i in testing_indices:

        situations_table = data_saver.load_table(i, folder = "situations_tables", relative_path = data_path)
        situations_images = data_images.situation_table_to_2_channels(situations_table, network.image_size)

        assignements_table = data_saver.load_table(i, folder = "assignements_tables", relative_path = data_path)
        assignements_images = data_images.assignement_table_to_image(assignements_table, network.image_size)

        batch_size = len(situations_images)
        batch_indices = range(batch_size)

        distances = []

        for b in range(0, batch_size, minibatch_size):
            minibatch_indices = batch_indices[b:b+minibatch_size]

            input_tensor = data2tensor.get_input_data_tensor(situations_images, minibatch_indices, data_norm = data_norm)

            network_output_tensor = network(input_tensor)
            expert_output_tensor = data2tensor.get_output_data_tensor(assignements_images, minibatch_indices)

            minibatch_distance = compute_distance_to_choice(network_output_tensor, expert_output_tensor)

            distances.append(minibatch_distance)
    distance = np.mean(distances)
    return distance

def compute_closest_vehicle_accuracy(network_output_tensor, input_tensor, expert_output_tensor):
    """
    This function computes the closest vehicle (or loose) accuracy of a network output tensor:
    that is the accuracy considering the choice is not the cell of higher score but the vehicle closest to it.
    ---
    Input:
        - network_output_tensor: float 3D tensor. The output tensor from a network
            (or any assignement images batch) describing the assignements.
                The first dimension is the batch dimension.
                The second dimension is the indices dimension (x,y) on the map.
        - input_tensor: float 4D tensor. The input tensor to the network
            (or any situation images batch) describing the input situation.
                The first dimension is the batch dimension.
                The second dimension is the channels dimension
                The third and fourth dimensions reflect the actual 2D map.
        - expert_output_tensor: float 3D array. The assignements from the expert.
                Dimensions akin to network_output_tensor.
    Output:
        - accuracy: float. The loose accuracy of the network.
    """
    situations_amount = network_output_tensor.shape[0]
    accuracies = []
    for i in range(situations_amount):
        situation = input_tensor[i].detach()
        network_choice = network_output_tensor[i].detach()
        expert_choice = expert_output_tensor[i].detach()

        network_chosen_cell_indices = np.unravel_index(
                                            np.argmax(network_choice, axis=None), network_choice.shape
                                            )

        expert_chosen_cell_indices = np.unravel_index(
                                            np.argmax(expert_choice, axis=None), expert_choice.shape
                                            )
        # Vehicles are indices of cells valued 1 in second situation channel,
        # but np.argwhere does return x indices list and y indices list
        vehicles_raw_cell_indices = np.argwhere(situation[1] == 1)
        vehicles_cell_indices = [(  vehicles_raw_cell_indices[0][i],
                                    vehicles_raw_cell_indices[1][i])
                                    for i in range(vehicles_raw_cell_indices.shape[1])]
        max_distance = situation.shape[0]**2
        closest_vehicle_positions = []
        for vehicle_position in vehicles_cell_indices:
            # There are type issues with tensors, so we convert into integers.
            distance =  data_generator.distance((int(vehicle_position[0]), int(vehicle_position[1])),
                                                network_chosen_cell_indices)

            if distance < max_distance:
                closest_vehicle_positions = [vehicle_position]
                max_distance = distance
            elif distance == max_distance:
                closest_vehicle_positions.append(vehicle_position)


        accuracies+=[1 if expert_chosen_cell_indices in closest_vehicle_positions else 0]
        accuracy = np.mean(accuracies)
    return accuracy

def evaluate_loose_accuracy(network, testing_indices, data_path = "", minibatch_size = 50, data_norm = False):
    """
    This function evaluates the loose accuracy of a network:
    that is its accuracy selecting not the cell of higher score
    but the closest vehicle to it.
    ---
    Input:
        - network: PyTorch Module. The network to be evaluated.
        - testing_indices: int list. The set of table testing indices on which the network will be evaluated.
    Parameters:
        - data_path: string. The path at which the table data can be found.
            Defaults to local folder.
        - minibatch_size: int. The size of a minibatch.
            These are used to limit the amount of memory used in a single run over a batch.
            Defaults to 50.
        - data_norm: boolean. Regulates data normalization when creating input tensor.
    Output:
        - accuracy: float. The loose accuracy of the network.
    """
    for i in testing_indices:

        situations_table = data_saver.load_table(i, folder = "situations_tables", relative_path = data_path)
        situations_images = data_images.situation_table_to_2_channels(situations_table, network.image_size)

        assignements_table = data_saver.load_table(i, folder = "assignements_tables", relative_path = data_path)
        assignements_images = data_images.assignement_table_to_image(assignements_table, network.image_size)

        batch_size = len(situations_images)
        batch_indices = range(batch_size)

        accuracies = []

        for b in range(0, batch_size, minibatch_size):
            minibatch_indices = batch_indices[b:b+minibatch_size]

            input_tensor = data2tensor.get_input_data_tensor(situations_images, minibatch_indices, data_norm = data_norm)

            network_output_tensor = network(input_tensor)
            expert_output_tensor = data2tensor.get_output_data_tensor(assignements_images, minibatch_indices)

            minibatch_accuracy = compute_closest_vehicle_accuracy(network_output_tensor, input_tensor, expert_output_tensor)

            accuracies.append(minibatch_accuracy)
    accuracy = np.mean(accuracies)
    return accuracy
