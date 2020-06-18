"""
Date: 25/05/2020
Last Modification: 18/06/2020
Author: Maxime Cauté (maxime.caute@epfl.ch)
---
The functions in this file are used to convert data to torch tensors.
"""
import numpy as np
import torch

def get_input_data_tensor(situations_images, batch_indices = [], data_norm = False):
    """
    This function transforms a set of situation images into an input tensor for a network.
    ---
    Input:
        - situations_images: float 4D array. The set of situation images.
                The first dimension is the batch dimension.
                The second dimension is the channels dimension
                The third and fourth dimensions reflect the actual 2D map.
    Parameters:
        - batch_indices: int list. The indices of the batch element to be taken.
                Defaults to the whole batch.
        - data_norm: boolean. Regulates data normalization in tensor creation.
                Defaults to False.
    Output:
        - input_tensor: float 4D tensor. The input tensor from the situations images set.
                Dimensions akin to situations_images.
    """
    # Setting batch indices as whole set if not specified.
    if batch_indices == []:
        batch_indices = [i for i in range(len(situations_images))]

    input_array = np.array([situations_images[i] for i in batch_indices])


    # Data normalization per channel
    if data_norm:
        channels_amount = input_array.shape[1]
        for ch in range(channels_amount):
            input_array[:,ch,:,:]-= np.mean(input_array[:, ch, :, :], axis = 0)
            input_array[:,ch,:,:]/= np.std(input_array[:, ch, :, :], axis = 0)

    # Converting the input array into a PyTorch Tensor.
    input_tensor = torch.tensor(input_array).type(torch.FloatTensor)

    return input_tensor

def get_output_data_tensor(assignements_images, batch_indices = []):
    """
    This function transforms a set of assignement_images into an output tensor for the network to compare to.
    ---
    Input:
        - assignements_images: float 3D array. The set of assignement images.
                The first dimension is the batch dimension.
                The second and third dimensions reflect the actual 2D map.
    Parameters:
        - batch_indices: int list. The indices of the batch element to be taken.
                Defaults to the whole batch.
    Output:
        - output_tensor: float 3D tensor. The output tensor from the assignements images set.
                Dimensions akin to assignements_images.
    """
    # Setting batch indices as whole set if not specified.
    if batch_indices == []:
        batch_indices = [i for i in range(len(situations_images))]
    output_tensor = torch.tensor([assignements_images[i] for i in batch_indices]).type(torch.FloatTensor)

    return output_tensor

def get_tensors(situations_images, assignements_images,
                        batch_indices = [], data_norm = False):
    """
    This functions prepares the tensors for a network training or testing.
    ---
    Input:
        - situations_images: float 4D array.
            The set of situation images.
                The first dimension is the batch dimension.
                The second dimension is the channels dimension
                The third and fourth dimensions reflect the actual 2D map.
        - assignements_images: float 3D array.
            The set of assignement images.
                The first dimension is the batch dimension.
                The second and third dimensions reflect the actual 2D map.
    Parameters:
        - batch_indices: int list.
            The indices of the batch element to be taken.
                Defaults to the whole batch.
        - data_norm: boolean.
            Regulates data normalization in tensor creation.
                Defaults to False.
    Output:
        - input_tensor: float 4D tensor.
            The input tensor from the situations images set.
                Dimensions akin to situations_images.
        - output_tensor: float 3D tensor.
            The output tensor from the assignements images set.
                Dimensions akin to assignements_images.
    """
    input_tensor  = get_input_data_tensor(situations_images,
                                        batch_indices = batch_indices,
                                        data_norm = data_norm)
    output_tensor = get_output_data_tensor (assignements_images,
                                            batch_indices = batch_indices)

    return input_tensor, output_tensor

def get_chosen_positions(output_tensor):
    # TO BE MOVED?
    """
    This function extracts the set of chosen positions from an output tensor.
    ---
    Input:
        - output_tensor: float 3D tensor. An output tensor from an assignements images set or a network output.
                The first dimension is the batch dimension.
                The second and third dimensions reflect the actual 2D map.
    Output:
        - chosen_positions: 2D numpy array. The set of indices of chosen positions for every assignement
            in the batch.
                The first dimension is the batch dimension.
                The second dimension is the indices dimension (x,y) on the map.
    """
    # Detaching tensors due to gradient protections
    chosen_positions = [np.argmax( situation_tensor.detach() ) for situation_tensor in output_tensor]
    return np.array(chosen_positions)
