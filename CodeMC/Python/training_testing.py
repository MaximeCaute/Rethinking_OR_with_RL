"""
Date: 13/04/2020
Last Modification: 18/06/2020
Author: Maxime Cauté (maxime.caute@epfl.ch)
---
The functions in this file are used to train and test neural networks.
"""
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

import data_images
import policies
import data_saver
import failure_testing

#Why not data2tensor? Moved if I recall
def get_input_data_tensor(situations_images, batch_indices = [], data_norm = False):
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

    input_tensor = torch.tensor(input_array).type(torch.FloatTensor)

    return input_tensor

def get_output_data_tensor(assignements_images, batch_indices = []):
    # Setting batch indices as whole set if not specified.
    if batch_indices == []:
        batch_indices = [i for i in range(len(situations_images))]
    output_tensor = torch.tensor([assignements_images[i] for i in batch_indices]).type(torch.FloatTensor)

    return output_tensor

# TODO move
def get_network_tensors(situations_images, assignements_images, batch_indices = [], data_norm = False):
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
    input_tensor  = get_input_data_tensor (situations_images, batch_indices =batch_indices, data_norm = data_norm)
    output_tensor = get_output_data_tensor (assignements_images, batch_indices = batch_indices)

    return input_tensor, output_tensor

def get_chosen_positions(output_tensor):
    # Detaching tensors due to gradient protections
    chosen_positions = [np.argmax( situation_tensor.detach() ) for situation_tensor in output_tensor]
    return np.array(chosen_positions)

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
    network_chosen_positions = get_chosen_positions(network_output_tensor)
    expert_chosen_positions =  get_chosen_positions(expert_output_tensor)

    matches = (network_chosen_positions == expert_chosen_positions)
    accuracy = np.mean(matches)
    return accuracy

def compute_loss(loss_criterion, network_output_tensor, expert_output_tensor):
    """
    This function computes the loss from an output tensor
    to a comparison tensor, according to a given loss criterion.
    ---
    Input:
        - loss_criterion: PyTorch Loss.
            A loss criterion as defined by PyTorch, to compute loss from.
        - network_output_tensor: float 3D tensor.
            The output tensor from the network.
                The first dimension is the batch dimension.
                The two other dimensions are the dimensions of the actual map.
        - expert_output_tensor: float 3D tensor.
            The ouptut tensor from the network.
                Dimensions akin to network_output_tensor.
    Output:
        - minibatch_loss: float. The loss over the batch.
    """
    batch_size = network_output_tensor.shape[0]

    # In case the loss criterion is a Weighted Loss,
    # data is processed accordingly
    if isinstance(loss_criterion, torch.nn.modules.loss._WeightedLoss):
        expert_choice_tensor = torch.tensor(get_chosen_positions(expert_output_tensor))
        minibatch_loss = loss_criterion(F.softmax(network_output_tensor.view(batch_size,-1), dim = 1),
                                        expert_choice_tensor)
    # Otherwise
    else:
        minibatch_loss = loss_criterion(network_output_tensor.view(batch_size,-1),
                                        expert_output_tensor.view(batch_size,-1) )
    return minibatch_loss

def epoch_from_indices(network, indices,
                       loss_criterion, optimizer = None,
                       minibatch_size = 50, epochs_id = -1,
                       data_norm = False, data_path = ""):
    """
    This function runs a training *OR* testing epoch for a network.
    The testing or training status is determined by the value of optimizer.
    ---
    Input:
        - network: PyTorch Module.
            The network to run an epoch from.
        - indices: int list.
            The indices of the tables on which to run the epoch.
        - loss_citerion: PyTorch Loss.
            A loss criterion as defined by PyTorch, to compute loss from.
    Parameters:
        - optimizer: PyTorch Optimizer or None.
            The optimizer to update network weights for training.
                If None, network weights are not updated (testing phase).
                Defaults to None.
        - minibatch_size: int.
            The size of the minibatches for the epoch (network is trained
            over the whole minibatch before updating weights).
                Defaults to 50.
        - epochs_id: int.
            The id of the current epoch.
                Defaults to -1.
        - data_norm: boolean.
            Regulates data normalization in tensor creation.
                Defaults to False.
        - data_path: string.
            The path to the folder where data is located.
                Defaults to local folder.
    Output:
        - summed_loss: float.
            The total loss over the epoch.
        - accuracies: float list.
            The accuracy of the network over each minibatch.
    """

    for i in indices:

        situations_table = data_saver.load_table(i, folder = "situations_tables", relative_path = data_path)
        situations_images = data_images.situation_table_to_2_channels(situations_table, network.image_size)

        assignements_table = data_saver.load_table(i, folder = "assignements_tables", relative_path = data_path)
        assignements_images = data_images.assignement_table_to_image(assignements_table, network.image_size)

        batch_size = len(situations_images)
        batch_indices = range(batch_size)

        accuracies = []
        summed_loss = 0

        for b in range(0, batch_size, minibatch_size):
            minibatch_indices = batch_indices[b:b+minibatch_size]


            # Setting gradient to zero for training
            if optimizer is not None:
                optimizer.zero_grad()

            input_tensor = get_input_data_tensor(situations_images, minibatch_indices, data_norm = data_norm)

            network_output_tensor = network(input_tensor)
            expert_output_tensor = get_output_data_tensor(assignements_images, minibatch_indices)

            # Computing loss on flattened tensor output
            minibatch_loss = compute_loss(loss_criterion, network_output_tensor, expert_output_tensor)

            if optimizer is not None:
                minibatch_loss.backward()
                optimizer.step()

            minibatch_accuracy = compute_accuracy(network_output_tensor, expert_output_tensor)

            summed_loss+= minibatch_loss.item()
            accuracies.append(minibatch_accuracy)

    return summed_loss, accuracies


def train_and_test_from_indices(network, training_indices, testing_indices,
                                loss_criterion, optimizer, quick_start = True,
                                epochs_amount = 100, minibatch_size = 50,
                                verbose = False, learning_curve = False,
                                break_accuracy = 1.2, data_norm = False,
                                data_path = "", start_epoch = 0):
"""
This function handles the whole training and testing of a network.
---
Input:
    - network: Pytorch Module.
        The network to train.
    - training_indices: int list.
        The list of the indices of the table on which the training is performed.
    - testing_indices: int list.
        The list of the indices of the table on which the testing is performed.
    - loss_citerion: PyTorch Loss.
        A loss criterion as defined by PyTorch, to compute loss from.
    - optimizer: PyTorch Optimizer.
        The optimizer to update network weights for training.
Parameters:
    - epochs_amount: int.
        The maximum number of epochs to be runned.
            Defaults to 100.
    - verbose: boolean.
        Toggles display of additional information.
            Defaults to False.
    - learning_curve: boolean.
        Toggles display of the learning curve of the network.
            Defaults to False.
    - break_accuracy: float.
        The training accuracy over which the run is stopped.
            Avoids to do all epochs if unnecessary.
    - data_norm: boolean.
        Regulates data normalization in tensor creation.
            Defaults to False.
    - data_path: string.
        The path to the folder where data is located.
            Defaults to local folder.
    - start_epoch: int.
        The epoch at which the training starts.
            If not 0, model saved at this epoch is loaded.
Output:
    - max_accuracy: float.
        The maximal testing accuracy hit by the network.
    - max_accuracy_epoch: int.
        The id of the epoch where the maximal testing accuracy was hit.
"""


    training_accuracies = []
    training_losses = []
    testing_accuracies = []
    testing_losses = []
    epochs = range(start_epoch, epochs_amount)
    if start_epoch!=0:
        network = data_saver.load_model(str(start_epoch-1), folder = "Models", relative_path = "./")

    # Learning rate handling.
    if quick_start and start_epoch==0:
        print("Increasing lr from: "+str(optimizer.param_groups[0]['lr']))
        for g in optimizer.param_groups:
            g['lr']+=0.0002
        print("to: "+str(g['lr'])+"\n", flush=True)

    for e in epochs:
        g = optimizer.param_groups[0]
        if e == 10 and quick_start:
            print("Decreasing lr from: "+str(g['lr']))
            for g in optimizer.param_groups:
                g['lr']-=0.0001
            print("to: "+str(g['lr'])+"\n", flush=True)
        if e == 20 and quick_start:
            print("Decreasing lr from: "+str(g['lr']))
            for g in optimizer.param_groups:
                g['lr']-=0.0001
            print("to: "+str(g['lr'])+"\n", flush=True)
        if e == 100:
            print("Decreasing lr from: "+str(g['lr']))
            for g in optimizer.param_groups:
                g['lr']-=0.0002
            print("to: "+str(g['lr'])+"\n", flush=True)
        if e == 150:
            print("Decreasing lr from: "+str(g['lr']))
            for g in optimizer.param_groups:
                g['lr']-=0.0002
            print("to: "+str(g['lr'])+"\n", flush=True)

        training_loss, training_epoch_accuracies = epoch_from_indices(network, training_indices,
                                                             loss_criterion, optimizer = optimizer,
                                                             minibatch_size = minibatch_size,
                                                             epochs_id = e, data_norm = data_norm,
                                                             data_path = data_path)
        testing_loss, testing_epoch_accuracies =   epoch_from_indices(network, testing_indices,
                                                             loss_criterion, optimizer = None,
                                                             minibatch_size = minibatch_size,
                                                             epochs_id = e, data_norm = data_norm,
                                                             data_path = data_path)

        # Periodical network saving.
        if int(e/50)== e/50:
            vehicle_selection_accuracy = failure_testing.evaluate_vehicle_selection(network, testing_indices, data_path= data_path)
            vehicle_distance = failure_testing.evaluate_vehicle_distance(network, testing_indices, data_path = data_path)
            print("Saving model... Vehicle selection accuracy: {:.3f}, Mean distance: {:.3f}".format(vehicle_selection_accuracy, vehicle_distance), flush= True)
            data_saver.save_model(network, str(e), folder = "Models/", relative_path = "./")

        training_accuracy = np.mean(training_epoch_accuracies)
        testing_accuracy = np.mean(testing_epoch_accuracies)

        training_accuracies.append(training_accuracy)
        training_losses.append(training_loss)
        testing_accuracies.append(testing_accuracy)
        testing_losses.append(testing_loss)

        if verbose:
            print("\rEpoch {}. Training Loss: {:.3f}, Training Accuracy: {:.3f}, Testing Loss: {:.3f}, Testing Accuracy: {:.3f} ".format(e, training_loss, training_accuracy, testing_loss, testing_accuracy), flush = True)
        if training_accuracy > break_accuracy:
            epochs = range(e+1)
            break

    max_accuracy_epoch = np.argmax(testing_accuracies)
    max_accuracy = testing_accuracies[max_accuracy_epoch]

    if verbose:
        print("Max accuracy reached at epoch {}, with value: {:.3f}".format(max_accuracy_epoch, max_accuracy))

    if learning_curve:
        f, ax = plt.subplots(2)
        ax[0].set_title("Accuracy over time")
        ax[0].plot(epochs, training_accuracies, color='black', linestyle='dashed')
        ax[0].plot(epochs, testing_accuracies, color='red', linestyle='solid')

        ax[1].set_title("Loss over time")
        ax[1].plot(epochs, training_losses, color='black', linestyle='dashed')
        ax[1].plot(epochs, testing_losses, color='red', linestyle='solid')

        plt.show()

    return max_accuracy, max_accuracy_epoch
