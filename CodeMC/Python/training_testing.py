"""
Date: 13/04/2020
Last Modification: 13/04/2020
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

def get_network_tensors(situations_images, assignements_images, batch_indices = [], data_norm = False):
    input_tensor  = get_input_data_tensor (situations_images, batch_indices =batch_indices, data_norm = data_norm)
    output_tensor = get_output_data_tensor (assignements_images, batch_indices = batch_indices)

    return input_tensor, output_tensor

def get_chosen_positions(output_tensor):
    # Detaching tensors due to gradient protections
    chosen_positions = [np.argmax( situation_tensor.detach() ) for situation_tensor in output_tensor]
    return np.array(chosen_positions)

def compute_accuracy(network_output_tensor, expert_output_tensor):
    network_chosen_positions = get_chosen_positions(network_output_tensor)
    expert_chosen_positions =  get_chosen_positions(expert_output_tensor)

    matches = (network_chosen_positions == expert_chosen_positions)
    #accuracy = float(matches.sum())/len(matches)
    accuracy = np.mean(matches)
    return accuracy

def compute_loss(loss_criterion, network_output_tensor, expert_output_tensor):
    batch_size = network_output_tensor.shape[0]

    if isinstance(loss_criterion, torch.nn.modules.loss._WeightedLoss):
        expert_choice_tensor = torch.tensor(get_chosen_positions(expert_output_tensor))
        minibatch_loss = loss_criterion(F.softmax(network_output_tensor.view(batch_size,-1), dim = 1),
                                        expert_choice_tensor)
    else:
        minibatch_loss = loss_criterion(network_output_tensor.view(batch_size,-1),
                                        expert_output_tensor.view(batch_size,-1) )
    return minibatch_loss



# def epoch_from_images(network, situations_images, assignements_images,
#                               loss_criterion, optimizer = None,
#                               minibatch_size = 50, epochs_id = -1):
#     """
#     Test if optimizer is none, training otherwise
#     """
#     batch_size = len(situations_images)
#     batch_indices = range(batch_size)
#
#     accuracies = []
#     summed_loss = 0
#
#     for b in range(0, batch_size, minibatch_size):
#         minibatch_indices = batch_indices[b:b+minibatch_size]
#
#
#         # Setting gradient to zero for training
#         if optimizer is not None:
#             optimizer.zero_grad()
#
#         #print(np.array(situations_images).shape, np.array(assignements_images).shape)
#         input_tensor = get_input_data_tensor(situations_images, minibatch_indices)
#         print(minibatch_indices)
#         network_output_tensor = network(input_tensor)
#         expert_output_tensor = get_output_data_tensor(assignements_images, minibatch_indices)
#
#         # Computing loss on flattened tensor output
#         #minibatch_loss = loss_criterion(network_output_tensor.view(-1, network.image_size**2),
#         #                                expert_output_tensor.view(-1, network.image_size**2) )
#         print(network_output_tensor.shape, expert_output_tensor.shape)
#         minibatch_loss = compute_loss(loss_criterion, network_output_tensor, expert_output_tensor)
#
#         if optimizer is not None:
#             minibatch_loss.backward()
#             optimizer.step()
#
#         minibatch_accuracy = compute_accuracy(network_output_tensor, expert_output_tensor)
#
#         summed_loss+= minibatch_loss.item()
#         accuracies.append(minibatch_accuracy)
#
#     return summed_loss, accuracies

def epoch_from_indices(network, indices,
                       loss_criterion, optimizer = None,
                       minibatch_size = 50, epochs_id = -1,
                       data_norm = False):
    """
    Test if optimizer is none, training otherwise
    """

    for i in indices:

        situations_table = data_saver.load_table(i, folder = "situations_tables")
        situations_images = data_images.situation_table_to_2_channels(situations_table, network.image_size)

        assignements_table = data_saver.load_table(i, folder = "assignements_tables")
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

            #print(np.array(situations_images).shape, np.array(assignements_images).shape)
            input_tensor = get_input_data_tensor(situations_images, minibatch_indices, data_norm = data_norm)

            network_output_tensor = network(input_tensor)
            expert_output_tensor = get_output_data_tensor(assignements_images, minibatch_indices)

            # Computing loss on flattened tensor output
            #minibatch_loss = loss_criterion(network_output_tensor.view(-1, network.image_size**2),
            #                                expert_output_tensor.view(-1, network.image_size**2) )
            minibatch_loss = compute_loss(loss_criterion, network_output_tensor, expert_output_tensor)

            if optimizer is not None:
                minibatch_loss.backward()
                optimizer.step()

            minibatch_accuracy = compute_accuracy(network_output_tensor, expert_output_tensor)

            summed_loss+= minibatch_loss.item()
            accuracies.append(minibatch_accuracy)

    return summed_loss, accuracies


# def train_and_test_from_images(network,
#                                training_situation_images, training_assignement_images,
#                                testing_situation_images, testing_assignement_images,
#                                loss_criterion, optimizer,
#                                epochs_amount = 15, minibatch_size = 50,
#                                verbose = False):
#
#     testing_accuracies = []
#
#     for e in range(epochs_amount):
#         training_loss, training_accuracy = epoch_from_images(network, training_situation_images, training_assignement_images,
#                                                              loss_criterion, optimizer = optimizer,
#                                                              minibatch_size = minibatch_size,
#                                                              epochs_id = e)
#         testing_loss, testing_accuracy =   epoch_from_images(network, testing_situation_images, testing_assignement_images,
#                                                              loss_criterion, optimizer = None,
#                                                              minibatch_size = minibatch_size,
#                                                              epochs_id = e)
#
#
#         testing_accuracies.append(testing_accuracy)
#
#         if verbose:
#             print("\rEpoch {}. Training Loss: {:.3f}, Training Accuracy: {:.3f}, Testing Loss: {:.3f}, Testing Accuracy: {:.3f} ".format(e, training_loss, training_accuracy, testing_loss, testing_accuracy))
#
#     max_accuracy_epoch = np.argmax(testing_accuracies)
#     max_accuracy = testing_accuracies[max_accuracy_epoch]
#
#     if verbose:
#         print("Max accuracy reached at epoch {}, with value: {:.3f}".format(max_accuracy_epoch, max_accuracy))
#
#     return max_accuracy, max_accuracy_epoch

def train_and_test_from_indices(network, training_indices, testing_indices,
                                loss_criterion, optimizer,
                                epochs_amount = 15, minibatch_size = 50,
                                verbose = False, learning_curve = False,
                                break_accuracy = 1.2, data_norm = False):

    training_accuracies = []
    training_losses = []
    testing_accuracies = []
    testing_losses = []
    epochs = range(epochs_amount)

    for e in epochs:
        training_loss, training_epoch_accuracies = epoch_from_indices(network, training_indices,
                                                             loss_criterion, optimizer = optimizer,
                                                             minibatch_size = minibatch_size,
                                                             epochs_id = e, data_norm = data_norm)
        testing_loss, testing_epoch_accuracies =   epoch_from_indices(network, testing_indices,
                                                             loss_criterion, optimizer = None,
                                                             minibatch_size = minibatch_size,
                                                             epochs_id = e, data_norm = data_norm)


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
