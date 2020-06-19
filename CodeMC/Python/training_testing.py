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
import data2tensor

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
        expert_choice_tensor = torch.tensor(
                        data2tensor.get_chosen_positions(expert_output_tensor)
                    )
        minibatch_loss = loss_criterion(
                F.softmax( network_output_tensor.view(batch_size,-1), dim = 1),
                expert_choice_tensor
            )
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

            input_tensor, expert_output_tensor = data2tensor.get_tensors(
                                        situations_images, assignements_images,
                                        batch_indices=minibatch_indices,
                                        data_norm=False)
            network_output_tensor = network(input_tensor)

            # Computing loss on flattened tensor output
            minibatch_loss = compute_loss(loss_criterion, network_output_tensor, expert_output_tensor)

            if optimizer is not None:
                minibatch_loss.backward()
                optimizer.step()

            minibatch_accuracy = failure_testing.compute_accuracy(
                                                    network_output_tensor,
                                                    expert_output_tensor)

            summed_loss+= minibatch_loss.item()
            accuracies.append(minibatch_accuracy)

    return summed_loss, accuracies


def train_and_test_from_indices(network, training_indices, testing_indices,
                                loss_criterion, optimizer,
                                epochs_amount = 100, minibatch_size = 50,
                                verbose = False, learning_curve = False,
                                break_accuracy = 1.2, data_norm = False,
                                data_path = "", start_epoch = 0,
                                warmup_time=0, adaptative_lr = {},
                                eval_frequency=0):
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
        - warmup_time: int.
            The number of epoch dedicated to warm-up
            (linear augmentation of the learning rate).
                Defaults to no warm-up.
        - adaptative_lr: int->float dict.
            A dictionnary of fixed-epoch variations of the learning rate.
                The keys are the epochs indices.
        - eval_frequency: int.
            The frequency in epochs for the network evaluation and save.
                Value zero deactivates saves and evaluation.
                Defaults to deactivated.
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
        network = data_saver.load_model(str(start_epoch-1), folder = "Models", relative_path = data_path)

    for e in epochs:
        # Learning rate handling.
        g = optimizer.param_groups[0]
        lr = g['lr']
        # Warm-up
        warmup_time = 0
        if e < warmup_time:
            for g in optimizer.param_groups:
                    g['lr'] = lr*(e+1)/20
            if verbose:
                txt = ("Warming up to learning rate: {}. "+
                "Current learning rate is {}.")
                print(txt.format(lr, g['lr']), flush=True)
        if e in adaptative_lr.keys():
            for g in optimizer.param_groups:
                    g['lr'] = adaptative_lr[e]
            if verbose:
                txt ="Updating learning rate from {} to {}."
                print(txt.format(lr,g['lr']), flush=True)

        training_loss, training_epoch_accuracies = epoch_from_indices(
                                        network, training_indices,
                                        loss_criterion, optimizer = optimizer,
                                        minibatch_size = minibatch_size,
                                        epochs_id = e, data_norm = data_norm,
                                        data_path = data_path)
        testing_loss, testing_epoch_accuracies =   epoch_from_indices(
                                        network, testing_indices,
                                        loss_criterion, optimizer = None,
                                        minibatch_size = minibatch_size,
                                        epochs_id = e, data_norm = data_norm,
                                        data_path = data_path)

        # Periodical network saving and evaluation.
        if eval_frequency != 0 and int(e/eval_frequency)== e/eval_frequency:
            vehicle_selection_accuracy = failure_testing.evaluate_vehicle_selection(
                                                network,
                                                testing_indices,
                                                data_path= data_path,
                                                minibatch_size=minibatch_size,
                                                data_norm=data_norm)
            vehicle_distance = failure_testing.evaluate_vehicle_distance(
                                                network,
                                                testing_indices,
                                                data_path = data_path,
                                                minibatch_size=minibatch_size,
                                                data_norm = data_norm)
            loose_accuracy = failure_testing.evaluate_loose_accuracy(
                                                network,
                                                testing_indices,
                                                data_path = data_path,
                                                minibatch_size=minibatch_size,
                                                data_norm = data_norm)
            data_saver.save_model(
                        network, str(e),
                        folder = "Models", relative_path = data_path)

            txt = ("Saving model..."+
                    "Vehicle selection accuracy: {:.3f}, "+
                    "Mean distance: {:.3f}, "+
                    "Loose accuracy: {:.3}.")
            print(txt.format(
                        vehicle_selection_accuracy,
                        vehicle_distance,
                        loose_accuracy), flush= True)


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


if __name__ == "__main__":
    """
    Upon being executed, trains and tests a network.
    ---
    Several arguments can be added, see 'train_and_test_from_indices'
    description for more precision.
    Learning rate can also be set with -lr.
    Networks parameters can also be specified, see their definitions.

    Call examples:
        python3 training_testing.py 10 30
        python3 training_testing.py 10 30 -net FC
        python3 training_testing.py 10 30 -p "Data/"
        python3 training_testing.py 10 30 -la 4 -ls 128 -d 0.5 -p "Data/"
        python3 training_testing.py 10 30 -adlr 1 0.0003 2 0.2 -p "TestTrashDir/"
        python3 training_testing.py 10 30 -eval 0 -p "TestTrashDir/"

    """
    import argparse
    import monoloco_net
    import networks

    parser = argparse.ArgumentParser()
    parser.add_argument("tables", type=int,
                        help="Number of tables to be used.")
    parser.add_argument("image_size", type=int,
                        help="Size of the images to be manipulated.")
    parser.add_argument("-train", "--training_set_share", type=float,
                        default=0.6,
                        help="Percentage of the data set used for training."+
                            "Defaults to 60 percents.")
    parser.add_argument("-test", "--testing_set_share", type=float,
                        default=0.3,
                        help="Percentage of the data set used for training."+
                            "Defaults to 30 percents.")
    parser.add_argument("-loss", "--loss_criterion", type=str,
                        default="CEL",
                        choices=["CEL"],
                        help="Loss criterion to be used for loss computation."+
                            "Possibles choices are "+
                            "CrossEntropyLoss (CEL)."+
                            "Defaults to CEL.")

    parser.add_argument("-lr","--learning_rate", type=float,
                        default="0.0005",
                        help="The initial learning rate for the network.")
    parser.add_argument("-wu", "--warmup_time", type=int,
                        default=0,
                        help="Warm up time for the network. Limits overfitting.")
    parser.add_argument("-adlr", "--adaptative_lr", nargs = '*',
                        default = [],
                        help="Learning rate evolution over epochs."+
                            "Come by pairs: "+
                            "the first element is the epoch index for change, "+
                            "the second element is the new learning rate.")

    parser.add_argument("-e","--epochs_amount", type=int,
                        default=200,
                        help="Number of epochs to be runned.")
    parser.add_argument("-mb","--minibatch_size", type=int,
                        default=50,
                        help="Size of the minibatches.")
    parser.add_argument("-lc", "--learning_curve", action="store_true",
                        default=False,
                        help="Displays the learning curve at the end of the run.")
    parser.add_argument("-b", "--break_accuracy", type=float,
                        default=0.99,
                        help="Training accuracy at which the training is halted.")
    parser.add_argument("-p", "--path", type=str,
                        default="",
                        help="The folder where the data is located.")
    parser.add_argument("-s","-start","--start_epoch", type = int,
                        default=0,
                        help="The starting epoch.")

    parser.add_argument("-t","--trials", type = int,
                        default=1,
                        help="Set the number of trials."+
                            "Defaults to 1.")
    parser.add_argument("-eval","--eval_frequency", type=int,
                        default=50,
                        help="Frequency in epochs for evaluation and saves."+
                            "Defaults to every 50 epochs."+
                            "Value 0 deactivates evaluation and saves.")


    parser.add_argument("-net","--network", type=str,
                        default="ML",
                        choices=["ML", "FC"],
                        help="Network category to be trained."+
                            "Possible choices are "+
                            "MonoLoco (monoloco). "+
                            "Defaults to monoloco")
    parser.add_argument("-ls","--layer_size", type=int,
                        default=256,
                        help="Size of the hidden fully-connected layers."+
                            "Defaults to 256")
    parser.add_argument("-la","--layers_amount", type=int,
                        default = 6,
                        help="The amount of hidden fully-connected layers"+
                            "Defaults to 6."+
                            "Note: for MonoLoco networks it has to be pair.")
    parser.add_argument("-d","--dropout", type=float,
                        default=0.2,
                        help="Dropout probability. Limits overfitting.")

    ### optional arguments
    args = parser.parse_args()

    tables_indices = np.arange(args.tables)
    training_set_size = int(args.tables*args.training_set_share)
    testing_set_size = int(args.tables*args.testing_set_share)
    validation_set_share=1-args.training_set_share-args.training_set_share
    validation_set_size = int(args.tables*validation_set_share)

    training_set_end = training_set_size
    testing_set_end = training_set_end + testing_set_size
    validation_set_end = testing_set_end + validation_set_size

    training_indices = tables_indices[:training_set_end]
    testing_indices = tables_indices[training_set_end:testing_set_end]
    validation_indices = tables_indices[testing_set_end: validation_set_end]


    # Creating adaptative learning rate dictionnary from parser arguments
    adaptative_lr = {}
    for i in range(int(len(args.adaptative_lr)/2)):
        index = int(args.adaptative_lr[2*i])
        learning_rate = float(args.adaptative_lr[2*i+1])
        adaptative_lr[index] = learning_rate

    #criterions = [torch.nn.MSELoss(), torch.nn.L1Loss(), torch.nn.CrossEntropyLoss()]

    accs = []
    # Running over several trials to limit randomness
    for i in range(args.trials):
        np.random.shuffle(tables_indices)

        training_indices = tables_indices[:training_set_end]
        testing_indices = tables_indices[training_set_end:testing_set_end]
        validation_indices = tables_indices[testing_set_end: validation_set_end]

        print("\rTrial {}".format(i))
        loss_criterion = (torch.nn.CrossEntropyLoss()
                            if args.loss_criterion == "CEL"
                else None)
        # trivia_net = networks.MultiTriviaNet(2, IMAGE_SIZE,
        #             hidden_layers_amount = 5, hidden_layer_size = 600)
        #trivia_net = networks.Net()
        network = (monoloco_net.LinearModel(args.image_size,
                                    channels_amount = 2,
                                    linear_size = args.layer_size,
                                    num_stage = int(args.layers_amount/2),
                                    p_dropout=args.dropout)
                            if args.network == "ML"
                else networks.MultiTriviaNet(2, args.image_size,
                                    hidden_layers_amount = args.layers_amount,
                                    hidden_layer_size = args.layer_size)
                            if args.network == "FC"
                else None)
        #trivia_net = monoloco_net.LinearModel(IMAGE_SIZE, linear_size=128, num_stage = 8)
        print('Num parameters: {}\t Num Trainable parameters: {}'.format(
            sum(p.numel() for p in network.parameters()),
            sum(p.numel() for p in network.parameters() if p.requires_grad)))

        optimizer = torch.optim.Adam(network.parameters(),
                                    lr = args.learning_rate)#changed form 0.0005
        acc = train_and_test_from_indices(network,
                                    training_indices, testing_indices,
                                    loss_criterion, optimizer,
                                    epochs_amount = args.epochs_amount,
                                    minibatch_size = args.minibatch_size,
                                    verbose = True,
                                    learning_curve = args.learning_curve,
                                    break_accuracy = args.break_accuracy,
                                    data_path = args.path,
                                    start_epoch = args.start_epoch,
                                    warmup_time = args.warmup_time,
                                    adaptative_lr = adaptative_lr,
                                    eval_frequency = args.eval_frequency)
        accs.append(acc)
    print("Average accuracy: {:.3f}".format(np.mean(accs)))
