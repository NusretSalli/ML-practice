import pandas as pd

import matplotlib.pyplot as plt

import torch

import numpy as np

import tqdm

from torch import nn

def train_binary_classification(model: torch.nn.Module,
                        X_train: torch.Tensor,
                        X_test: torch.Tensor,
                        y_train: torch.Tensor,
                        y_test: torch.Tensor,
                        n_max_iterations: int,
                        loss_function,
                        optimizer,
                        accuracy_function,
                        report_interval = 1000):

    """
    Trains the ANN given the training data (which is a binary classification) - it will then calculate the loss with the test

    Inputs:

    X_train -> your training dataset with the attributes
    X_test -> your test dataset with the attributes
    y_train -> Your training dataset that contains the class ID
    y_test -> Your test dataset that contains the class ID
    
    n_max_iterations -> maximum number of iterations that will be performed
    loss_function -> The loss function that will be used
    optimizer -> The optimizer that will be used
    accuracy_function -> the function that will be used to determine the accuracy
    class_type -> is it a binary- or multiclass training. 
    report_interval -> the interval in which a report will be printed out - default value is 1000

    
    Outputs: 
    
    - Report that show the current result / improvement
    
    - A trained model.
    
    
    """


    # build the training and evaluation loop

    for epoch in tqdm(range(n_max_iterations)):

        ### training section ###

        model.train()

        # do the forward pass

        y_logits = model(X_train).squeeze()

        y_pred = torch.round(torch.sigmoid(y_logits)) # turn logits -> pred probs -> pred labels

        # calculating loss / accuracy by using our function

        loss = loss_function(y_logits, y_train) # calculating the loss

        acc = accuracy_function(y_train, y_pred.squeeze()) # 1: target 2: prediction

        # optimizer zero grad

        optimizer.zero_grad()

        # loss backward

        loss.backward()

        # optimizer step (gradient descent)

        optimizer.step()


        ### TESTING SECTION ###

        model.eval()

        with torch.inference_mode():

            # forward pass

            test_logits = model(X_test).squeeze()

            test_pred = torch.round(torch.sigmoid(test_logits))

            # calculate the test loss / acc

            test_loss = loss_function(test_logits, y_test)

            test_acc = accuracy_function(y_test, test_pred)


        # print the information.

        if epoch % report_interval == 0:

            print(f"| Loss: {loss}, acc: {acc} | Test loss: {test_loss}, test acc: {test_acc}")




def train_multi_classification(model: torch.nn.Module,
                        X_train: torch.Tensor,
                        X_test: torch.Tensor,
                        y_train: torch.Tensor,
                        y_test: torch.Tensor,
                        n_max_iterations: int,
                        loss_function,
                        optimizer,
                        accuracy_function,
                        report_interval = 1000):
    
    """
    Trains the ANN given the training data (which is a multiclass classification) - it will then calculate the loss with the test

    Inputs:

    X_train -> your training dataset with the attributes
    X_test -> your test dataset with the attributes
    y_train -> Your training dataset that contains the class ID
    y_test -> Your test dataset that contains the class ID
    
    n_max_iterations -> maximum number of iterations that will be performed
    loss_function -> The loss function that will be used
    optimizer -> The optimizer that will be used
    accuracy_function -> the function that will be used to determine the accuracy
    class_type -> is it a binary- or multiclass training. 
    report_interval -> the interval in which a report will be printed out - default value is 1000

    
    Outputs: 
    
    - Report that show the current result / improvement
    
    - A trained model.
    
    
    """

    # build the training and evaluation loop

    for epoch in tqdm(range(n_max_iterations)):

        ### training section ###

        model.train()

        # do the forward pass

        y_logits = model(X_train).squeeze()

        y_pred_probs = torch.softmax(y_logits, dim = 1)

        y_pred = torch.argmax(y_pred_probs, dim = 1) # assign label to the most likely class

        # calculating loss / accuracy by using our function

        loss = loss_function(y_logits, y_train) # calculating the loss

        acc = accuracy_function(y_train, y_pred.squeeze()) # 1: target 2: prediction

        # optimizer zero grad

        optimizer.zero_grad()

        # loss backward

        loss.backward()

        # optimizer step (gradient descent)

        optimizer.step()


        ### TESTING SECTION ###

        model.eval()

        with torch.inference_mode():

            # forward pass

            test_logits = model(X_test).squeeze()

            test_probs = torch.softmax(test_logits, dim = 1)

            test_pred = torch.argmax(test_probs, dim = 1) # assign label to the most likely class

            # calculate the test loss / acc

            test_loss = loss_function(test_logits, y_test)

            test_acc = accuracy_function(y_test, test_pred)


        # print the information.

        if epoch % report_interval == 0:

            print(f"| Loss: {loss}, acc: {acc} | Test loss: {test_loss}, test acc: {test_acc}")