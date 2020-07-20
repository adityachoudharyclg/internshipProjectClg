# Imports
import matplotlib.pyplot as plt
import numpy as np
import networkx.generators as gen
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torch.nn.functional as F
import networkx as nx
from sklearn.cluster import KMeans
from sklearn import metrics
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from Datasets.lfrData import *

# Basic autoencoder model


class AE(nn.Module):
    def __init__(self, in_layers, hidden_layers):
        super().__init__()
        # Encoder
        self.encoder_hidden_layer = nn.Linear(
            in_features=in_layers, out_features=hidden_layers
        )
        self.encoder_output_layer = nn.Linear(
            in_features=hidden_layers, out_features=hidden_layers
        )
        # DEcoder
        self.decoder_hidden_layer = nn.Linear(
            in_features=hidden_layers, out_features=hidden_layers
        )
        self.decoder_output_layer = nn.Linear(
            in_features=hidden_layers, out_features=in_layers
        )

    def forward(self, features):
        activation = self.encoder_hidden_layer(features)
        activation = torch.tanh(activation)
        # activation = nn.Dropout(p=0.01)(activation)
        code = self.encoder_output_layer(activation)
        code = torch.tanh(code)
        # code = nn.Dropout(p=0.01)(code)
        encoder = code
        activation = self.decoder_hidden_layer(code)
        activation = torch.tanh(activation)
        # activation = nn.Dropout(p=0.01)(activation)
        activation = self.decoder_output_layer(activation)
        reconstructed = torch.tanh(activation)
        # reconstructed = nn.Dropout(p=0.01)(reconstructed)
        return encoder, reconstructed

# Load the dataset and return modularity matrix and truth values


def load_dataset(name):
    dolphin_path = 'Datasets/dolphins.gml'
    adjnoun_path = "Datasets/adjnoun.gml"
    football_path = 'Datasets/football.gml'
    polbooks_path = 'Datasets/polbooks.gml'
    email_edges = 'Datasets/email-Eu-core.txt'
    email_labels = 'Datasets/email-Eu-core-department-labels.txt'

    # # Polblogs
    # if(name=='polblogs'):
    #     G_data = nx.read_gml(polblogs_path)
    #     G_data = G_data.to_undirected()
    #     G_data = nx.Graph(G_data)
    #     B_data = nx.modularity_matrix(G_data)

    # Karate
    if(name == 'karate'):
        G_data = nx.karate_club_graph()
        B_data = nx.modularity_matrix(G_data)

    # Football
    elif(name == 'football'):
        G_data = nx.read_gml(football_path)
        B_data = nx.modularity_matrix(G_data)

    # Polbooks
    elif(name == 'polbooks'):
        G_data = nx.read_gml(polbooks_path)
        B_data = nx.modularity_matrix(G_data)

    # Dolphin
    elif(name == 'dolphin'):
        G_data = nx.read_gml(dolphin_path)
        B_data = nx.modularity_matrix(G_data)

    # lfr 0.1
    elif(name == 'lfr 0.1'):
        G_data = nx.Graph()
        data, labels = load_data(0.1)

        for index, item in enumerate(labels):
            G_data.add_node(index+1, value=item)
        for item in data:
            G_data.add_edge(*item)
        B_data = nx.modularity_matrix(G_data)

    # lfr 0.3
    elif(name == 'lfr 0.3'):
        G_data = nx.Graph()
        data, labels = load_data(0.3)

        for index, item in enumerate(labels):
            G_data.add_node(index+1, value=item)
        for item in data:
            G_data.add_edge(*item)
        B_data = nx.modularity_matrix(G_data)

    # lfr 0.5
    elif(name == 'lfr 0.5'):
        G_data = nx.Graph()
        data, labels = load_data(0.5)

        for index, item in enumerate(labels):
            G_data.add_node(index+1, value=item)
        for item in data:
            G_data.add_edge(*item)
        B_data = nx.modularity_matrix(G_data)

    elif(name == 'email'):
        G_data=nx.Graph()

        with open(email_edges) as f:
            for line in f:
                if(len(line)>0):
                    x=(line.split())
                    G_data.add_edge(int(x[0]),int(x[1]))

        G_data.remove_nodes_from([658,653,648,798,731,772,670,691,675,684,660,711,744,808,746,580,633,732,703])
        B_data = nx.modularity_matrix(G_data)

    return G_data, B_data

# Function to draw dataset


def drawDataset(G_data, name):
    nx.draw(G_data)
    plt.title(name)
    plt.show()

# Load to pytorch


def load_to_pytorch(inputs, targets):
    train_ds = TensorDataset(inputs, targets)
    train_dl = DataLoader(train_ds, batch_size=len(train_ds))
    return train_dl

# Function to train complete model


def train_model(epochs, train_dl, model, optimizer, fl=0):
    enc, out = 0, 0
    hist = []
    for epoch in range(epochs):
        loss = 0
        for batch_features, _ in train_dl:
          # reset the gradients back to zero
          # PyTorch accumulates gradients on subsequent backward passes
            # optimizer.zero_grad()
            optimizer.zero_grad()
            # compute reconstructions
            encoder, outputs = model(batch_features.float())
            # out=outputs
            if(fl == 1):
                outputs = outputs.double()
            # compute training reconstruction loss
            train_loss = loss_func(outputs, batch_features)

            # compute accumulated gradients
            train_loss.backward()

            # perform parameter update based on current gradients
            optimizer.step()
            enc = encoder
            # add the mini-batch training loss to epoch loss
            loss += train_loss.item()

            # enc=encoder
        # compute the epoch training loss
        loss = loss / len(train_dl)
        hist.append(loss)

        # display the epoch training loss
        print("epoch : {}/{}, recon loss = {:.8f}".format(epoch + 1, epochs, loss))
    return(enc, hist)

# mean-squared error loss


def loss_func(input, target):
    return F.mse_loss(input, target)

# Plot loss

def plot_loss(*hist):
    for index,item in enumerate(hist):
        plt.plot(item,label="autoencoder {}".format(str(index)))
    plt.legend()
    plt.show()


# main function to run model


def fit_dataset(B_data, hidden_layers, epoch_per_layer, learning_rate=1e-3):
    if(len(hidden_layers) == 3):
        # Creating required models
        model1 = AE(hidden_layers=hidden_layers[0], in_layers=B_data.shape[0])
        model2 = AE(hidden_layers=hidden_layers[1], in_layers=hidden_layers[0])
        model3 = AE(hidden_layers=hidden_layers[2], in_layers=hidden_layers[1])

        # create an optimizer object
        # Adam optimizer with learning rate 1e-3
        optimizer1 = optim.Adam(model1.parameters(), lr=learning_rate)
        optimizer2 = optim.Adam(model2.parameters(), lr=learning_rate)
        optimizer3 = optim.Adam(model3.parameters(), lr=learning_rate)

        # Loading dataset
        B_data = np.asarray(B_data, dtype=np.float64)
        inputs = torch.from_numpy(B_data)
        targets = torch.from_numpy(B_data)

        train_dl = load_to_pytorch(inputs, targets)

        # AE 1 training
        encoder, hist1 = train_model(
            epoch_per_layer[0], train_dl, model1, optimizer1, fl=1)
        train_dl = load_to_pytorch(encoder.detach(), encoder.detach())
        print(encoder.detach().shape)

        # AE 2 training
        encoder, hist2 = train_model(
            epoch_per_layer[1], train_dl, model2, optimizer2)
        train_dl = load_to_pytorch(encoder.detach(), encoder.detach())
        print(encoder.detach().shape)

        # AE 3 training
        encoder, hist3 = train_model(
            epoch_per_layer[2], train_dl, model3, optimizer3)
        train_dl = load_to_pytorch(encoder.detach(), encoder.detach())
        print(encoder.detach().shape)

        plot_loss(hist1,hist2,hist3)

        return encoder
    else:
        # Creating required models
        model1 = AE(hidden_layers=hidden_layers[0], in_layers=B_data.shape[0])
        model2 = AE(hidden_layers=hidden_layers[1], in_layers=hidden_layers[0])

        # create an optimizer object
        # Adam optimizer with learning rate 1e-3
        optimizer1 = optim.Adam(model1.parameters(), lr=learning_rate)
        optimizer2 = optim.Adam(model2.parameters(), lr=learning_rate)

        # Loading dataset
        B_data = np.asarray(B_data, dtype=np.float64)
        inputs = torch.from_numpy(B_data)
        targets = torch.from_numpy(B_data)

        train_dl = load_to_pytorch(inputs, targets)

        # AE 1 training
        encoder, hist1 = train_model(
            epoch_per_layer[0], train_dl, model1, optimizer1, fl=1)
        train_dl = load_to_pytorch(encoder.detach(), encoder.detach())
        print(encoder.detach().shape)

        # AE 2 training
        encoder, hist2 = train_model(
            epoch_per_layer[1], train_dl, model2, optimizer2)
        train_dl = load_to_pytorch(encoder.detach(), encoder.detach())
        print(encoder.detach().shape)

        # plt.plot(hist1,label='encoder 1')
        # plt.plot(hist2,label='encoder 2')
        # plt.legend()
        # plt.show()
        plot_loss(hist1,hist2)

        return encoder

# Calculating number of clusters


def get_clusters(G_data,name):
    if name =='karate':
        c_groups=[0,0,0,0,0,0,0,0,1,1,0,0,0,0,1,1,0,0,1,0,1,0,1,1,1,1,1,1,1,1,1,1,1,1]
    elif name == 'email':
        c_groups=[1, 1, 21, 21, 21, 25, 25, 14, 14, 14, 9, 14, 14, 26, 4, 17, 34, 1, 1, 14, 9, 9, 9, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 5, 34, 14, 14, 17, 17, 10, 10, 36, 37, 5, 7, 4, 22, 22, 21, 21, 21, 21, 7, 7, 36, 21, 25, 4, 8, 15, 15, 15, 37, 37, 9, 1, 1, 10, 10, 3, 3, 3, 29, 15, 36, 36, 37, 1, 36, 34, 20, 20, 8, 15, 9, 4, 5, 4, 20, 16, 16, 16, 16, 16, 38, 7, 7, 34, 38, 36, 8, 27, 8, 8, 8, 10, 10, 13, 13, 6, 26, 10, 1, 36, 0, 13, 16, 16, 22, 6, 5, 4, 0, 28, 28, 4, 2, 13, 13, 21, 21, 17, 17, 14, 36, 8, 40, 35, 15, 23, 0, 0, 7, 10, 37, 27, 35, 35, 0, 0, 19, 19, 36, 14, 37, 24, 17, 13, 36, 4, 4, 13, 13, 10, 4, 38, 32, 32, 4, 1, 0, 0, 0, 7, 7, 4, 15, 16, 40, 15, 15, 15, 15, 0, 21, 21, 21, 21, 5, 4, 4, 4, 4, 4, 4, 4, 5, 5, 4, 4, 22, 19, 19, 22, 34, 14, 0, 1, 17, 37, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 10, 23, 0, 4, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 10, 14, 14, 1, 14, 7, 13, 20, 31, 40, 6, 4, 0, 8, 9, 9, 10, 0, 10, 14, 14, 14, 14, 39, 17, 4, 28, 17, 17, 17, 4, 4, 0, 0, 23, 4, 21, 36, 36, 0, 22, 21, 15, 37, 0, 4, 4, 4, 14, 4, 7, 7, 1, 15, 15, 38, 26, 20, 20, 20, 21, 9, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 10, 19, 7, 7, 17, 16, 14, 9, 9, 9, 8, 8, 13, 39, 14, 10, 17, 17, 13, 13, 13, 13, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16, 16, 27, 8, 8, 14, 14, 14, 10, 14, 35, 37, 14, 36, 10, 7, 20, 10, 16, 36, 36, 14, 8, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 4, 9, 4, 0, 4, 16, 38, 14, 14, 21, 26, 27, 28, 21, 4, 1, 1, 9, 10, 15, 4, 26, 14, 35, 10, 34, 4, 4, 12, 17, 17, 14, 37, 37, 37, 34, 6, 13, 13, 13, 13, 4, 14, 10, 10, 10, 3, 17, 17, 17, 1, 4, 14, 14, 6, 27, 22, 21, 4, 4, 1, 34, 17, 30, 30, 4, 23, 14, 15, 1, 22, 12, 31, 6, 15, 15, 8, 15, 8, 8, 1, 15, 22, 2, 3, 4, 10, 4, 14, 14, 25, 6, 6, 40, 4, 36, 23, 14, 3, 14, 14, 14, 14, 14, 14, 14, 14, 14, 31, 15, 15, 14, 0, 23, 35, 8, 4, 1, 1, 35, 23, 21, 2, 4, 4, 9, 14, 4, 10, 25, 14, 14, 3, 21, 35, 4, 9, 15, 6, 9, 3, 15, 23, 4, 4, 4, 11, 35, 10, 6, 15, 15, 15, 22, 2, 2, 14, 4, 3, 14, 27, 31, 34, 4, 4, 19, 14, 14, 4, 4, 14, 14, 21, 4, 14, 4, 0, 4, 27, 27, 17, 3, 15, 2, 4, 4, 21, 21, 11, 23, 11, 23, 17, 5, 36, 15, 23, 23, 2, 19, 4, 36, 14, 1, 22, 1, 21, 34, 14, 13, 6, 4, 37, 6, 24, 35, 6, 17, 16, 6, 4, 0, 21, 4, 26, 21, 4, 15, 7, 1, 20, 19, 7, 21, 21, 21, 19, 38, 19, 16, 23, 6, 37, 25, 1, 22, 6, 14, 1, 26, 8, 37, 4, 0, 17, 6, 17, 14, 16, 4, 32, 14, 15, 0, 23, 21, 29, 14, 14, 1, 17, 26, 15, 0, 0, 0, 22, 34, 21, 6, 16, 4, 15, 21, 0, 36, 4, 1, 1, 22, 14, 14, 30, 4, 9, 10, 4, 4, 14, 16, 16, 15, 21, 0, 4, 15, 29, 24, 21, 14, 11, 11, 9, 13, 10, 31, 4, 22, 14, 23, 1, 4, 9, 17, 27, 28, 22, 14, 20, 7, 23, 1, 6, 15, 15, 23, 4, 20, 5, 36, 10, 21, 39, 41, 31, 17, 7, 21, 34, 1, 14, 2, 18, 16, 27, 16, 38, 7, 38, 21, 1, 9, 15, 15, 15, 0, 6, 23, 28, 11, 23, 34, 24, 4, 4, 4, 24, 23, 17, 10, 17, 1, 1, 15, 15, 4, 21, 14, 14, 20, 28, 20, 22, 26, 3, 32, 4, 0, 21, 13, 4, 15, 17, 5, 4, 14, 0, 9, 21, 14, 38, 4, 14, 31, 21, 14, 6, 4, 4, 6, 17, 0, 4, 7, 16, 4, 4, 21, 1, 10, 3, 21, 4, 0, 1, 7, 17, 15, 14, 0, 9, 32, 13, 5, 2, 21, 28, 21, 22, 22, 7, 7, 33, 0, 1, 15, 4, 31, 30, 15, 11, 19, 21, 9, 21, 13, 21, 9, 32, 9, 32, 38, 9, 38, 38, 14, 9, 10, 38, 10, 22, 21, 13, 21, 4, 0, 1, 1, 23, 0, 5, 4, 4, 15, 14, 14, 13, 11, 1, 5, 5, 10, 23, 21, 14, 9, 20, 10, 19, 19, 21, 17, 19, 19, 36, 17, 35, 16, 4, 16, 4, 6, 4, 41, 6, 7, 23, 9, 23, 7, 6, 22, 36, 14, 15, 11, 35, 5, 14, 14, 15, 4, 6, 4, 9, 19, 11, 4, 29, 14, 15, 15, 5, 32, 15, 14, 5, 9, 10, 19, 13, 23, 12, 10, 21, 10, 35, 7, 22, 22, 22, 8, 21, 32, 4, 21, 21, 6, 14, 11, 14, 15, 4, 21, 1, 6, 22]
    else:
        c_attributes = nx.get_node_attributes(G_data, 'value')
        c_groups = []
        for i, val in enumerate(c_attributes.values()):
            c_groups.append(val)

    return len(set(c_groups))


def compute_results(G_data, B_data, name, encoder,r_state=0,only_kmeans=False):
    B_data_X = encoder.detach().numpy()

    kmeans = KMeans(n_clusters=get_clusters(G_data,name),init='k-means++',random_state=r_state)

    if not only_kmeans:
        kmeans.fit(B_data_X)
    else: 
        kmeans.fit(B_data)

    X_ae = kmeans.labels_ # Calculated labels

    # Finding truth values
    if name =='karate':
        c_groups=[0,0,0,0,0,0,0,0,1,1,0,0,0,0,1,1,0,0,1,0,1,0,1,1,1,1,1,1,1,1,1,1,1,1]
    elif name == 'email':
        c_groups=[1, 1, 21, 21, 21, 25, 25, 14, 14, 14, 9, 14, 14, 26, 4, 17, 34, 1, 1, 14, 9, 9, 9, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 5, 34, 14, 14, 17, 17, 10, 10, 36, 37, 5, 7, 4, 22, 22, 21, 21, 21, 21, 7, 7, 36, 21, 25, 4, 8, 15, 15, 15, 37, 37, 9, 1, 1, 10, 10, 3, 3, 3, 29, 15, 36, 36, 37, 1, 36, 34, 20, 20, 8, 15, 9, 4, 5, 4, 20, 16, 16, 16, 16, 16, 38, 7, 7, 34, 38, 36, 8, 27, 8, 8, 8, 10, 10, 13, 13, 6, 26, 10, 1, 36, 0, 13, 16, 16, 22, 6, 5, 4, 0, 28, 28, 4, 2, 13, 13, 21, 21, 17, 17, 14, 36, 8, 40, 35, 15, 23, 0, 0, 7, 10, 37, 27, 35, 35, 0, 0, 19, 19, 36, 14, 37, 24, 17, 13, 36, 4, 4, 13, 13, 10, 4, 38, 32, 32, 4, 1, 0, 0, 0, 7, 7, 4, 15, 16, 40, 15, 15, 15, 15, 0, 21, 21, 21, 21, 5, 4, 4, 4, 4, 4, 4, 4, 5, 5, 4, 4, 22, 19, 19, 22, 34, 14, 0, 1, 17, 37, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 10, 23, 0, 4, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 10, 14, 14, 1, 14, 7, 13, 20, 31, 40, 6, 4, 0, 8, 9, 9, 10, 0, 10, 14, 14, 14, 14, 39, 17, 4, 28, 17, 17, 17, 4, 4, 0, 0, 23, 4, 21, 36, 36, 0, 22, 21, 15, 37, 0, 4, 4, 4, 14, 4, 7, 7, 1, 15, 15, 38, 26, 20, 20, 20, 21, 9, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 10, 19, 7, 7, 17, 16, 14, 9, 9, 9, 8, 8, 13, 39, 14, 10, 17, 17, 13, 13, 13, 13, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16, 16, 27, 8, 8, 14, 14, 14, 10, 14, 35, 37, 14, 36, 10, 7, 20, 10, 16, 36, 36, 14, 8, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 4, 9, 4, 0, 4, 16, 38, 14, 14, 21, 26, 27, 28, 21, 4, 1, 1, 9, 10, 15, 4, 26, 14, 35, 10, 34, 4, 4, 12, 17, 17, 14, 37, 37, 37, 34, 6, 13, 13, 13, 13, 4, 14, 10, 10, 10, 3, 17, 17, 17, 1, 4, 14, 14, 6, 27, 22, 21, 4, 4, 1, 34, 17, 30, 30, 4, 23, 14, 15, 1, 22, 12, 31, 6, 15, 15, 8, 15, 8, 8, 1, 15, 22, 2, 3, 4, 10, 4, 14, 14, 25, 6, 6, 40, 4, 36, 23, 14, 3, 14, 14, 14, 14, 14, 14, 14, 14, 14, 31, 15, 15, 14, 0, 23, 35, 8, 4, 1, 1, 35, 23, 21, 2, 4, 4, 9, 14, 4, 10, 25, 14, 14, 3, 21, 35, 4, 9, 15, 6, 9, 3, 15, 23, 4, 4, 4, 11, 35, 10, 6, 15, 15, 15, 22, 2, 2, 14, 4, 3, 14, 27, 31, 34, 4, 4, 19, 14, 14, 4, 4, 14, 14, 21, 4, 14, 4, 0, 4, 27, 27, 17, 3, 15, 2, 4, 4, 21, 21, 11, 23, 11, 23, 17, 5, 36, 15, 23, 23, 2, 19, 4, 36, 14, 1, 22, 1, 21, 34, 14, 13, 6, 4, 37, 6, 24, 35, 6, 17, 16, 6, 4, 0, 21, 4, 26, 21, 4, 15, 7, 1, 20, 19, 7, 21, 21, 21, 19, 38, 19, 16, 23, 6, 37, 25, 1, 22, 6, 14, 1, 26, 8, 37, 4, 0, 17, 6, 17, 14, 16, 4, 32, 14, 15, 0, 23, 21, 29, 14, 14, 1, 17, 26, 15, 0, 0, 0, 22, 34, 21, 6, 16, 4, 15, 21, 0, 36, 4, 1, 1, 22, 14, 14, 30, 4, 9, 10, 4, 4, 14, 16, 16, 15, 21, 0, 4, 15, 29, 24, 21, 14, 11, 11, 9, 13, 10, 31, 4, 22, 14, 23, 1, 4, 9, 17, 27, 28, 22, 14, 20, 7, 23, 1, 6, 15, 15, 23, 4, 20, 5, 36, 10, 21, 39, 41, 31, 17, 7, 21, 34, 1, 14, 2, 18, 16, 27, 16, 38, 7, 38, 21, 1, 9, 15, 15, 15, 0, 6, 23, 28, 11, 23, 34, 24, 4, 4, 4, 24, 23, 17, 10, 17, 1, 1, 15, 15, 4, 21, 14, 14, 20, 28, 20, 22, 26, 3, 32, 4, 0, 21, 13, 4, 15, 17, 5, 4, 14, 0, 9, 21, 14, 38, 4, 14, 31, 21, 14, 6, 4, 4, 6, 17, 0, 4, 7, 16, 4, 4, 21, 1, 10, 3, 21, 4, 0, 1, 7, 17, 15, 14, 0, 9, 32, 13, 5, 2, 21, 28, 21, 22, 22, 7, 7, 33, 0, 1, 15, 4, 31, 30, 15, 11, 19, 21, 9, 21, 13, 21, 9, 32, 9, 32, 38, 9, 38, 38, 14, 9, 10, 38, 10, 22, 21, 13, 21, 4, 0, 1, 1, 23, 0, 5, 4, 4, 15, 14, 14, 13, 11, 1, 5, 5, 10, 23, 21, 14, 9, 20, 10, 19, 19, 21, 17, 19, 19, 36, 17, 35, 16, 4, 16, 4, 6, 4, 41, 6, 7, 23, 9, 23, 7, 6, 22, 36, 14, 15, 11, 35, 5, 14, 14, 15, 4, 6, 4, 9, 19, 11, 4, 29, 14, 15, 15, 5, 32, 15, 14, 5, 9, 10, 19, 13, 23, 12, 10, 21, 10, 35, 7, 22, 22, 22, 8, 21, 32, 4, 21, 21, 6, 14, 11, 14, 15, 4, 21, 1, 6, 22]
    else:
        c_attributes = nx.get_node_attributes(G_data, 'value')
        c_groups = []
        for i, val in enumerate(c_attributes.values()):
            c_groups.append(val)

    X_gt = np.array(c_groups)
    # print(X_ae)
    # print(X_gt)

    return metrics.fowlkes_mallows_score(X_gt, X_ae, average_method='arithmetic')


# Calculating max state
def calcMaxState(G_data, B_data, name, encoder):
    index = 0
    max_value = 0

    if name not in 'email':
        iterations=1001
    else:
        iterations=300

    for r_state in range(0,iterations):
        B_data_X = encoder.detach().numpy()

        kmeans = KMeans(n_clusters=get_clusters(G_data,name),init='k-means++',random_state=r_state)
        kmeans.fit(B_data_X)

        X_ae = kmeans.labels_ # Calculated labels

        # Finding truth values
        if name =='karate':
            c_groups=[0,0,0,0,0,0,0,0,1,1,0,0,0,0,1,1,0,0,1,0,1,0,1,1,1,1,1,1,1,1,1,1,1,1]
        elif name == 'email':
            c_groups=[1, 1, 21, 21, 21, 25, 25, 14, 14, 14, 9, 14, 14, 26, 4, 17, 34, 1, 1, 14, 9, 9, 9, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 5, 34, 14, 14, 17, 17, 10, 10, 36, 37, 5, 7, 4, 22, 22, 21, 21, 21, 21, 7, 7, 36, 21, 25, 4, 8, 15, 15, 15, 37, 37, 9, 1, 1, 10, 10, 3, 3, 3, 29, 15, 36, 36, 37, 1, 36, 34, 20, 20, 8, 15, 9, 4, 5, 4, 20, 16, 16, 16, 16, 16, 38, 7, 7, 34, 38, 36, 8, 27, 8, 8, 8, 10, 10, 13, 13, 6, 26, 10, 1, 36, 0, 13, 16, 16, 22, 6, 5, 4, 0, 28, 28, 4, 2, 13, 13, 21, 21, 17, 17, 14, 36, 8, 40, 35, 15, 23, 0, 0, 7, 10, 37, 27, 35, 35, 0, 0, 19, 19, 36, 14, 37, 24, 17, 13, 36, 4, 4, 13, 13, 10, 4, 38, 32, 32, 4, 1, 0, 0, 0, 7, 7, 4, 15, 16, 40, 15, 15, 15, 15, 0, 21, 21, 21, 21, 5, 4, 4, 4, 4, 4, 4, 4, 5, 5, 4, 4, 22, 19, 19, 22, 34, 14, 0, 1, 17, 37, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 10, 23, 0, 4, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 10, 14, 14, 1, 14, 7, 13, 20, 31, 40, 6, 4, 0, 8, 9, 9, 10, 0, 10, 14, 14, 14, 14, 39, 17, 4, 28, 17, 17, 17, 4, 4, 0, 0, 23, 4, 21, 36, 36, 0, 22, 21, 15, 37, 0, 4, 4, 4, 14, 4, 7, 7, 1, 15, 15, 38, 26, 20, 20, 20, 21, 9, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 10, 19, 7, 7, 17, 16, 14, 9, 9, 9, 8, 8, 13, 39, 14, 10, 17, 17, 13, 13, 13, 13, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16, 16, 27, 8, 8, 14, 14, 14, 10, 14, 35, 37, 14, 36, 10, 7, 20, 10, 16, 36, 36, 14, 8, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 4, 9, 4, 0, 4, 16, 38, 14, 14, 21, 26, 27, 28, 21, 4, 1, 1, 9, 10, 15, 4, 26, 14, 35, 10, 34, 4, 4, 12, 17, 17, 14, 37, 37, 37, 34, 6, 13, 13, 13, 13, 4, 14, 10, 10, 10, 3, 17, 17, 17, 1, 4, 14, 14, 6, 27, 22, 21, 4, 4, 1, 34, 17, 30, 30, 4, 23, 14, 15, 1, 22, 12, 31, 6, 15, 15, 8, 15, 8, 8, 1, 15, 22, 2, 3, 4, 10, 4, 14, 14, 25, 6, 6, 40, 4, 36, 23, 14, 3, 14, 14, 14, 14, 14, 14, 14, 14, 14, 31, 15, 15, 14, 0, 23, 35, 8, 4, 1, 1, 35, 23, 21, 2, 4, 4, 9, 14, 4, 10, 25, 14, 14, 3, 21, 35, 4, 9, 15, 6, 9, 3, 15, 23, 4, 4, 4, 11, 35, 10, 6, 15, 15, 15, 22, 2, 2, 14, 4, 3, 14, 27, 31, 34, 4, 4, 19, 14, 14, 4, 4, 14, 14, 21, 4, 14, 4, 0, 4, 27, 27, 17, 3, 15, 2, 4, 4, 21, 21, 11, 23, 11, 23, 17, 5, 36, 15, 23, 23, 2, 19, 4, 36, 14, 1, 22, 1, 21, 34, 14, 13, 6, 4, 37, 6, 24, 35, 6, 17, 16, 6, 4, 0, 21, 4, 26, 21, 4, 15, 7, 1, 20, 19, 7, 21, 21, 21, 19, 38, 19, 16, 23, 6, 37, 25, 1, 22, 6, 14, 1, 26, 8, 37, 4, 0, 17, 6, 17, 14, 16, 4, 32, 14, 15, 0, 23, 21, 29, 14, 14, 1, 17, 26, 15, 0, 0, 0, 22, 34, 21, 6, 16, 4, 15, 21, 0, 36, 4, 1, 1, 22, 14, 14, 30, 4, 9, 10, 4, 4, 14, 16, 16, 15, 21, 0, 4, 15, 29, 24, 21, 14, 11, 11, 9, 13, 10, 31, 4, 22, 14, 23, 1, 4, 9, 17, 27, 28, 22, 14, 20, 7, 23, 1, 6, 15, 15, 23, 4, 20, 5, 36, 10, 21, 39, 41, 31, 17, 7, 21, 34, 1, 14, 2, 18, 16, 27, 16, 38, 7, 38, 21, 1, 9, 15, 15, 15, 0, 6, 23, 28, 11, 23, 34, 24, 4, 4, 4, 24, 23, 17, 10, 17, 1, 1, 15, 15, 4, 21, 14, 14, 20, 28, 20, 22, 26, 3, 32, 4, 0, 21, 13, 4, 15, 17, 5, 4, 14, 0, 9, 21, 14, 38, 4, 14, 31, 21, 14, 6, 4, 4, 6, 17, 0, 4, 7, 16, 4, 4, 21, 1, 10, 3, 21, 4, 0, 1, 7, 17, 15, 14, 0, 9, 32, 13, 5, 2, 21, 28, 21, 22, 22, 7, 7, 33, 0, 1, 15, 4, 31, 30, 15, 11, 19, 21, 9, 21, 13, 21, 9, 32, 9, 32, 38, 9, 38, 38, 14, 9, 10, 38, 10, 22, 21, 13, 21, 4, 0, 1, 1, 23, 0, 5, 4, 4, 15, 14, 14, 13, 11, 1, 5, 5, 10, 23, 21, 14, 9, 20, 10, 19, 19, 21, 17, 19, 19, 36, 17, 35, 16, 4, 16, 4, 6, 4, 41, 6, 7, 23, 9, 23, 7, 6, 22, 36, 14, 15, 11, 35, 5, 14, 14, 15, 4, 6, 4, 9, 19, 11, 4, 29, 14, 15, 15, 5, 32, 15, 14, 5, 9, 10, 19, 13, 23, 12, 10, 21, 10, 35, 7, 22, 22, 22, 8, 21, 32, 4, 21, 21, 6, 14, 11, 14, 15, 4, 21, 1, 6, 22]
        else:
            c_attributes = nx.get_node_attributes(G_data, 'value')
            c_groups = []
            for i, val in enumerate(c_attributes.values()):
                c_groups.append(val)  

        X_gt = np.array(c_groups)
        fmi=metrics.fowlkes_mallows_score(X_gt, X_ae, average_method='arithmetic')

        if(fmi>max_value):
            index=r_state
            max_value=fmi

        if(r_state%100==0):
            print("Index:{}\tMax fmi till now:{}".format(index,max_value))
    
    return index

def show_clustering(G_data, B_data, name, encoder, r_state):
    
    # if(name=='lfr 0.1'):

    #     B_data_X = encoder.detach().numpy()
    #     kmeans = KMeans(init='k-means++',n_clusters=4,random_state=r_state)
    #     kmeans.fit(B_data_X)
    #     X_ae = kmeans.labels_
    #     labels_dict={0:[],1:[],2:[],3:[]}

    #     for index,item in enumerate(X_ae):
    #         labels_dict[item].append(index+1)

    #     G=G_data
    #     pos=nx.spring_layout(G)
    #     # print(pos)
    #     nx.draw_networkx_nodes(G,pos,nodelist=labels_dict[0],node_color='r',node_size=75)
    #     nx.draw_networkx_nodes(G,pos,nodelist=labels_dict[1],node_color='g',node_size=75)
    #     nx.draw_networkx_nodes(G,pos,nodelist=labels_dict[2],node_color='b',node_size=75)
    #     nx.draw_networkx_nodes(G,pos,nodelist=labels_dict[3],node_color='y',node_size=75)

    #     plt.title('lfr 0.1')
    #     plt.show()

    # elif(name=='lfr 0.3'):

    #     B_data_X = encoder.detach().numpy()
    #     kmeans = KMeans(init='k-means++',n_clusters=4,random_state=r_state)
    #     kmeans.fit(B_data_X)
    #     X_ae = kmeans.labels_
    #     labels_dict={0:[],1:[],2:[],3:[]}

    #     for index,item in enumerate(X_ae):
    #         labels_dict[item].append(index+1)

    #     G=G_data
    #     pos=nx.spring_layout(G)
    #     # print(pos)
    #     nx.draw_networkx_nodes(G,pos,nodelist=labels_dict[0],node_color='r',node_size=75)
    #     nx.draw_networkx_nodes(G,pos,nodelist=labels_dict[1],node_color='g',node_size=75)
    #     nx.draw_networkx_nodes(G,pos,nodelist=labels_dict[2],node_color='b',node_size=75)
    #     nx.draw_networkx_nodes(G,pos,nodelist=labels_dict[3],node_color='y',node_size=75)

    #     plt.title('lfr 0.3')
    #     plt.show()
    
    # elif(name=='lfr 0.5'):

    #     B_data_X = encoder.detach().numpy()
    #     kmeans = KMeans(init='k-means++',n_clusters=4,random_state=r_state)
    #     kmeans.fit(B_data_X)
    #     X_ae = kmeans.labels_
    #     labels_dict={0:[],1:[],2:[],3:[]}

    #     for index,item in enumerate(X_ae):
    #         labels_dict[item].append(index+1)

    #     G=G_data
    #     pos=nx.spring_layout(G)
    #     # print(pos)
    #     nx.draw_networkx_nodes(G,pos,nodelist=labels_dict[0],node_color='r',node_size=75)
    #     nx.draw_networkx_nodes(G,pos,nodelist=labels_dict[1],node_color='g',node_size=75)
    #     nx.draw_networkx_nodes(G,pos,nodelist=labels_dict[2],node_color='b',node_size=75)
    #     nx.draw_networkx_nodes(G,pos,nodelist=labels_dict[3],node_color='y',node_size=75)

    #     plt.title('lfr 0.5')
    #     plt.show()
    
    # else:
    G=G_data
    B_data_X = encoder.detach().numpy()

    kmeans = KMeans(init='k-means++',n_clusters=get_clusters(G_data,name),random_state=r_state)
    kmeans.fit(B_data_X)
    X_ae = kmeans.labels_

    labels_dict={}
    for index,item in enumerate(X_ae):
        labels_dict[item]=[]

    for index,item in enumerate(X_ae):
        labels_dict[item].append(list(G.nodes)[index])

    G=G_data
    pos=nx.spring_layout(G)
    plt.figure(figsize=(20,10))
    colors=['#f20905','#ab7533','#de9c0d','#a5e841','#09ed7b','#0af0e0','#0f5ea8','#08046e','#9c2be3','#e607c4','#e607c4','#0a0105']
    
    for key,col in enumerate(labels_dict):
        rgb = np.random.rand(3,)
        nx.draw(G,pos,nodelist=labels_dict[key],node_color=[rgb],node_size=150,width=0.10,)

    plt.title(name)
    plt.show()