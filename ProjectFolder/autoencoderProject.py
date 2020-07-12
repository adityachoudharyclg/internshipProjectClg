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
        plt.plot(item)
        plt.legend("autoencoder {}".format(str(index)))
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
    if name not in 'karate':
        c_attributes = nx.get_node_attributes(G_data, 'value')
        c_groups = []
        for i, val in enumerate(c_attributes.values()):
            c_groups.append(val)
    else:
        c_groups=[0,0,0,0,0,0,0,0,1,1,0,0,0,0,1,1,0,0,1,0,1,0,1,1,1,1,1,1,1,1,1,1,1,1]

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
    if name not in 'karate':
        c_attributes = nx.get_node_attributes(G_data, 'value')
        c_groups = []
        for i, val in enumerate(c_attributes.values()):
            c_groups.append(val)
    else:
        c_groups=[0,0,0,0,0,0,0,0,1,1,0,0,0,0,1,1,0,0,1,0,1,0,1,1,1,1,1,1,1,1,1,1,1,1]     

    X_gt = np.array(c_groups)
    # print(X_ae)
    # print(X_gt)

    return metrics.normalized_mutual_info_score(X_gt, X_ae, average_method='arithmetic')


# Calculating max state
def calcMaxState(G_data, B_data, name, encoder):
    index = 0
    max_value = 0

    for r_state in range(0,1001):
        B_data_X = encoder.detach().numpy()

        kmeans = KMeans(n_clusters=get_clusters(G_data,name),init='k-means++',random_state=r_state)
        kmeans.fit(B_data_X)

        X_ae = kmeans.labels_ # Calculated labels

        # Finding truth values
        if name not in 'karate':
            c_attributes = nx.get_node_attributes(G_data, 'value')
            c_groups = []
            for i, val in enumerate(c_attributes.values()):
                c_groups.append(val)
        else:
            c_groups=[0,0,0,0,0,0,0,0,1,1,0,0,0,0,1,1,0,0,1,0,1,0,1,1,1,1,1,1,1,1,1,1,1,1]     

        X_gt = np.array(c_groups)
        nmi=metrics.normalized_mutual_info_score(X_gt, X_ae, average_method='arithmetic')

        if(nmi>max_value):
            index=r_state
            max_value=nmi

        if(r_state%100==0):
            print("Index:{}\tMax NMI till now:{}".format(index,max_value))
    
    return index