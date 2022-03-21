"""
Interaction model classes.
"""

import numpy as np
import torch
import torch.nn as nn


########################### Module 1: Projection of embedding dimension ##########################
class EmbeddingsProjection(nn.Module):
    def __init__(self, nin=1024, nout=100, dropout=0.1, activation=nn.ELU()):
        super(EmbeddingsProjection, self).__init__()
        self.nin = nin
        self.nout = nout
        self.dropout_p = dropout

        self.transform = nn.Linear(nin, nout)
        self.drop = nn.Dropout(p=self.dropout_p)
        self.activation = activation

    def forward(self, x):
        t = self.transform(x)
        t = self.activation(t)
        t = self.drop(t)
        return t


########################### Module 2: Construct InteractionMap ##########################
class MapProjection(nn.Module):
    def __init__(self, embed_dim=100, hidden_dim=50, activation=nn.ELU()):
        super(MapProjection, self).__init__()

        self.D = embed_dim
        self.H = hidden_dim
        self.conv = nn.Conv2d(2 * self.D, self.H, 1)
        self.batchnorm = nn.BatchNorm2d(self.H)
        self.activation = activation

    def forward(self, z0, z1):
        # z0 is (b,N,d), z1 is (b,M,d)
        z0 = z0.transpose(1, 2)
        z1 = z1.transpose(1, 2)
        # z0 is (b,d,N), z1 is (b,d,M)

        z_dif = torch.abs(z0.unsqueeze(3) - z1.unsqueeze(2))
        z_mul = z0.unsqueeze(3) * z1.unsqueeze(2)
        z_cat = torch.cat([z_dif, z_mul], 1)

        b = self.conv(z_cat)
        b = self.activation(b)
        b = self.batchnorm(b)

        return b


class ContactCNN(nn.Module):
    def __init__(self, embed_dim=100, hidden_dim=50, width=7,
                 projection_activation=nn.ELU(),
                 cnn_activation=nn.Sigmoid()):
        super(ContactCNN, self).__init__()

        self.hidden = MapProjection(embed_dim, hidden_dim)
        self.proj_activation = projection_activation
        self.conv = nn.Conv2d(hidden_dim, 1, width, padding=width // 2)
        self.batchnorm = nn.BatchNorm2d(1)
        self.cnn_activation = cnn_activation

    def forward(self, z0, z1):
        B = self.hidden(z0, z1)
        B = self.proj_activation(B)
        C = self.conv(B)
        C = self.batchnorm(C)
        C = self.cnn_activation(C)
        return C


########################### Module 3: Network combined and prediction ##########################
class InteractionMap(nn.Module):
    def __init__(
        self,
        emb_projection_dim, dropout_p,
        map_hidden_dim, kernel_width,
        pool_size=9, gamma_init=0, activation=nn.Sigmoid()
    ):
        super(InteractionMap, self).__init__()
        self.embedding = EmbeddingsProjection(1024, emb_projection_dim, dropout_p)
        self.contact = ContactCNN(emb_projection_dim, map_hidden_dim, kernel_width)

        self.maxPool = nn.MaxPool2d(pool_size, padding=pool_size // 2)
        self.gamma = nn.Parameter(torch.FloatTensor([gamma_init]))
        self.activation = activation

    def embed(self, z):
        return self.embedding(z)

    def cpred(self, z0, z1):
        e0 = self.embed(z0)
        e1 = self.embed(z1)
        C = self.contact(e0, e1)
        return C

    def map_predict(self, z0, z1):
        C = self.cpred(z0, z1)
        yhat = self.maxPool(C)

        # Mean of contact predictions where p_ij > mu + gamma*sigma
        mu = torch.mean(yhat)
        sigma = torch.var(yhat)
        Q = torch.relu(yhat - mu - (self.gamma * sigma))
        phat = torch.sum(Q) / (torch.sum(torch.sign(Q)) + 1)
        phat = self.activation(phat)
        return C, phat

    def predict(self, z0, z1):
        _, phat = self.map_predict(z0, z1)
        return phat


    def forward(self, z0, z1):
        return self.predict(z0, z1)
