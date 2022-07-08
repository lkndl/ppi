import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class EmbeddingsProjection(nn.Module):
    """Module 1: Projection of embedding dimension"""

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


class MapProjection(nn.Module):
    """Module 2: Construct InteractionMap"""

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

        self.hidden = MapProjection(embed_dim, hidden_dim,
                                    activation=nn.ELU())
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


class ContactCNNDscript(nn.Module):
    def __init__(self, embed_dim=100, hidden_dim=50, width=7,
                 cnn_activation=nn.Sigmoid()):
        super(ContactCNNDscript, self).__init__()

        self.hidden = MapProjection(embed_dim, hidden_dim, activation=nn.ReLU())
        self.conv = nn.Conv2d(hidden_dim, 1, width, padding=width // 2)
        self.batchnorm = nn.BatchNorm2d(1)
        self.cnn_activation = cnn_activation
        self.clip()

    def clip(self):
        w = self.conv.weight
        self.conv.weight.data[:] = 0.5 * (w + w.transpose(2, 3))

    def forward(self, z0, z1):
        B = self.hidden(z0, z1)
        C = self.conv(B)
        C = self.batchnorm(C)
        C = self.cnn_activation(C)
        return C


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # change receptive field via stride + dilation
        # https://discuss.pytorch.org/t/how-to-create-convnet-for-variable-size-input-dimension-images/1906/2
        self.adp = nn.AdaptiveAvgPool2d((5, 5))  # for fixed-size, square (5, 5) output
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 5*5 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square, you can specify with a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = self.adp(x)
        x = torch.flatten(x, 1)  # flatten all dimensions except the batch dimension
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = nn.Sigmoid()(self.fc3(x))
        # x = torch.clamp(x, min=0, max=1)
        return x


class InteractionMap(nn.Module):
    """Module 3: Network combined and prediction"""

    def __init__(
            self,
            emb_projection_dim,
            dropout_p,
            map_hidden_dim,
            kernel_width,
            pool_size=9,
            gamma_init=0,
            activation=nn.GELU(),
            use_w=False,
            architecture='cnn',
            **kwargs
    ):
        super(InteractionMap, self).__init__()
        self.embedding = EmbeddingsProjection(1024, emb_projection_dim, dropout_p,
                                              activation=nn.ELU())
        self.contact = ContactCNN(emb_projection_dim, map_hidden_dim, kernel_width,
                                  cnn_activation=nn.Identity())

        self.maxPool = nn.MaxPool2d(pool_size, padding=pool_size // 2)
        self.gamma = nn.Parameter(torch.FloatTensor([gamma_init]))
        self.activation = activation
        self.cmap_cnn = Net()
        # self.cmap_fft = cosinus + sinus anteile verrechnen
        # phase value zurück abstrahieren = imaginäre einheit
        # fourier series bekommen
        self.predict_func = dict(cnn=self.cnn_predict, gelu=self.gelu_predict,
                                 fft=self.fft_predict)[architecture]

    def embed(self, z):
        return self.embedding(z)

    def cpred(self, z0, z1):
        e0 = self.embed(z0)
        e1 = self.embed(z1)
        C = self.contact(e0, e1)
        return C

    def map_predict(self, z0, z1):
        C = self.cpred(z0, z1)  # hier ist schon ein sigmoid drauf
        yhat = self.maxPool(C)

        # Mean of contact predictions where p_ij > mu + gamma*sigma
        mu = torch.mean(yhat)
        sigma = torch.var(yhat)
        Q = torch.relu(yhat - mu - (self.gamma * sigma))  # normalization would be *division* by std
        phat = torch.sum(Q) / (torch.sum(torch.sign(Q)) + 1)
        phat = self.activation(phat)
        return C, phat

    def predict(self, z0, z1):
        _, phat = self.map_predict(z0, z1)
        return phat

    def gelu_predict(self, z0, z1):
        C = self.cpred(z0, z1)
        # yhat = nn.GELU()(C)
        # yhat = torch.gelu(C)
        yhat = self.activation(C)
        phat = torch.mean(yhat)
        phat = torch.clamp(phat, min=0, max=1)
        return phat

    def cnn_predict(self, z0, z1):
        C = self.cpred(z0, z1)
        phat = self.cmap_cnn(C)
        return phat

    def fft_predict(self, z0, z1):
        C = self.cpred(z0, z1)
        phat = self.cmap_fft(C)
        return phat

    def forward(self, z0, z1):
        return self.predict(z0, z1)


class InteractionMapDscript(InteractionMap):
    """Module 3: Network combined and prediction"""

    def __init__(
            self,
            emb_projection_dim,
            dropout_p,
            map_hidden_dim,
            kernel_width,
            pool_size=9,
            gamma_init=0,
            activation=nn.ReLU(),
            use_w=True,
            theta_init=1,
            lambda_init=0,
    ):
        super(InteractionMapDscript, self).__init__(
            emb_projection_dim,
            dropout_p,
            map_hidden_dim,
            kernel_width,
            pool_size,
            gamma_init,
            activation)
        self.use_w = use_w
        self.contact = ContactCNNDscript(emb_projection_dim, map_hidden_dim, kernel_width)
        if self.use_w:
            self.theta = nn.Parameter(torch.FloatTensor([theta_init]))
            self.lambda_ = nn.Parameter(torch.FloatTensor([lambda_init]))
        self.activation = LogisticActivation(x0=0.5, k=20)
        self.clip()

    def clip(self):
        self.contact.clip()

        if self.use_w:
            self.theta.data.clamp_(min=0, max=1)
            self.lambda_.data.clamp_(min=0)

        self.gamma.data.clamp_(min=0)

    def map_predict(self, z0, z1):
        C = self.cpred(z0, z1)

        if self.use_w:
            # Create contact weighting matrix
            N, M = C.shape[2:]

            x1 = torch.from_numpy(-1 * ((np.arange(N) + 1 - ((N + 1) / 2))
                                        / (-1 * ((N + 1) / 2))) ** 2).float()
            if self.gamma.device.type == 'cuda':
                x1 = x1.cuda()
            x1 = torch.exp(self.lambda_ * x1)

            x2 = torch.from_numpy(-1 * ((np.arange(M) + 1 - ((M + 1) / 2))
                                        / (-1 * ((M + 1) / 2))) ** 2).float()
            if self.gamma.device.type == 'cuda':
                x2 = x2.cuda()
            x2 = torch.exp(self.lambda_ * x2)

            W = x1.unsqueeze(1) * x2
            W = (1 - self.theta) * W + self.theta

            yhat = C * W
        else:
            yhat = C

        yhat = self.maxPool(yhat)

        # Mean of contact predictions where p_ij > mu + gamma*sigma
        mu = torch.mean(yhat)
        sigma = torch.var(yhat)
        Q = torch.relu(yhat - mu - (self.gamma * sigma))
        phat = torch.sum(Q) / (torch.sum(torch.sign(Q)) + 1)
        phat = self.activation(phat)
        return C, phat


class LogisticActivation(nn.Module):
    def __init__(self, x0=0.5, k=1, train=False):
        super(LogisticActivation, self).__init__()
        self.x0 = x0
        self.k = nn.Parameter(torch.FloatTensor([float(k)]))
        self.k.requiresGrad = train

    def forward(self, x):
        out = torch.clamp(1 / (1 + torch.exp(-self.k * (x - self.x0))), min=0, max=1).squeeze()
        return out

    def clip(self):
        self.k.data.clamp_(min=0)
