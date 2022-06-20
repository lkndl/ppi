import torch
from torch import nn


class Linear(nn.Module):
    def __init__(self, embed_dim, out_dim=1):
        super(Linear, self).__init__()

        self.linear = nn.Linear(2 * embed_dim, out_dim)

    def forward(self, emb_prot1, emb_prot2):
        """
        Shape:
            - emb_prot1: (batch_size, 1, emb_dim)
            - emb_prot2: (batch_size, 1, emb_dim)
        """
        concat = torch.cat((emb_prot1, emb_prot2), 1)
        concat = concat.view(concat.shape[0], -1)
        return self.linear(concat)

    def predict(self, emb_prot1, emb_prot2):
        logit = self.forward(emb_prot1, emb_prot2)
        return nn.Sigmoid()(logit)

    def get_num_parameters(self):
        return sum([p.numel() for p in self.parameters() if p.requires_grad])


class MLP(nn.Module):
    def __init__(self, embed_dim, projection_dim, hidden, out_dim=1):
        super(MLP, self).__init__()

        # (batch, 1, emb_dim) ---> (batch, 1, proj)
        self.proj_layer = nn.Linear(embed_dim, projection_dim)
        self.proj_activation = nn.GELU()
        # (batch, 2, proj) ---> (batch, 2, hidden)
        self.hidden_layer = nn.Linear(projection_dim, hidden)
        self.hidden_activation = nn.GELU()
        # (batch, 2, proj) ---> (batch, 2)
        self.linear = nn.Linear(2 * hidden, out_dim)

    def forward(self, emb_prot1, emb_prot2):
        """
        Shape:
            - emb_prot1: (batch_size, 1, emb_dim)
            - emb_prot2: (batch_size, 1, emb_dim)
        """
        proj_emb_prot1, proj_emb_prot2 = self.proj_layer(emb_prot1), self.proj_layer(emb_prot2)
        proj_emb_prot1, proj_emb_prot2 = self.proj_activation(proj_emb_prot1), self.proj_activation(proj_emb_prot2)
        concat = torch.cat((proj_emb_prot1, proj_emb_prot2), 1)
        x = self.hidden_layer(concat)
        x = self.hidden_activation(x)
        x = x.view(x.shape[0], -1)
        return self.linear(x)

    def predict(self, emb_prot1, emb_prot2):
        logit = self.forward(emb_prot1, emb_prot2)
        return nn.Sigmoid()(logit)

    def get_num_parameters(self):
        return sum([p.numel() for p in self.parameters() if p.requires_grad])


class ProjectedAttention(nn.Module):
    def __init__(self,
                 embed_dim,
                 projection_dim,
                 num_heads=8,
                 num_layers=2,
                 dim_feedforward=128,
                 out_dim=1,
                 activation=nn.GELU()):
        super(ProjectedAttention, self).__init__()

        # (batch, 1, emb_dim) ---> (batch, 1, proj)
        self.proj_layer = nn.Linear(embed_dim, projection_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=projection_dim, nhead=num_heads,
                                                   dim_feedforward=dim_feedforward, activation=activation,
                                                   batch_first=True)
        # (batch, 2, proj) ---> (batch, 2, proj)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # (batch, 2, proj) ---> (batch, 2)
        self.linear = nn.Linear(2 * projection_dim, out_dim)

    def forward(self, emb_prot1, emb_prot2):
        """
        Shape:
            - emb_prot1: (batch_size, 1, emb_dim)
            - emb_prot2: (batch_size, 1, emb_dim)
        """
        proj_emb_prot1, proj_emb_prot2 = self.proj_layer(emb_prot1), self.proj_layer(emb_prot2)
        concat = torch.cat((proj_emb_prot1, proj_emb_prot2), 1)
        x = self.transformer_encoder(concat)
        x = x.view(x.shape[0], -1)
        return self.linear(x)

    def predict(self, emb_prot1, emb_prot2):
        logit = self.forward(emb_prot1, emb_prot2)
        return nn.Sigmoid()(logit)

    def get_num_parameters(self):
        return sum([p.numel() for p in self.parameters() if p.requires_grad])


class Attention(nn.Module):
    def __init__(self, embed_dim, num_heads=8, num_layers=2, dim_feedforward=1024, out_dim=1, activation=nn.GELU()):
        super(Attention, self).__init__()

        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads,
                                                   dim_feedforward=dim_feedforward, activation=activation,
                                                   batch_first=True)
        # batch x 2 x embed_dim
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # batch x 2
        self.linear = nn.Linear(2 * embed_dim, out_dim)

    def forward(self, emb_prot1, emb_prot2):
        """
        Shape:
            - emb_prot1: (batch_size, 1, emb_dim)
            - emb_prot2: (batch_size, 1, emb_dim)
        """
        concat = torch.cat((emb_prot1, emb_prot2), 1)
        x = self.transformer_encoder(concat)
        x = x.view(x.shape[0], -1)
        return self.linear(x)

    def predict(self, emb_prot1, emb_prot2):
        logit = self.forward(emb_prot1, emb_prot2)
        return nn.Sigmoid()(logit)

    def get_num_parameters(self):
        return sum([p.numel() for p in self.parameters() if p.requires_grad])


if __name__ == '__main__':
    att = Attention(1024, num_heads=8, num_layers=2, dim_feedforward=100)
    print(att.get_num_parameters())

    proj_att = ProjectedAttention(1024, 128, num_heads=8, num_layers=2, dim_feedforward=128)
    print(proj_att.get_num_parameters())

    mlp = MLP(1024, 256, 128, 2)
    print(mlp.get_num_parameters())

    from dscript.models.contact import *
    from dscript.models.interaction import *
    from dscript.models.embedding import *
    projection_dim = 100
    dropout_p = .5
    embedding = FullyConnectedEmbed(6165, projection_dim, dropout=dropout_p)
    # Create contact model
    hidden_dim = 50
    kernel_width = 7
    contact = ContactCNN(projection_dim, hidden_dim, kernel_width)
    # Create the full model
    use_w = True
    pool_width = 9
    model = ModelInteraction(embedding, contact, use_w=use_w, pool_size=pool_width)

    print(sum([p.numel() for p in model.parameters() if p.requires_grad]))
