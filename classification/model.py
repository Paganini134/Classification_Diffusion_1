import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet18, resnet50
from torchvision.models import resnet18, ResNet18_Weights

# ConditionalLinear: multiplies the linear projection by a time-dependent embedding.
# Supports two types of embeddings: 'linear' (learnable) and 'sinusoidal' (fixed sinusoidal encoding).
class ConditionalLinear(nn.Module):
    def __init__(self, num_in, num_out, n_steps, embedding_type="linear"):
        super(ConditionalLinear, self).__init__()
        self.num_out = num_out
        self.lin = nn.Linear(num_in, num_out)
        self.embedding_type = embedding_type
        if embedding_type == "linear":
            self.embed = nn.Embedding(n_steps, num_out)  # Learnable embedding for conditioning
            self.embed.weight.data.uniform_()  # Uniform initialization
        elif embedding_type == "sinusoidal":
            # Precompute sinusoidal embeddings and register as buffer (non-learnable)
            self.register_buffer('sinusoidal_embedding', self._build_sinusoidal_embedding(n_steps, num_out))
        else:
            raise ValueError("Invalid embedding type. Use 'linear' or 'sinusoidal'.")

    def _build_sinusoidal_embedding(self, n_steps, num_out):
        # Create sinusoidal embeddings for each timestep.
        # Each row corresponds to a timestep embedding.
        embedding = torch.zeros(n_steps, num_out)
        position = torch.arange(0, n_steps, dtype=torch.float).unsqueeze(1)  # Shape: (n_steps, 1)
        div_term = torch.exp(torch.arange(0, num_out, 2, dtype=torch.float) * -(math.log(10000.0) / num_out))
        embedding[:, 0::2] = torch.sin(position * div_term)
        embedding[:, 1::2] = torch.cos(position * div_term)
        return embedding

    def forward(self, x, t):
        out = self.lin(x)
        if self.embedding_type == "linear":
            gamma = self.embed(t)
        elif self.embedding_type == "sinusoidal":
            gamma = self.sinusoidal_embedding[t]  # t is assumed to be a tensor of indices
        out = gamma.view(-1, self.num_out) * out
        return out

# Modified ConditionalModel to include multihead attention blocks when arch == 'attention'
# and to choose between sinusoidal and linear timestep embeddings.
class ConditionalModel(nn.Module):
    def __init__(self, config, guidance=False):
        super(ConditionalModel, self).__init__()
        n_steps = config.diffusion.timesteps + 1
        # data_dim is used only for non-attention encoders; for images, x has shape (3, 128, 128)
        data_dim = config.model.data_dim  
        y_dim = config.data.num_classes  # e.g., 4 classes, so y shape is (batch, 4)
        arch = config.model.arch         # Architecture type: 'linear', 'simple', 'attention', etc.
        self.arch = arch                # Save architecture type for use in forward()
        feature_dim = config.model.feature_dim  # Desired output feature dimension
        hidden_dim = config.model.hidden_dim
        self.guidance = guidance

        # Choose the embedding type (default is "linear")
        embedding_type = getattr(config.model, "embeddings", "linear")

        # ---------------------------
        # Encoder for x
        # ---------------------------
        if arch == 'attention':
            # Use a patch-based attention encoder for images.
            patch_size = getattr(config.model, "patch_size", 16)
            self.patch_size = patch_size
            token_dim = getattr(config.model, "token_dim", 64)
            self.token_dim = token_dim
            self.patch_embed = nn.Conv2d(in_channels=3, out_channels=token_dim,
                                         kernel_size=patch_size, stride=patch_size)
            self.num_patches = (128 // patch_size) ** 2
            num_heads = getattr(config.model, "num_heads", 4)
            self.attention = nn.MultiheadAttention(embed_dim=token_dim, num_heads=num_heads)
            self.token_combiner = nn.Sequential(
                nn.LayerNorm(token_dim),
                nn.Linear(token_dim, feature_dim)
            )
        else:
            if config.data.dataset == 'toy':
                self.encoder_x = nn.Linear(data_dim, feature_dim)
            elif config.data.dataset in ['FashionMNIST', 'MNIST', 'CIFAR10', 'CIFAR100', 'IMAGENE100', 'MED_DATA']:
                if arch == 'linear':
                    self.encoder_x = nn.Sequential(
                        nn.Linear(data_dim, hidden_dim),
                        nn.BatchNorm1d(hidden_dim),
                        nn.Softplus(),
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.BatchNorm1d(hidden_dim),
                        nn.Softplus(),
                        nn.Linear(hidden_dim, feature_dim)
                    )
                elif arch == 'simple':
                    self.encoder_x = nn.Sequential(
                        nn.Linear(data_dim, 300),
                        nn.BatchNorm1d(300),
                        nn.ReLU(),
                        nn.Linear(300, 100),
                        nn.BatchNorm1d(100),
                        nn.ReLU(),
                        nn.Linear(100, feature_dim)
                    )
                elif arch == 'lenet':
                    self.encoder_x = LeNet(feature_dim, config.model.n_input_channels, config.model.n_input_padding)
                elif arch == 'lenet5':
                    self.encoder_x = LeNet5(feature_dim, config.model.n_input_channels, config.model.n_input_padding)
                else:
                    self.encoder_x = FashionCNN(out_dim=feature_dim)
            else:
                self.encoder_x = nn.Sequential(
                    nn.Linear(data_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.Softplus(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.Softplus(),
                    nn.Linear(hidden_dim, feature_dim)
                )
        
        self.norm = nn.BatchNorm1d(feature_dim)

        # ---------------------------
        # Conditional Unet (Fully Connected) Part
        # ---------------------------
        if self.guidance:
            self.lin1 = ConditionalLinear(y_dim * 2, feature_dim, n_steps, embedding_type=embedding_type)
        else:
            self.lin1 = ConditionalLinear(y_dim, feature_dim, n_steps, embedding_type=embedding_type)
        self.unetnorm1 = nn.BatchNorm1d(feature_dim)
        self.lin2 = ConditionalLinear(feature_dim, feature_dim, n_steps, embedding_type=embedding_type)
        self.unetnorm2 = nn.BatchNorm1d(feature_dim)
        self.lin3 = ConditionalLinear(feature_dim, feature_dim, n_steps, embedding_type=embedding_type)
        self.unetnorm3 = nn.BatchNorm1d(feature_dim)
        self.lin4 = nn.Linear(feature_dim, y_dim)

    def forward(self, x, y, t, yhat=None):
        # ---------------------------
        # Process Input x Using the Appropriate Encoder
        # ---------------------------
        if self.arch == 'attention':
            batch_size = x.size(0)
            patches = self.patch_embed(x)
            patches = patches.flatten(2).transpose(1, 2)
            patches = patches.transpose(0, 1)
            attn_output, _ = self.attention(patches, patches, patches)
            attn_output = attn_output.mean(dim=0)
            x = self.token_combiner(attn_output)
            x = self.norm(x)
        else:
            x = self.encoder_x(x)
            x = self.norm(x)
        
        # ---------------------------
        # Process Conditional Inputs
        # ---------------------------
        if self.guidance:
            y = torch.cat([y, yhat], dim=-1)
        y = self.lin1(y, t)
        y = self.unetnorm1(y)
        y = F.softplus(y)
        y = x * y
        y = self.lin2(y, t)
        y = self.unetnorm2(y)
        y = F.softplus(y)
        y = self.lin3(y, t)
        y = self.unetnorm3(y)
        y = F.softplus(y)
        return self.lin4(y)
