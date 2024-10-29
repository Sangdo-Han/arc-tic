from typing import Optional
import numpy as np
import torch
from torch import nn
from torch import functional as F

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class ResBlock(nn.Module):
    def __init__(self, ch: int, dropout: float):
        super().__init__()
        self.activation = nn.ReLU(inplace=True)
        self.batch_norm1 = nn.BatchNorm2d(ch)
        self.batch_norm2 = nn.BatchNorm2d(ch)
        self.conv_same1 = nn.Conv2d(ch, ch, kernel_size=3, padding=1)
        self.conv_same2 = nn.Conv2d(ch, ch, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        residual = x
        
        out = self.batch_norm1(x)
        out = self.activation(out)
        out = self.conv_same1(out)
        
        out = self.dropout(out)

        out = self.batch_norm2(out)
        out = self.activation(out)
        out = self.conv_same2(out)

        return out + residual

# down: DownSampling, up: Upsampling
class ConvBlock(nn.Module):
    def __init__(
            self,
            mode: str,
            ch_in: int,
            ch_out: int,
            dropout: float
        ):
        super(ConvBlock, self).__init__()

        self.activation = nn.ReLU()
        self.batch_norm = nn.BatchNorm2d(ch_out)

        if mode=="down":
            self.conv = nn.Conv2d(
                ch_in, ch_out, kernel_size=4, stride=2, padding=0
            )
        elif mode=="up":
            self.conv = nn.ConvTranspose2d(
                ch_in, ch_out, kernel_size=4, stride=2, padding=0
            )
        else: # same
            self.conv = nn.Conv2d(
                ch_in, ch_out, kernel_size=3, padding=1
            )

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, z):
        x = self.conv(z)
        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x


class UpEncoder(nn.Module):
    def __init__(self, channels=[10, 10, 10], latent_dim=10, dropout=0.1):
        super(UpEncoder, self).__init__()
        self.conv1 = ConvBlock("up", 10,  channels[0], dropout)
        self.res12 = ResBlock(channels[0], dropout)
        self.conv2 = ConvBlock("up", channels[0], channels[1], dropout)
        self.res23 = ResBlock(channels[1], dropout)
        self.conv3 = ConvBlock("up", channels[1], channels[2], dropout)
        self.fc = nn.Linear(channels[2] * 38 * 38, latent_dim)

    def forward(self, x):
        residuals = [0] * 3
        x = self.conv1(x)
        x = self.res12(x)
        residuals[0] = x
        x = self.conv2(x)
        x = self.res23(x)
        residuals[1] = x
        x = self.conv3(x)
        residuals[2] = x
        print(f"xshape: {x.shape}")
        x = x.reshape(x.size(0), -1)
        encoded = self.fc(x)
        return encoded, residuals
    
class Decoder(nn.Module):
    def __init__(self, channels=[256, 512, 512], latent_dim=512, dropout=0.1):
        super(Decoder, self).__init__()
        self.channels = channels
        self.fc = nn.Linear(latent_dim, channels[-1] * 2 * 2)
        self.conv3 = ConvBlock("up", channels[-1]*2, channels[-2], dropout)
        self.res32 = ResBlock(channels[-2], dropout)
        self.conv2 = ConvBlock("up", channels[-2]*2, channels[-3], dropout)
        self.res21 = ResBlock(channels[-3], dropout)
        self.conv1 = ConvBlock("up", channels[-3]*2, channels[-3], dropout)
        self.conv0 = nn.Conv2d(channels[-3], 11, kernel_size=3, padding=1)

    def forward(self, z, residuals):
        x = self.fc(z)
        x = x.reshape(x.size(0), self.channels[-1], 2, 2)  # Unflatten using reshape instead of view
        x = torch.cat((x, residuals[2]), dim=1)
        x = self.conv3(x)
        x = self.res32(x)
        x = torch.cat((x, residuals[1]), dim=1)
        x = self.conv2(x)
        x = self.res21(x)
        x = torch.cat((x, residuals[0]), dim=1)
        x = self.conv1(x)
        x = self.conv0(x)
        return x


class Encoder(nn.Module):
    def __init__(self, channels=[256, 512, 512], latent_dim=512, dropout=0.1):
        super(Encoder, self).__init__()
        self.conv1 = ConvBlock("down", 11,  channels[0], dropout)
        self.res12 = ResBlock(channels[0], dropout)
        self.conv2 = ConvBlock("down", channels[0], channels[1], dropout)
        self.res23 = ResBlock(channels[1], dropout)
        self.conv3 = ConvBlock("down", channels[1], channels[2], dropout)
        self.fc = nn.Linear(channels[2] * 2 * 2, latent_dim)

    def forward(self, z):
        residuals = [0] * 3
        x = preprocess_images(z)
        x = self.conv1(x)
        x = self.res12(x)
        residuals[0] = x
        x = self.conv2(x)
        x = self.res23(x)
        residuals[1] = x
        x = self.conv3(x)
        residuals[2] = x
        x = x.reshape(x.size(0), -1)
        encoded = self.fc(x)
        return encoded, residuals
class AttentionSolver(nn.Module):
    def __init__(
            self,
        ):
        super(AttentionSolver, self).__init__()

    
class Decoder(nn.Module):
    def __init__(self, channels=[256, 512, 512], latent_dim=512, dropout=0.1):
        super(Decoder, self).__init__()
        self.channels = channels
        self.fc = nn.Linear(latent_dim, channels[-1] * 2 * 2)
        self.conv3 = ConvBlock("up", channels[-1]*2, channels[-2], dropout)
        self.res32 = ResBlock(channels[-2], dropout)
        self.conv2 = ConvBlock("up", channels[-2]*2, channels[-3], dropout)
        self.res21 = ResBlock(channels[-3], dropout)
        self.conv1 = ConvBlock("up", channels[-3]*2, channels[-3], dropout)
        self.conv0 = nn.Conv2d(channels[-3], 11, kernel_size=3, padding=1)

    def forward(self, z, residuals):
        x = self.fc(z)
        x = x.reshape(x.size(0), self.channels[-1], 2, 2)  # Unflatten using reshape instead of view
        x = torch.cat((x, residuals[2]), dim=1)
        x = self.conv3(x)
        x = self.res32(x)
        x = torch.cat((x, residuals[1]), dim=1)
        x = self.conv2(x)
        x = self.res21(x)
        x = torch.cat((x, residuals[0]), dim=1)
        x = self.conv1(x)
        x = self.conv0(x)
        return x
    
    
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout=0.1):
        super(MLP, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.fc1  = nn.Linear(input_size, hidden_size)
        self.bn1  = nn.BatchNorm1d(hidden_size)
        self.fc2  = nn.Linear(hidden_size, hidden_size)
        self.bn2  = nn.BatchNorm1d(hidden_size)
        self.fc3  = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, z):
        x = self.relu(self.bn1(self.fc1(z)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        output = self.fc3(x)
        return output