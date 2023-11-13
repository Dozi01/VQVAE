import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super(ResidualBlock, self).__init__()

        self.res_block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3,
                      stride=stride, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(out_channels, out_channels,
                      kernel_size=1, stride=stride, bias=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Residual 블록의 forward pass
        x = x + self.res_block(x)
        return x


class Encoder(nn.Module):
    def __init__(self, in_channels: int, num_residual_hidden: int):
        super(Encoder, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(
            in_channels, num_residual_hidden//2 , kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(num_residual_hidden//2 , num_residual_hidden, kernel_size=4, stride=2, padding=1)
        self.relu = nn.ReLU()
        self.resudial1 = ResidualBlock(num_residual_hidden, num_residual_hidden)
        self.resudial2 = ResidualBlock(num_residual_hidden, num_residual_hidden)

    def forward(self, x):
        # Forward pass through convolutional layers
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.resudial1(x)
        x = self.resudial2(x)
        return x



class VectorQuantisation(nn.Module):
    # n_e : embedding num(K) , n_e = 512 in experiment
    # e_dim : embedding dimension , z_e 로부터 결정되는 값이다.
    # beta : parameter for commitment loss
    def __init__(self, n_e : int, e_dim : int, commitment_cost : float):
        super(VectorQuantisation, self).__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.commitment_cost = commitment_cost
        self.embedding = nn.Embedding(
            self.n_e, self.e_dim)  # (K, D) embedding 생성

        self._vq_ste = vq_ste()
        
    def forward(self, z_e):
        z_q = self._vq_ste(z_e, self.embedding, self.e_dim)

        dictionary_learning_loss = F.mse_loss(z_q.detach(), z_e)
        commitment_loss = F.mse_loss(z_q, z_e.detach())
        
        vq_loss = dictionary_learning_loss + self.commitment_cost * commitment_loss
    
        return z_q, vq_loss


def vq(z_e: torch.Tensor, embedding: torch.nn.Embedding, e_dim: int):
    # convert inputs from BCHW -> BHWC
    z_e_shape = z_e.shape
    z_e = z_e.permute(0, 2, 3, 1).contiguous() # contiguous for memory 
    
    # Flatten input
    flatten = z_e.view(-1, e_dim)

    distances = (torch.sum(flatten**2, dim=1, keepdim=True) 
                + torch.sum(embedding.weight**2, dim=1)
                - 2 * torch.matmul(flatten, embedding.weight.t()))
    
    
    encoding_indices = torch.argmin(distances, dim=1)
    z_q = embedding.weight[encoding_indices]

    z_q = z_q.T.reshape(z_e_shape)
    
    return z_q


class straight_thorugh_estimator(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z_e, embedding, e_dim):
        ctx.save_for_backward(z_e)
        return vq(z_e, embedding, e_dim)
    @staticmethod
    def backward(ctx, grad_out):
        z_e, = ctx.saved_tensors
        return grad_out, None, None
    
class vq_ste(nn.Module):
    def __init__(self):
        super(vq_ste, self).__init__()
    def forward(self, z_e, embedding, e_dim):
        return straight_thorugh_estimator.apply(z_e, embedding, e_dim)
    
    
    
class Decoder(nn.Module):
    def __init__(self, e_dim : int, num_residual_hidden, out_channels):
        super(Decoder, self).__init__()

        # Convolutional layers
        self.convTrans1 = nn.ConvTranspose2d(
            e_dim, num_residual_hidden, kernel_size=3, stride=1, padding=1)

        self.resudial1 = ResidualBlock(num_residual_hidden, num_residual_hidden)
        self.resudial2 = ResidualBlock(num_residual_hidden, num_residual_hidden)
        self.convTrans2 = nn.ConvTranspose2d(
            num_residual_hidden, num_residual_hidden // 2, kernel_size=4, stride=2, padding=1)
        self.convTrans3 = nn.ConvTranspose2d(
            num_residual_hidden // 2, out_channels, kernel_size=4, stride=2, padding=1)

        self.relu = nn.ReLU()

    def forward(self, x):
        # Forward pass through convolutional layers
        x = self.convTrans1(x)
        x = self.resudial1(x)
        x = self.resudial2(x)
        x = self.convTrans2(x)
        x = self.convTrans3(x)

        return x


    
class VQVAE(nn.Module):
    def __init__(self, in_channels: int, embedding_dim: int, 
                 num_embeddings: int, commitment_cost: float):
        super(VQVAE, self).__init__()
        
        self._encoder = Encoder(in_channels = in_channels, num_residual_hidden = embedding_dim)
        self._vq = VectorQuantisation(n_e = num_embeddings, e_dim = embedding_dim, commitment_cost = commitment_cost)
        self._decoder = Decoder(e_dim = embedding_dim, num_residual_hidden = embedding_dim, out_channels = in_channels)

    def forward(self, x):
        z_e = self._encoder(x)
        z_q, vq_loss = self._vq(z_e)
        x_recon = self._decoder(z_q)
        
        
        # To minimize inference time, compute the loss outside of the model
        return x_recon, vq_loss


