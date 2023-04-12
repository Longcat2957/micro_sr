import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from typing import Tuple

class k3Conv(nn.Module):
    def __init__(self, in_channels, out_channels, stride:int=1, padding:int=1, bias:bool=True, act=nn.ReLU):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=padding, bias=bias)
        # Xavier 초기화
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_normal_(m.weight)
                if m.bias is not None:
                    init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                init.ones_(m.weight)
                init.zeros_(m.bias)
        if act is not None:
            self.act = act(inplace=False)
        else:
            self.act = None

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.act is not None:
            x = self.act(x)
        return x
    

class k1Conv(nn.Module):
    def __init__(self, in_channels, out_channels, stride:int=1, padding:int=0, bias=True, act=nn.ReLU):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=padding,
                              bias=bias)
        # Xavier 초기화
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_normal_(m.weight)
                if m.bias is not None:
                    init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                init.ones_(m.weight)
                init.zeros_(m.bias)
        if act is not None:
            self.act = act(inplace=False)
        else:
            self.act = None

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.act is not None:
            x = self.act(x)
        return x

class Concatenate(nn.Module):
    def __init__(self, dim:int=1) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x1:torch.Tensor, x2:torch.Tensor) -> torch.Tensor:
        return torch.cat((x1, x2), dim=self.dim)

class RepBlock_for_train(nn.Module):
    def __init__(self, hidden_dim:int):
        super().__init__()
        self.hidden_dim = hidden_dim
        # in training phase
        self.t_conv3x3 = k3Conv(hidden_dim, hidden_dim, act=None)
        self.t_conv1x1 = k1Conv(hidden_dim, hidden_dim, act=None)
        self.act = nn.ReLU(inplace=False)
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        skip = x.clone()
        y = self.t_conv3x3(x)
        y = self.t_conv1x1(y)
        y += skip
        y = self.act(y)
        return y

class RepBlock_for_inference(nn.Module):
    def __init__(self, hidden_dim:int):
        super().__init__()
        # in inference phase
        self.i_conv3x3 = k3Conv(hidden_dim, hidden_dim, act=nn.ReLU)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        y = self.i_conv3x3(x)
        return y

class SCSRN(nn.Module):
    """
        SCSRN Network for TRAINING
    """
    def __init__(self, hidden_dim:int=36, upscale_ratio:int=3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.upscale_ratio = upscale_ratio
        _final_feature = 3 * upscale_ratio ** 2

        # Feature 추출기
        self.stem = k3Conv(3, hidden_dim)
        self.fe1 = RepBlock_for_train(hidden_dim)
        self.fe2 = RepBlock_for_train(hidden_dim)
        self.fe3 = RepBlock_for_train(hidden_dim)
        self.fe4 = RepBlock_for_train(hidden_dim)

        # Concatenate
        self.concatenate = Concatenate()

        # 최종 x2 Convolution layer
        self.last1 = k3Conv(hidden_dim+3, _final_feature)
        self.last2 = RepBlock_for_train(_final_feature)

        self.depth_to_space = nn.PixelShuffle(upscale_factor=upscale_ratio)
        

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        skip_connect = x.clone()
        y = self.stem(x)
        y = self.fe1(y)
        y = self.fe2(y)
        y = self.fe3(y)
        y = self.fe4(y)
        z = self.concatenate(skip_connect, y)
        z = self.last1(z)
        z = self.last2(z)
        z = self.depth_to_space(z)
        return z

def reparameterize_weights(rep_block_train: RepBlock_for_train) -> RepBlock_for_inference:
    # Extract weights and biases from the training layers
    W_1 = rep_block_train.t_conv3x3.conv.weight.data
    W_2 = rep_block_train.t_conv1x1.conv.weight.data
    b_1 = rep_block_train.t_conv3x3.conv.bias.data
    b_2 = rep_block_train.t_conv1x1.conv.bias.data
    
    DEVICE = W_1.device

    W_1_flat = W_1.view(W_1.shape[0], -1)
    W_2_flat = W_2.view(W_2.shape[0], W_2.shape[1])
    wh, ww = W_2_flat.shape
    
    # Re-parameterize weights
    W_3 = torch.matmul(W_2_flat, W_1_flat).view(W_1.shape[0],W_1.shape[1],W_1.shape[2],W_1.shape[3]) + torch.eye(3).repeat(wh, ww, 1, 1).to(DEVICE)
    b_3 = torch.matmul(W_2_flat, b_1.view(-1, 1)).squeeze() + b_2
    
    # Create an inference instance and set the weights and biases
    rep_block_inference = RepBlock_for_inference(rep_block_train.hidden_dim)
    rep_block_inference.i_conv3x3.conv.weight.data.copy_(W_3)
    rep_block_inference.i_conv3x3.conv.bias.data.copy_(b_3)

    # return rep_block_inference
    return rep_block_inference

def convert_model(train_net:SCSRN) -> SCSRN:
    DEVICE = next(train_net.parameters()).device
    # 새로운 네트워크를 생성
    inference_network = SCSRN(hidden_dim=train_net.hidden_dim, upscale_ratio=train_net.upscale_ratio).to(DEVICE)
    inference_network.load_state_dict(train_net.state_dict())
    
    # 교체해야 할 모듈은 총 5개
    inference_network.fe1 = reparameterize_weights(train_net.fe1)
    inference_network.fe2 = reparameterize_weights(train_net.fe2)
    inference_network.fe3 = reparameterize_weights(train_net.fe3)
    inference_network.fe4 = reparameterize_weights(train_net.fe4)
    inference_network.last2 = reparameterize_weights(train_net.last2)
    
    return inference_network.to(DEVICE)

if __name__ == "__main__":
    random_input = torch.randn(1, 3, 360, 640)
    scsrn = SCSRN(36, 3)
    sr = scsrn(random_input)
    print(sr.shape)

    train_layer = RepBlock_for_train(36)
    infer_layer = reparameterize_weights(train_layer)
    
    new_network = convert_model(scsrn)
    # print(new_network)