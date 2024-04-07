import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.parameter import Parameter


# Creating a convulutional model with same padding and stride o 1 and relu
class Conv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, num_groups, kernel_size=1, stride=1,
                 bias=True, ):
        super(Conv, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv = torch.nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.activation = nn.ReLU()
        self.initialize_weights()
        self.groupNorm = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)

    def initialize_weights(self):
        nn.init.kaiming_uniform_(self.conv.weight, nonlinearity='relu')

    def forward(self, out):
        return self.conv(out)

class Encoder(nn.Module):
    """Encoder module:
    """
    def __init__(self, hparams):
        super(Encoder, self).__init__()
        
        self.dim_freq = hparams.dim_freq_sea  #dimensionality of the frequency dimension in input data.
        self.dim_enc = hparams.dim_enc_sea # number of channels or features in the encoding layer 
        self.chs_grp = hparams.chs_grp #This parameter represents the number of groups for the GroupNorm layers
        
        layers = []        
        for i in range(5): # first 5 layers
            conv_layer = Conv(self.dim_freq if i==0 else self.dim_enc,512, self.dim_enc//self.chs_grp)
            layers.append(conv_layer)
        #layer 6
        conv_layer = Conv(self.dim_enc,128, 128//self.chs_grp)
        layers.append(conv_layer)   
        #layer 7
        conv_layer = Conv(128, 32, 32//self.chs_grp)
        layers.append(conv_layer)           
        #layer 8
        conv_layer = Conv(32, 4, 1)
        layers.append(conv_layer)   
        
        self.layers_list = nn.ModuleList(layers)
        
    def forward(self, x, mask):       
        for conv in self.layers_list:
            x = (conv(x))   
        codes = x.permute(0, 2, 1) * mask.unsqueeze(-1)
        return codes


class ResampleLayer(nn.Module):
    def __init__(self, b=20, ul=0.95, ur=1.05):
        super(ResampleLayer, self).__init__()
        self.b = b
        self.ul = ul
        self.ur = ur

    def resample(self, frames):
        resampled_frames = []
        G = np.random.uniform(self.ul, self.ur)  # Draw global variable G

        for i, frame in enumerate(frames):
            L = np.random.uniform(G - 0.05, G + 0.05)  # Draw local variable L(t)

            # Calculate threshold p(t)
            p = L - np.percentile(frames[max(0, i - self.b):min(len(frames), i + self.b)], 50)

            if np.random.rand() < p:
                if np.random.rand() < 1:
                    # Merge into previous segment or start a new segment
                    resampled_frames[-1] += frame if resampled_frames else frame
                else:
                    # Form one or two new segments
                    resampled_frames.append(frame)
                    if np.random.rand() < 1 - p:
                        resampled_frames.append(frame)
            else:
                resampled_frames.append(frame)

        return resampled_frames

    def forward(self, x):
        # Assuming x is a batch of frames
        resampled_x = [self.resample(frame) for frame in x]
        return resampled_x

# Define the Transformer decoder model
class Decoder(nn.Module):
    def __init__(self, d_model=256, num_heads=8, num_encoder_layers = 4, num_decoder_layers = 4, dim_feedforward=2048, dropout=0.1):
        super(Decoder, self).__init__()
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads,
                                                                        dim_feedforward=dim_feedforward,dropout=dropout), 
                                            num_layers=num_encoder_layers)
        self.decoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(d_model=d_model, nhead=num_heads, 
                                                                        dim_feedforward=dim_feedforward, dropout=dropout), 
                                             num_layers=num_decoder_layers)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None, src_key_padding_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        memory = self.encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        output = self.decoder(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask, tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask)
        return output


class Model(nn.Module):
    def __init__(self, hparams):
        super(Model, self).__init__()
        self.encoder = Encoder(hparams)
        self.resample_layer = ResampleLayer()
        self.decoder = Decoder()

    def forward(self, x, c_trg): 
        x = x.transpose(2,1)
        codes = self.encoder(x)
        
        # Pass through resampler
        resampled_codes = self.resample_layer(codes)
        
        encoder_outputs = torch.cat((codes, c_trg.unsqueeze(1).expand(-1,x.size(-1),-1)), dim=-1)
        mel_outputs = self.decoder(encoder_outputs)

        return mel_outputs
    
    def encode(self, x, mask):
        x = x.transpose(2,1)
        codes = self.encoder(x, mask)
        return codes
    
    def decode(self, codes, c_trg):
        encoder_outputs = torch.cat((codes, c_trg.unsqueeze(1).expand(-1,codes.size(1),-1)), dim=-1)
        mel_outputs = self.decoder(encoder_outputs)
        return mel_outputs
    
    
    
# class Encoder_2(nn.Module):
#     """Encoder module:
#     """
#     def __init__(self, hparams):
#         super().__init__()
        
#         self.dim_freq = hparams.dim_freq_sea
#         self.dim_enc = hparams.dim_enc_sea
#         self.chs_grp = hparams.chs_grp
#         self.dim_neck = hparams.dim_neck_sea
        
#         convolutions = []        
#         for i in range(5):
#             conv_layer = M43_Sequential(
#                 ConvNorm(self.dim_freq if i==0 else self.dim_enc,
#                          self.dim_enc,
#                          kernel_size=5, stride=1,
#                          padding=2,
#                          dilation=1, w_init_gain='relu'),
#                 GroupNorm_Mask(self.dim_enc//self.chs_grp, self.dim_enc))
#             convolutions.append(conv_layer)
             
#         conv_layer = M43_Sequential(
#                 ConvNorm(self.dim_enc,
#                          128,
#                          kernel_size=5, stride=1,
#                          padding=2,
#                          dilation=1, w_init_gain='relu'),
#                 GroupNorm_Mask(128//self.chs_grp, 128))
#         convolutions.append(conv_layer)   
        
#         conv_layer = M43_Sequential(
#                 ConvNorm(128,
#                          32,
#                          kernel_size=5, stride=1,
#                          padding=2,
#                          dilation=1, w_init_gain='relu'),
#                 GroupNorm_Mask(32//self.chs_grp, 32))
#         convolutions.append(conv_layer)           
        
#         conv_layer = M43_Sequential(
#                 ConvNorm(32,
#                          self.dim_neck,
#                          kernel_size=5, stride=1,
#                          padding=2,
#                          dilation=1, w_init_gain='linear'),
#                 GroupNorm_Mask(1, self.dim_neck))
#         convolutions.append(conv_layer)   
            
#         self.convolutions = nn.ModuleList(convolutions)
        

#     def forward(self, x, mask):
                
#         for i in range(len(self.convolutions)-1):
#             x = F.relu(self.convolutions[i](x, mask))
            
#         x = self.convolutions[-1](x, mask)    
            
#         codes = x.permute(0, 2, 1) * mask.unsqueeze(-1)

#         return codes    