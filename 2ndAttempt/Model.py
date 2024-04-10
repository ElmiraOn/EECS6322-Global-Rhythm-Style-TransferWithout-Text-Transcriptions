import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.parameter import Parameter
import math
from scipy.stats import uniform
from scipy.spatial.distance import cosine
import warnings
warnings.filterwarnings("ignore")


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
        
    def forward(self, x, mask=None):       
        for conv in self.layers_list:
            x = (conv(x))   
        codes = x.permute(0, 2, 1) #* mask.unsqueeze(-1)
        return codes


class ResampleLayer(nn.Module):
    def __init__(self, b=20, ul=0.95, ur=1.05):
        super(ResampleLayer, self).__init__()
        self.b = b
        self.ul = ul
        self.ur = ur
    def compute_threshold(similarity_values, quantile_range):
        return np.percentile(similarity_values, quantile_range)
    def compute_cosine_similarity(vec1, vec2):
        return 1 - cosine(vec1, vec2)
    def downsample_segment_boundaries(similarity_values, quantile_range, b):
        boundaries = []
        for i in range(1, len(similarity_values)):
            window_start = max(0, i - b)
            window_end = min(len(similarity_values), i + b)
            threshold = compute_threshold(similarity_values[window_start:window_end], quantile_range)
            if similarity_values[i] < threshold:
                boundaries.append(i)
        return boundaries

    def upsample_segment_boundaries(similarity_values, quantile_range, b):
        boundaries = []
        for i in range(1, len(similarity_values)):
            window_start = max(0, i - b)
            window_end = min(len(similarity_values), i + b)
            threshold = compute_threshold(similarity_values[window_start:window_end], quantile_range)
            if similarity_values[i] >= 1 - threshold:
                boundaries.append(i)
        return boundaries

    def mean_pooling(segment):
        if len(segment) == 0:
            return None  # or np.nan
        return np.mean(segment, axis=0)
    def resampler2(frames, ul=0.1, ur=0.9, quantile_range=5, b=20):
        boundaries = [0]
        # Randomly draw global variable G
        G = np.random.uniform(ul, ur)
        for i in range(1, len(frames)):
            # Randomly draw local variable L(t)
            Lt = np.random.uniform(G - 0.05, G + 0.05)
            similarity_values = [compute_cosine_similarity(frames[i], frames[j]) for j in range(i)]
            threshold = Lt - compute_threshold(similarity_values, quantile_range)
            if threshold < 1:
                boundaries += downsample_segment_boundaries(similarity_values, quantile_range, b)
            else:
                boundaries += upsample_segment_boundaries(similarity_values, quantile_range, b)
        boundaries.sort()
        segments = [frames[boundaries[i]:boundaries[i+1]] for i in range(len(boundaries)-1)]
        pooled_segments = [mean_pooling(segment) for segment in segments]
        return pooled_segments


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
        # resampled_x = [self.resample(frame) for frame in x]
        # return resampled_x
        resampled_x = [self.resampler2(frame) for frame in x]
        return resampled_x
class PositionalEncoding(torch.nn.Module):
    """ From the paper: Global Rhythm Style Transfer Without Text Transcriptions

    Sinusoidal positional encoding for non-recurrent neural networks.

    Implementation based on "Attention Is All You Need"
    :cite:`DBLP:journals/corr/VaswaniSPUJGKP17`

    Args:
       dropout (float): dropout parameter
       dim (int): embedding size
    """

    def __init__(self, dropout, dim, max_len=5000):
        if dim % 2 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                             "odd dim (got dim={:f})".format(dim))
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.float) *
                             -(math.log(10000.0) / dim)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(1)
        super(PositionalEncoding, self).__init__()
        self.register_buffer('pe', pe)
        self.dropout = torch.nn.Dropout(p=dropout)
        self.dim = dim

    def forward(self, emb, step=None):
        """Embed inputs.

        Args:
            emb (FloatTensor): Sequence of word vectors
                ``(seq_len, batch_size, self.dim)``
            step (int or NoneType): If stepwise (``seq_len = 1``), use
                the encoding for this position.
        """

        emb = emb * math.sqrt(self.dim)
        if step is None:
            emb = emb + self.pe[:emb.size(0)]
        else:
            emb = emb + self.pe[step]
        emb = self.dropout(emb)
        return emb

class Decoder(torch.nn.Module):
  def __init__(self, num_heads=8, num_layers=4, d_model=256, d_freq=80, dropout=0.1):
    super(Decoder, self).__init__()

    self.pos_encoder = PositionalEncoding(dropout, d_model)

    encoder_layer = torch.nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads)
    self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    decoder_layer = torch.nn.TransformerDecoderLayer(d_model=d_model, nhead=num_heads)
    self.transformer_decoder = torch.nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

  def forward(self, src, tgt):
    src_embed = self.pos_encoder(src)
    tgt_embed = self.pos_encoder(tgt)

    memory = self.transformer_encoder(src_embed)
    output = self.transformer_decoder(tgt_embed, memory)
    return output
# Define the Transformer decoder model
# class Decoder(nn.Module):
#     def __init__(self, d_model=256, num_heads=8, num_encoder_layers = 4, num_decoder_layers = 4, dim_feedforward=2048, dropout=0.1):
#         super(Decoder, self).__init__()
#         self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads,
#                                                                         dim_feedforward=dim_feedforward,dropout=dropout), 
#                                             num_layers=num_encoder_layers)
#         self.decoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(d_model=d_model, nhead=num_heads, 
#                                                                         dim_feedforward=dim_feedforward, dropout=dropout), 
#                                              num_layers=num_decoder_layers)

#     def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None, src_key_padding_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
#         memory = self.encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
#         output = self.decoder(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask, tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask)
#         return output


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