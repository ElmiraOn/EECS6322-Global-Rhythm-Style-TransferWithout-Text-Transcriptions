import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# models based on previous studies by bahati et al
from onmt_modules.misc import sequence_mask
from onmt_modules.embeddings import PositionalEncoding
from onmt_modules.encoder_transformer import TransformerEncoder as OnmtEncoder
from onmt_modules.decoder_transformer import TransformerDecoder

def Filter_mean(num_rep, codes_mask, max_len_long):
    num_rep = num_rep.unsqueeze(-1)
    codes_mask = codes_mask.unsqueeze(-1) 
    num_rep = num_rep * codes_mask
    right_edge = num_rep.cumsum(dim=1)
    left_edge = torch.zeros_like(right_edge)
    left_edge[:, 1:, :] = right_edge[:, :-1, :]
    right_edge = right_edge.ceil()
    left_edge = left_edge.floor()
    index = torch.arange(1, max_len_long+1, device=num_rep.device).view(1, 1, -1)
    lower = index - left_edge
    right_edge_flip = max_len_long - right_edge
    upper = (index - right_edge_flip).flip(dims=(2,))
    fb = F.relu(torch.min(lower, upper)).float()
    fb = (fb > 0).float() #mean pooling
    norm = fb.sum(dim=-1, keepdim=True)
    norm[norm==0] = 1.0
    fb = fb / norm
    return fb * codes_mask    

class Bhati_Decoder(TransformerDecoder):
    def forward(self, tgt, memory_bank, step=None, **kwargs):
        if step == 0:
            self._init_cache(memory_bank)
        if step is None:
            tgt_lens = kwargs["tgt_lengths"]
        else:    
            tgt_words = kwargs["tgt_words"]
        emb = self.embeddings(tgt, step=step)
        assert emb.dim() == 3 
        output = emb.transpose(0, 1).contiguous()
        src_memory_bank = memory_bank.transpose(0, 1).contiguous()
        pad_idx = self.embeddings.word_padding_idx
        src_lens = kwargs["memory_lengths"]
        src_max_len = self.state["src"].shape[0]
        src_pad_mask = ~sequence_mask(src_lens, src_max_len).unsqueeze(1)
        if step is None:
            tgt_max_len = tgt_lens.max()
            tgt_pad_mask = ~sequence_mask(tgt_lens, tgt_max_len).unsqueeze(1)
        else:    
            tgt_pad_mask = tgt_words.data.eq(pad_idx).unsqueeze(1)
        with_align = kwargs.pop('with_align', False)
        attn_aligns = []
        for i, layer in enumerate(self.transformer_layers):
            layer_cache = self.state["cache"]["layer_{}".format(i)] \
                if step is not None else None
            output, attn, attn_align = layer(
                output,
                src_memory_bank,
                src_pad_mask,
                tgt_pad_mask,
                layer_cache=layer_cache,
                step=step,
                with_align=with_align)
            if attn_align is not None:
                attn_aligns.append(attn_align)
        output = self.layer_norm(output)
        dec_outs = output.transpose(0, 1).contiguous()
        attn = attn.transpose(0, 1).contiguous()
        attentions = {"std": attn}
        if self._copy:
            attentions["copy"] = attn
        if with_align:
            attentions["align"] = attn_aligns[self.alignment_layer] 

        return dec_outs, attentions

class Fast_decoder(object):
    def __init__(self, hparams, type_out):
        if type_out == 'Speech':
            self.dim_freq = hparams.dim_freq
            self.max_decoder_steps = hparams.dec_steps_sp
        elif type_out == 'Text':
            self.dim_freq = hparams.dim_code
            self.max_decoder_steps = hparams.dec_steps_tx
        else: raise ValueError
        self.gate_threshold = hparams.gate_threshold
        self.type_out = type_out
        
    def __call__(self, tgt, memory_bank, memory_lengths, decoder, postnet):
        dec_outs, attns = decoder(tgt, memory_bank, step=None, memory_lengths=memory_lengths)
        spect_gate = postnet(dec_outs)
        spect, gate = spect_gate[:, :, 1:], spect_gate[:, :, :1]
        return spect, gate

    def infer(self, tgt_real, memory_bank, memory_lengths, decoder, postnet):
        bank = memory_bank.size(1)
        device = memory_bank.device
        spect_outputs = torch.zeros((self.max_decoder_steps, bank, self.dim_freq), dtype=torch.float, device=device)
        gate_outputs = torch.zeros((self.max_decoder_steps, B, 1), dtype=torch.float, device=device)
        tgt_words = torch.zeros([B, 1], dtype=torch.float, device=device)
        current_pred = torch.zeros([1, B, self.dim_freq], dtype=torch.float, device=device)
        
        for t in range(self.max_decoder_steps):
            dec_outs, _ = decoder(current_pred, memory_bank, t, memory_lengths=memory_lengths,tgt_words=tgt_words)
            spect_gate = postnet(dec_outs)
            spect, gate = spect_gate[:, :, 1:], spect_gate[:, :, :1]
            spect_outputs[t:t+1] = spect
            gate_outputs[t:t+1] = gate
            stop = (torch.sigmoid(gate) - self.gate_threshold + 0.5).round()
            current_pred = spect.data
            tgt_words = stop.squeeze(-1).t()
            if (stop == 1).all():
                break
        stop_quant = (torch.sigmoid(gate_outputs.data) - self.gate_threshold + 0.5).round().squeeze(-1) 
        len_spect = (stop_quant.cumsum(dim=0)==0).sum(dim=0)
        
        return spect_outputs, len_spect, gate_outputs

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

# model to get attention -- Bhati et al work    
class Prenet(nn.Module):
    def __init__(self, dim_input, dim_output, dropout=0.1):
        super().__init__() 
        
        mlp = nn.Linear(dim_input, dim_output, bias=True)
        pe = PositionalEncoding(dropout, dim_output, 1600)
        
        self.make_prenet = nn.Sequential()
        self.make_prenet.add_module('mlp', mlp)
        self.make_prenet.add_module('pe', pe)
        
        self.word_padding_idx = 1
        
    def forward(self, source, step=None):
        
        for i, module in enumerate(self.make_prenet._modules.values()):
            if i == len(self.make_prenet._modules.values()) - 1:
                source = module(source, step=step)
            else:
                source = module(source)
                
        return source
    
    
 # speech decoder based on Bhati et al work
class speech_decoder(nn.Module):
 
    def __init__(self, hparams):
        super().__init__() 
        
        self.dim_freq = hparams.dim_freq
        self.max_decoder_steps = hparams.dec_steps_sp
        self.gate_threshold = hparams.gate_threshold
        prenet = Prenet(hparams.dim_freq, hparams.dec_rnn_size)
        self.decoder = Bhati_Decoder.from_opt(hparams, prenet) # using decoder from previous study by Bhati et al
        self.postnet = nn.Linear(hparams.dec_rnn_size, hparams.dim_freq+1, bias=True)
        
    def forward(self, tgt, tgt_lengths, memory_bank, memory_lengths):
        
        dec_outs, attentions = self.decoder(tgt, memory_bank, step=None, memory_lengths=memory_lengths, tgt_lengths=tgt_lengths)
        spect_gate = self.postnet(dec_outs)
        spect, gate = spect_gate[:, :, 1:], spect_gate[:, :, :1]
        return spect, gate
    
    
    
class Text_Encoder(nn.Module): #use attentions
    def __init__(self, hparams):
        super().__init__() 
        prenet = Prenet(hparams.dim_code+hparams.dim_spk, hparams.enc_rnn_size)
        self.encoder = OnmtEncoder.from_opt(hparams, prenet)
        
    def forward(self, src, src_lengths, spk_emb):
        
        spk_emb = spk_emb.unsqueeze(0).expand(src.size(0),-1,-1)
        src_spk = torch.cat((src, spk_emb), dim=-1)
        enc_states, memory_bank, src_lengths = self.encoder(src_spk, src_lengths)
        
        return enc_states, memory_bank, src_lengths
         
class training_1(nn.Module):
    def __init__(self, hparams):
        super().__init__() 
        
        self.encoder = Encoder(hparams)
        self.text_encoder = Text_Encoder(hparams)
        self.speech_decoder = speech_decoder(hparams)   
        self.encoder2 = nn.Linear(hparams.dim_spk, hparams.enc_rnn_size, bias=True)
        self.fast_decoder_speech = Fast_decoder(hparams, 'Speech')
        
        
    def pad_sequences_rnn(self, cd_short, num_rep, len_long):
        B, L, C = cd_short.size()
        out_tensor = torch.zeros((B, len_long.max(), C), device=cd_short.device)
        for i in range(B):
            code_sync = cd_short[i].repeat_interleave(num_rep[i], dim=0)
            out_tensor[i, :len_long[i]-1, :] = code_sync
            
        return out_tensor 
    def forward(self, cep_in, mask_long, codes_mask, num_rep, len_short, tgt_spect, len_spect, spk_emb):
        
        cd_long = self.encoder(cep_in, mask_long)
        fb = Filter_mean(num_rep, codes_mask, cd_long.size(1))
        cd_short = torch.bmm(fb.detach(), cd_long)
        cd_short_sync = self.pad_sequences_rnn(cd_short, num_rep, len_spect)
        spk_emb_1 = self.encoder2(spk_emb)
        # text to speech
        _, memory_tx, _ = self.text_encoder(cd_short_sync.transpose(1,0), len_spect, spk_emb)
        memory_tx_spk = torch.cat((spk_emb_1.unsqueeze(0), memory_tx), dim=0)
        self.speech_decoder.decoder.init_state(memory_tx_spk, None, None)
        spect_out, gate_sp_out \
        = self.speech_decoder(tgt_spect, len_spect, memory_tx_spk, len_spect+1)
        
        return spect_out, gate_sp_out
    
    
    
class Generator_2(nn.Module):

    def __init__(self, hparams):
        super().__init__() 
        
        self.encoder = Encoder(hparams)
        self.text_encoder = Text_Encoder(hparams)
        self.speech_decoder = speech_decoder(hparams)   
        self.encoder2 = nn.Linear(hparams.dim_spk, 
                                     hparams.enc_rnn_size, bias=True)
        self.fast_decoder_speech = Fast_decoder(hparams, 'Speech')
        
        
    def forward(self, cep_in, mask_long, codes_mask, num_rep, len_short,
                      tgt_spect, len_spect, 
                      spk_emb):
        
        cd_long = self.encoder(cep_in, mask_long)
        fb = Filter_mean(num_rep, codes_mask, cd_long.size(1))
        
        cd_short = torch.bmm(fb.detach(), cd_long.detach())
        
        spk_emb_1 = self.encoder2(spk_emb)
        
        # text to speech
        _, memory_tx, _ = self.text_encoder(cd_short.transpose(1,0), len_short, 
                                          spk_emb)
        memory_tx_spk = torch.cat((spk_emb_1.unsqueeze(0), memory_tx), dim=0)
        self.speech_decoder.decoder.init_state(memory_tx_spk, None, None)
        spect_out, gate_sp_out \
        = self.speech_decoder(tgt_spect, len_spect, memory_tx_spk, len_short+1)
        
        return spect_out, gate_sp_out
    