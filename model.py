import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from random import sample
import math
from torch.distributions.kl import kl_divergence
from torch.distributions import Normal

class embItemLayerEnhance(nn.Module):
    def __init__(self,item_length,emb_dim):
        super(embItemLayerEnhance, self).__init__()
        self.emb_item = nn.Embedding(item_length,emb_dim)

    def forward(self,item_id):
        item_f = self.emb_item(item_id)
        return item_f

class predictModule(nn.Module):
    def __init__(self, emb_dim, hid_dim):
        super(predictModule, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(emb_dim*2,hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim,1))
    
    def forward(self, user_spf1, user_spf2, i_feat):
        '''
            user_spf : [bs,dim]
            i_feat : [bs,dim]
            neg_samples_feat: [bs,1/99,dim] 1 for train, 99 for test
        '''
        user_spf1 = user_spf1.unsqueeze(1).expand_as(i_feat)
        user_item_concat_feat_d1 = torch.cat((user_spf1,i_feat),-1)
        logits_d1 = torch.sigmoid(self.fc(user_item_concat_feat_d1))

        user_spf2 = user_spf2.unsqueeze(1).expand_as(i_feat)
        user_item_concat_feat_d2 = torch.cat((user_spf2,i_feat),-1)
        logits_d2 = torch.sigmoid(self.fc(user_item_concat_feat_d2))

        return logits_d1.squeeze(), logits_d2.squeeze()

class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):

        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2) # as Conv1D requires (N, C, Length)
        outputs += inputs
        return outputs

# pls use the following self-made multihead attention layer
# in case your pytorch version is below 1.16 or for other reasons
# https://github.com/pmixer/TiSASRec.pytorch/blob/master/model.py
class Log2feats(torch.nn.Module):
    def __init__(self, user_length, user_emb_dim, item_length, item_emb_dim, seq_len, hid_dim):
        super(Log2feats, self).__init__()
        self.pos_emb = torch.nn.Embedding(seq_len, item_emb_dim) # TO IMPROVE
        self.emb_dropout = torch.nn.Dropout(p=0.5)

        self.attention_layernorms = torch.nn.ModuleList() # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        self.last_layernorm = torch.nn.LayerNorm(user_emb_dim, eps=1e-8)

        for _ in range(2):
            new_attn_layernorm = torch.nn.LayerNorm(user_emb_dim, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer =  torch.nn.MultiheadAttention(user_emb_dim,
                                                            8,
                                                            0.5)
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(user_emb_dim, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(user_emb_dim, 0.5)
            self.forward_layers.append(new_fwd_layer)

    def forward(self, log_seqs):
        seqs = log_seqs
        positions = np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1])
        seqs += self.pos_emb(torch.LongTensor(positions).cuda())
        seqs = self.emb_dropout(seqs)

        timeline_mask = torch.BoolTensor(log_seqs.cpu() == 0).cuda()
        seqs *= ~timeline_mask # broadcast in last dim

        tl = seqs.shape[1] # time dim len for enforce causality
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device="cuda"))

        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            Q = self.attention_layernorms[i](seqs)
            mha_outputs, _ = self.attention_layers[i](Q, seqs, seqs, 
                                            attn_mask=attention_mask)
                                            # key_padding_mask=timeline_mask
                                            # need_weights=False) this arg do not work?
            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs *=  ~timeline_mask

        log_feats = self.last_layernorm(seqs) # (U, T, C) -> (U, -1, C)

        return log_feats


class ATTENTION(torch.nn.Module): # from cao et al. C2DSR
    def __init__(self):
        super(ATTENTION, self).__init__()
        self.emb_dropout = torch.nn.Dropout(p=0.2)
        # TODO: loss += args.l2_emb for regularizing embedding vectors during training
        # https://stackoverflow.com/questions/42704283/adding-l1-l2-regularization-in-pytorch
        self.attention_layernorms = torch.nn.ModuleList() # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        self.last_layernorm = torch.nn.LayerNorm(128, eps=1e-8)

        for _ in range(2):
            new_attn_layernorm = torch.nn.LayerNorm(128, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer =  torch.nn.MultiheadAttention(128,
                                                            1,
                                                            0.2)
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(128, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(128, 0.2)
            self.forward_layers.append(new_fwd_layer)

    def forward(self, seqs):
        # positions = np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1])
        seqs = self.emb_dropout(seqs)
        tl = seqs.shape[1] # time dim len for enforce causality
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool))
        attention_mask = attention_mask.cuda()

        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            Q = self.attention_layernorms[i](seqs)
            mha_outputs, _ = self.attention_layers[i](Q, seqs, seqs,
                                            attn_mask=attention_mask)
                                            # key_padding_mask=timeline_mask
                                            # need_weights=False) this arg do not work?
            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)

        log_feats = self.last_layernorm(seqs) # (U, T, C) -> (U, -1, C)

        return log_feats

class IMVAE(nn.Module):
    # p_dims: [200, 600, n_items] --> [128,256,128]
    def __init__(self, item_length, item_emb_dim, hid_dim, p_dims, q_dims=None, trans_encoder=None, cs_setting=None, dropout=0.5):
        super(IMVAE, self).__init__()
        self.item_emb_layer = embItemLayerEnhance(item_length, item_emb_dim)
        self.encode1 = ATTENTION()
        self.encode2 = ATTENTION()
        self.p_dims = p_dims
        self.trans_encoder = trans_encoder
        self.cs_setting = cs_setting
        print('cs setting:', self.cs_setting)
        print('encoder:', self.trans_encoder)
        if q_dims:
            assert q_dims[0] == p_dims[-1], "In and Out dimensions must equal to each other"
            assert q_dims[-1] == p_dims[0], "Latent dimension for p- and q- network mismatches."
            self.q_dims = q_dims
        else:
            self.q_dims = p_dims[::-1]
        # Last dimension of q- network is for mean and variance
        temp_q_dims = self.q_dims[:-1] + [self.q_dims[-1] * 2]
        temp_q_dims_trans = [self.q_dims[0] * 2] + self.q_dims[1:-1] + [self.q_dims[-1] * 2]

        i = 0 
        self.q_layers = nn.ModuleList()
        self.p_layers = nn.ModuleList()
        self.q_layers_d2 = nn.ModuleList()
        self.p_layers_d2 = nn.ModuleList()
        for d_in, d_out in zip(temp_q_dims[:-1], temp_q_dims[1:]):
            self.q_layers.append(nn.Linear(d_in, d_out))

        for d_in, d_out in zip(self.p_dims[:-1], self.p_dims[1:]):
            self.p_layers.append(nn.Linear(d_in, d_out))

        for d_in, d_out in zip(temp_q_dims[:-1], temp_q_dims[1:]):
            self.q_layers_d2.append(nn.Linear(d_in, d_out))

        for d_in, d_out in zip(self.p_dims[:-1], self.p_dims[1:]):
            self.p_layers_d2.append(nn.Linear(d_in, d_out))

        if self.trans_encoder.lower() == 'mlp':
            self.q_layers_trans = nn.ModuleList()
            for d_in, d_out in zip(temp_q_dims_trans[:-1], temp_q_dims_trans[1:]):
                self.q_layers_trans.append(nn.Linear(d_in, d_out))
        self.p_layers_trans = nn.ModuleList()
        for d_in, d_out in zip(self.p_dims[:-1], self.p_dims[1:]):
            self.p_layers_trans.append(nn.Linear(d_in, d_out))
        
        if self.trans_encoder.lower() == 'mlp':
            self.q_layers_d2_trans = nn.ModuleList()
            for d_in, d_out in zip(temp_q_dims_trans[:-1], temp_q_dims_trans[1:]):
                self.q_layers_d2_trans.append(nn.Linear(d_in, d_out))
        self.p_layers_d2_trans = nn.ModuleList()
        for d_in, d_out in zip(self.p_dims[:-1], self.p_dims[1:]):
            self.p_layers_d2_trans.append(nn.Linear(d_in, d_out))
        
        
        if self.trans_encoder.lower() == 'attention':
            self.encode1_trans = nn.MultiheadAttention(embed_dim=item_emb_dim, num_heads=4)
            self.encode2_trans = nn.MultiheadAttention(embed_dim=item_emb_dim, num_heads=4)
            self.encode1_trans_down = nn.Linear(item_emb_dim, temp_q_dims[-1])
            self.encode2_trans_down = nn.Linear(item_emb_dim, temp_q_dims[-1])
            self.layernorm = nn.LayerNorm(item_emb_dim)
        
        if self.cs_setting:
            self.qr_layers_trans = nn.ModuleList()
            self.qr_layers_d2_trans = nn.ModuleList()
            for d_in, d_out in zip(temp_q_dims[:-1], temp_q_dims[1:]):
                self.qr_layers_trans.append(nn.Linear(d_in, d_out))
            for d_in, d_out in zip(temp_q_dims[:-1], temp_q_dims[1:]):
                self.qr_layers_d2_trans.append(nn.Linear(d_in, d_out))
        
        self.register_parameter('prior_mu',nn.Parameter(torch.tensor(0.0)))
        self.register_parameter('prior_logvar',nn.Parameter(torch.tensor(1.0)))
        self.register_parameter('prior_mu2',nn.Parameter(torch.tensor(0.0)))
        self.register_parameter('prior_logvar2',nn.Parameter(torch.tensor(1.0)))
        
        self.drop = nn.Dropout(dropout)
        self.down1 = nn.Linear(item_emb_dim*3, item_emb_dim) #recon_z, recon_z2t, recon_z_aug
        self.down2 = nn.Linear(item_emb_dim*3, item_emb_dim)
        self.predictModule = predictModule(item_emb_dim,hid_dim)
        self.disc1 = nn.Linear(item_emb_dim*2, 1)
        self.disc2 = nn.Linear(item_emb_dim*2, 1)
        self.init_weights()
        

    
    def forward(self,u_node,i_node,neg_samples,seq_d1,seq_d2,corr_seq_d1,corr_seq_d2,cs_label,domain_id,isTrain):
        i_feat = self.item_emb_layer(i_node).unsqueeze(1)
        neg_samples_feat = self.item_emb_layer(neg_samples)
        
        seq_d1_feat = self.encode1(self.item_emb_layer(seq_d1))
        input_trans = seq_d1_feat #[N,L,E]        
        input  = torch.mean(seq_d1_feat,1)
        seq_d2_feat = self.encode2(self.item_emb_layer(seq_d2))
        input2_trans = seq_d2_feat
        input2  = torch.mean(seq_d2_feat,1)
        corr_seq_d1_feat = self.encode1(self.item_emb_layer(corr_seq_d1))
        aug_input  = torch.mean(corr_seq_d1_feat,1)
        corr_seq_d2_feat = self.encode2(self.item_emb_layer(corr_seq_d2))
        aug_input2  = torch.mean(corr_seq_d2_feat,1)

        mu, logvar = self.encode(input,self.q_layers)
        z = self.reparameters(mu, logvar)
        recon_z = self.decode(z,self.p_layers)

        mu2, logvar2 = self.encode(input2,self.q_layers_d2)
        z2 = self.reparameters(mu2, logvar2)
        recon_z2 = self.decode(z2,self.p_layers_d2)

        mu_aug, logvar_aug = self.encode(aug_input,self.q_layers)
        z_aug = self.reparameters(mu_aug, logvar_aug)
        recon_z_aug = self.decode(z,self.p_layers)

        mu2_aug, logvar2_aug = self.encode(aug_input2,self.q_layers_d2)
        z2_aug = self.reparameters(mu2_aug, logvar2_aug)
        recon_z2_aug = self.decode(z2_aug,self.p_layers_d2)

        if self.cs_setting:
            mu_r, logvar_r = self.encode(input,self.qr_layers_trans) #previous mlp
            z_r = self.reparameters(mu_r, logvar_r)
            mu2_r, logvar2_r = self.encode(input2,self.qr_layers_d2_trans)
            z_r2 = self.reparameters(mu2_r, logvar2_r)
        
        if self.trans_encoder.lower() == 'attention':
            mu_t, logvar_t = self.encode_t(input2_trans, input_trans, input_trans, self.encode1_trans, self.encode1_trans_down) #attention
            z_t = self.reparameters(mu_t, logvar_t)
            recon_z_t = self.decode(z_t,self.p_layers_trans)
            mu2_t, logvar2_t = self.encode_t(input_trans, input2_trans, input2_trans, self.encode2_trans, self.encode2_trans_down)
            z2_t = self.reparameters(mu2_t, logvar2_t)
            recon_z2_t = self.decode(z2_t,self.p_layers_d2_trans)
        
        elif self.trans_encoder.lower() == 'mlp':
            mu_t, logvar_t = self.encode(torch.cat((input, input2), dim = 1),self.q_layers_trans)
            z_t = self.reparameters(mu_t, logvar_t)
            recon_z_t = self.decode(z_t,self.p_layers_trans)
            mu2_t, logvar2_t = self.encode(torch.cat((input, input2), dim = 1),self.q_layers_d2_trans)
            z2_t = self.reparameters(mu2_t, logvar2_t)
            recon_z2_t = self.decode(z2_t,self.p_layers_d2_trans)

        i_feat = torch.cat((i_feat,neg_samples_feat),1)   

        #reconstruction
        if self.cs_setting:
            if isTrain:
                logits_d1, logits_d2 = self.predictModule(self.down1(torch.cat((recon_z,recon_z2_t, recon_z_aug),-1)), self.down2(torch.cat((recon_z2,recon_z_t, recon_z2_aug),-1)), i_feat)
            else:
                recon_z_r = self.decode(z_r,self.p_layers_trans)
                recon_z_r2 = self.decode(z_r2,self.p_layers_d2_trans) 
                logits_d1_o, logits_d2_o = self.predictModule(self.down1(torch.cat((recon_z,recon_z2_t, recon_z_aug),-1)), self.down2(torch.cat((recon_z2,recon_z_t, recon_z2_aug),-1)), i_feat)
                logits_d1_cs, logits_d2_cs = self.predictModule(self.down1(torch.cat((recon_z,recon_z_r2, recon_z_aug),-1)), self.down2(torch.cat((recon_z2,recon_z_r, recon_z2_aug),-1)), i_feat)
                logits_d1 = logits_d1_o * (1-cs_label.unsqueeze(1)) + logits_d1_cs * cs_label.unsqueeze(1) * (1-domain_id.unsqueeze(1))
                logits_d2 = logits_d2_o * (1-cs_label.unsqueeze(1)) + logits_d2_cs * cs_label.unsqueeze(1) * domain_id.unsqueeze(1)
        else:
            logits_d1, logits_d2 = self.predictModule(self.down1(torch.cat((recon_z,recon_z2_t, recon_z_aug),-1)), self.down2(torch.cat((recon_z2,recon_z_t, recon_z2_aug),-1)), i_feat)
        
        KLD1_Generative =  self._kld_gauss(mu2_t, logvar2_t, torch.zeros_like(mu2_t), torch.ones_like(logvar2_t)) + self._kld_gauss(mu, logvar, self.prior_mu.expand(self.q_dims[-1]), self.prior_logvar.expand(self.q_dims[-1])) + self._kld_gauss(mu_aug, logvar_aug, self.prior_mu.expand(self.q_dims[-1]), self.prior_logvar.expand(self.q_dims[-1])) #-0.5 * torch.sum(1 + logvar_t - mu_t.pow(2) - logvar_t.exp())  #-0.5 * torch.sum(1 + logvar_t - mu_t.pow(2) - logvar_t.exp())
        KLD2_Generative =  self._kld_gauss(mu_t, logvar_t, torch.zeros_like(mu_t), torch.ones_like(logvar_t)) + self._kld_gauss(mu2, logvar2, self.prior_mu2.expand(self.q_dims[-1]), self.prior_logvar2.expand(self.q_dims[-1])) + self._kld_gauss(mu2_aug, logvar2_aug, self.prior_mu2.expand(self.q_dims[-1]), self.prior_logvar2.expand(self.q_dims[-1])) # KLD2_Generative = - 0.5 * torch.sum(1 + logvar2_t - mu2_t.pow(2) - logvar2_t.exp())

        if self.cs_setting:
            KLD1_t =  self._kld_gauss(mu2_t, logvar2_t, mu2_r, logvar2_r)  
            KLD2_t =  self._kld_gauss(mu_t, logvar_t, mu_r, logvar_r) 
        else:
            KLD1_t =  self._kld_gauss(mu2_t, logvar2_t, mu2, logvar2)  
            KLD2_t =  self._kld_gauss(mu_t, logvar_t, mu, logvar) 

        KLD1_aug =  self._kld_gauss(mu_aug, logvar_aug, mu, logvar) 
        KLD2_aug =  self._kld_gauss(mu2_aug, logvar2_aug, mu2, logvar2) 

        aug_cls_d1, aug_cls_d2 = self.disc1(torch.cat((aug_input,recon_z_aug),-1)),self.disc2(torch.cat((aug_input2,recon_z2_aug),-1))
        return logits_d1,logits_d2,torch.sigmoid(aug_cls_d1), torch.sigmoid(aug_cls_d2),KLD1_Generative,KLD1_t, KLD1_aug,KLD2_Generative,KLD2_t, KLD2_aug

    def encode(self, input, q_layers):
        h = F.normalize(input)
        h = self.drop(h)
        
        for i, layer in enumerate(q_layers):
            h = layer(h)
            if i != len(q_layers) - 1:
                h = F.tanh(h)
            else:
                mu = h[:, :self.q_dims[-1]]
                logvar = h[:, self.q_dims[-1]:]
        return mu, logvar
    
    def encode_t(self, k, q, v, attention_layers, attention_layer_down):
        h_t, _t = attention_layers(q.permute(1,0,2), k.permute(1,0,2) , v.permute(1,0,2))
        h_t = self.layernorm(h_t)
        h_t = h_t.mean(dim = 0)
        h_t_down = attention_layer_down(h_t)
        mu = h_t_down[:, :self.q_dims[-1]]
        logvar = h_t_down[:, self.q_dims[-1]:]
        return mu, logvar

    def encode_cvae(self, input, condition, q_layers):
        h = F.normalize(input)
        h = self.drop(h)
        for i, layer in enumerate(q_layers):
            if i != len(q_layers) - 1:
                h = layer(h)
                h = F.tanh(h)
            else:
                # print("h shape:{}".format(h.shape))
                h = layer(torch.cat((h,condition),-1))
                mu = h[:, :self.q_dims[-1]]
                logvar = h[:, self.q_dims[-1]:]
        return mu, logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu
    
    def reparameters(self, mean, logstd):
        sigma = torch.exp(0.1 + 0.9 * F.softplus(torch.clamp_max(logstd, 0.4)))
        gaussian_noise = torch.randn(mean.size(0), self.q_dims[-1]).cuda(mean.device)
        if self.training:
            self.sigma = sigma
            sampled_z = gaussian_noise * sigma + mean
        else:
            sampled_z = mean
            # kld_loss = self._kld_gauss(mean, logstd, torch.zeros_like(mean), torch.ones_like(logstd))
        return sampled_z
    
    
    def _kld_gauss(self, mu_1, logsigma_1, mu_2, logsigma_2): #follow cdrib
        """Using std to compute KLD"""
        sigma_1 = torch.exp(0.1 + 0.9 * F.softplus(torch.clamp_max(logsigma_1, 0.4)))
        sigma_2 = torch.exp(0.1 + 0.9 * F.softplus(torch.clamp_max(logsigma_2, 0.4))) ##kld with clamp, logsigma = torch.ones_like(logsigma_1)
        q_target = Normal(mu_1, sigma_1)
        q_context = Normal(mu_2, sigma_2)
        kl = kl_divergence(q_target, q_context).mean(dim=0).sum()
        return kl

    
    def decode(self, z, p_layers):
        h = z
        for i, layer in enumerate(p_layers):
            h = layer(h)
            if i != len(p_layers) - 1:
                h = F.tanh(h)
        return h

    def decode_cvae(self, z, condition, p_layers):
        h = z
        for i, layer in enumerate(p_layers):
            if i != len(p_layers) - 1:
                h = layer(h)
                h = F.tanh(h)
            else:
                h = layer(torch.cat((h,condition),-1))
        return h    

    def init_weights(self):
        for layer in self.q_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0/(fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for Biases
            layer.bias.data.normal_(0.0, 0.001)
        
        for layer in self.p_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0/(fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for Biases
            layer.bias.data.normal_(0.0, 0.001)
    
