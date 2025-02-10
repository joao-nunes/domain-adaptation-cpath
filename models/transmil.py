import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from nystrom_attention import NystromAttention

# Source: https://github.com/szc19990412/TransMIL/blob/main/models/TransMIL.py

class TransformerLayer(nn.Module):
    
    def __init__(self, layer_norm=nn.LayerNorm, dim=512, return_att=False):
        super(TransformerLayer, self).__init__()
        self.norm = layer_norm(dim)
        self.attn = NystromAttention(
            dim=dim,
            dim_head=dim // 8,
            heads=8,
            num_landmarks = dim // 2,
            pinv_iterations=6,
            residual=True,
            dropout=0.1
        )
        self.return_att=return_att

    def forward(self, x):
        att_matrix = self.attn(self.norm(x))
        x = x + att_matrix
        if self.return_att:
            return x, att_matrix
        return x
    
class PPEG(nn.Module):
    
    def __init__(self, dim=512):
        
        super(PPEG, self).__init__()
        
        self.proj = nn.Conv2d(dim, dim, 7, 1, 7 // 2, groups= dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5//2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3//2, groups=dim)
    
    def forward(self, x, H, W):
        B, _, C = x.shape
        cls_token, feat_token = x[:, 0], x[:, 1:]
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        x = self.proj(cnn_feat)+cnn_feat+self.proj1(cnn_feat)+self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x


class TransMIL(nn.Module):
    def __init__(self, num_classes, size=768, n_heads=16, return_att=False):
        super(TransMIL, self).__init__()
        self.pos_layer = PPEG(dim=size)
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, size))
        self.n_classes = num_classes
        self.layer1 = TransformerLayer(dim=size)
        self.layer2 = TransformerLayer(dim=size, return_att=True)
        self.norm = nn.LayerNorm(size)
        self.cls_heads = nn.ModuleList([nn.Linear(size, self.n_classes) for _ in range(n_heads)])
        self.return_att = return_att


    def forward(self, h):
        device = h.device
        # shape(h) = [B, n, 512]
        
        #---->pad
        H = torch.tensor([h.shape[1]])
        _H, _W = int(torch.ceil(torch.sqrt(H))), int(torch.ceil(torch.sqrt(H)))
        add_length = _H * _W - H
        h = torch.cat([h, h[:,:add_length,:]], dim = 1) #[B, N, 512]

        #---->cls_token
        B = h.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1).to(device)
        h = torch.cat((cls_tokens, h), dim=1)

        #---->Translayer x1
        h = self.layer1(h) #[B, N, 512]

        #---->PPEG
        h = self.pos_layer(h, _H, _W) #[B, N, 512]
        
        #---->Translayer x2
        h, att_matrix = self.layer2(h) #[B, N, 512]

        #---->cls_token
        h = self.norm(h)[:,0]

        #---->predict
        logits = [linear(h) for linear in self.cls_heads] #[B, n_classes]
        Y_hat = [torch.argmax(logits[i], dim=1) for i in range(len(logits))]
        Y_prob = [F.softmax(logits[i], dim = 1) for i in range(len(logits))]
        results_dict = {'logits': logits, 'probas': Y_prob, 'preds': Y_hat, 'h': h}
        if self.return_att:
            return att_matrix, results_dict
        return results_dict
        
    
