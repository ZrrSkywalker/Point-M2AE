import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from timm.models.layers import trunc_normal_
import sys
sys.path.append("../utils")
sys.path.append("..")
sys.path.append("./")
from checkpoint import get_missing_parameters_message, get_unexpected_parameters_message
import random
from extensions.chamfer_dist import ChamferDistanceL2
from pointnet2_utils import PointNetFeaturePropagation_
from utils.logger import *
from modules import *

# Hierarchical Encoder
class H_Encoder_seg(nn.Module):

    def __init__(self, encoder_depths=[5, 5, 5], num_heads=6, encoder_dims=[96, 192, 384], local_radius=[0.32, 0.64, 1.28]):
        super().__init__()

        self.encoder_depths = encoder_depths
        self.encoder_num_heads = num_heads
        self.encoder_dims = encoder_dims
        self.local_radius = local_radius

        # token merging and positional embeddings
        self.token_embed = nn.ModuleList()
        self.encoder_pos_embeds = nn.ModuleList()
        for i in range(len(self.encoder_dims)):
            if i == 0:
                self.token_embed.append(Token_Embed(in_c=3, out_c=self.encoder_dims[i]))
            else:
                self.token_embed.append(Token_Embed(in_c=self.encoder_dims[i - 1], out_c=self.encoder_dims[i]))
            
            self.encoder_pos_embeds.append(nn.Sequential(
                            nn.Linear(3, self.encoder_dims[i]),
                            nn.GELU(),
                            nn.Linear(self.encoder_dims[i], self.encoder_dims[i]),
                        ))

        # encoder blocks
        self.encoder_blocks = nn.ModuleList()

        depth_count = 0
        dpr = [x.item() for x in torch.linspace(0, 0.1, sum(self.encoder_depths))]
        for i in range(len(self.encoder_depths)):
            self.encoder_blocks.append(Encoder_Block(
                            embed_dim=self.encoder_dims[i],
                            depth=self.encoder_depths[i],
                            drop_path_rate=dpr[depth_count: depth_count + self.encoder_depths[i]],
                            num_heads=self.encoder_num_heads,
                        ))
            depth_count += self.encoder_depths[i]

        self.encoder_norms = nn.ModuleList()
        for i in range(len(self.encoder_depths)):
            self.encoder_norms.append(nn.LayerNorm(self.encoder_dims[i]))

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def local_att_mask(self, xyz, radius, dist=None):
        with torch.no_grad():
            if dist is None or dist.shape[1] != xyz.shape[1]:
                dist = torch.cdist(xyz, xyz, p=2)
            # entries that are True in the mask do not contribute to self-attention
            # so points outside the radius are not considered
            mask = dist >= radius
        return mask, dist

    def forward(self, neighborhoods, centers, idxs, eval=False):
        # hierarchical encoding
        x_vis_list = []
        xyz_dist = None
        for i in range(len(centers)):
            # 1st-layer encoder, conduct token embedding
            if i == 0:
                group_input_tokens = self.token_embed[i](neighborhoods[0])
            # intermediate layers, conduct token merging
            else:
                b, g1, _ = x_vis.shape
                b, g2, k2, _ = neighborhoods[i].shape
                x_vis_neighborhoods = x_vis.reshape(b * g1, -1)[idxs[i], :].reshape(b, g2, k2, -1)
                group_input_tokens = self.token_embed[i](x_vis_neighborhoods)
            
            if self.local_radius[i] > 0:
                mask_radius, xyz_dist = self.local_att_mask(centers[i], self.local_radius[i], xyz_dist)
                mask_vis_att = mask_radius 
            else:
                mask_vis_att = None

            pos = self.encoder_pos_embeds[i](centers[i])
            x_vis = self.encoder_blocks[i](group_input_tokens, pos, mask_vis_att)
            x_vis_list.append(x_vis)

        for i in range(len(x_vis_list)):
            x_vis_list[i] = self.encoder_norms[i](x_vis_list[i]).transpose(-1, -2).contiguous()
        return x_vis_list

# finetune model
class Point_M2AE_SEG(nn.Module):
    def __init__(self, cls_dim):
        super().__init__()
        self.trans_dim = 384
        self.group_sizes = [16, 8, 8]
        self.num_groups = [512, 256, 64]
        self.cls_dim = cls_dim
        self.encoder_dims = [96, 192, 384]

        self.group_dividers = nn.ModuleList()
        for i in range(len(self.group_sizes)):
            self.group_dividers.append(Group(num_group=self.num_groups[i], group_size=self.group_sizes[i]))

        # hierarchical encoder
        self.h_encoder = H_Encoder_seg()

        self.label_conv = nn.Sequential(nn.Conv1d(16, 64, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(64),
                                   nn.LeakyReLU(0.2))

        self.propagations = nn.ModuleList()
        for i in range(3):
            self.propagations.append(PointNetFeaturePropagation_(in_channel=self.encoder_dims[i] + 3, mlp=[self.trans_dim * 4, 1024]))

        self.convs1 = nn.Conv1d(6208, 1024, 1)
        self.dp1 = nn.Dropout(0.5)
        self.convs2 = nn.Conv1d(1024, 512, 1)
        self.convs3 = nn.Conv1d(512, 256, 1)
        self.convs4 = nn.Conv1d(256, self.cls_dim, 1)
        self.bns1 = nn.BatchNorm1d(1024)
        self.bns2 = nn.BatchNorm1d(512)
        self.bns3 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def load_model_from_ckpt(self, ckpt_path):
        state_dict = torch.load(ckpt_path)
        incompatible = self.load_state_dict(state_dict['base_model'], strict=False)
        if incompatible.missing_keys:
            print_log('missing_keys', logger='Point_M2AE_ModelNet40')
            print_log(
                get_missing_parameters_message(incompatible.missing_keys),
                logger='Point_M2AE_ModelNet40'
            )
        if incompatible.unexpected_keys:
            print_log('unexpected_keys', logger='Point_M2AE_ModelNet40')
            print_log(
                get_unexpected_parameters_message(incompatible.unexpected_keys),
                logger='Point_M2AE_ModelNet40'
            )

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, pts, cls_label):
        B, C, N = pts.shape
        pts = pts.transpose(-1, -2).contiguous() # B N 3
        # divide the point cloud in the same form. This is important
        
        neighborhoods, centers, idxs = [], [], []
        for i in range(len(self.group_dividers)):
            if i == 0:
                neighborhood, center, idx = self.group_dividers[i](pts)
            else:
                neighborhood, center, idx = self.group_dividers[i](center)
            neighborhoods.append(neighborhood)
            centers.append(center)
            idxs.append(idx)  # b*g*k

        # hierarchical encoder
        x_vis_list = self.h_encoder(neighborhoods, centers, idxs, eval=True)

        for i in range(len(x_vis_list)):
            x_vis_list[i] = self.propagations[i](pts.transpose(-1, -2), centers[i].transpose(-1, -2), pts.transpose(-1, -2), x_vis_list[i])
            
        x = torch.cat((x_vis_list[0], x_vis_list[1], x_vis_list[2]), dim=1)  # 96 + 192 + 384
        x_max = torch.max(x, 2)[0]
        x_avg = torch.mean(x, 2)
        x_max_feature = x_max.view(B, -1).unsqueeze(-1).repeat(1, 1, N)
        x_avg_feature = x_avg.view(B, -1).unsqueeze(-1).repeat(1, 1, N)
        cls_label_one_hot = cls_label.view(B, 16, 1)
        cls_label_feature = self.label_conv(cls_label_one_hot).repeat(1, 1, N)
        x_global_feature = torch.cat((x_max_feature + x_avg_feature, cls_label_feature), 1) # 672 * 2 + 64

        x = torch.cat((x_global_feature, x), 1)
        x = self.relu(self.bns1(self.convs1(x)))
        x = self.dp1(x)
        x = self.relu(self.bns2(self.convs2(x)))
        x = self.relu(self.bns3(self.convs3(x)))
        x = self.convs4(x)
        x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)
        return x

class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target):
        total_loss = F.nll_loss(pred, target)
        return total_loss