# import torch
# import clip
# from PIL import Image

# device = "cuda" if torch.cuda.is_available() else "cpu"
# # model, preprocess = clip.load("ViT-B/16", device=device)
# # torch.save(model.visual.state_dict(), "model/clip_visual_16.pth")
# # image = preprocess(Image.open("CLIP.png")).unsqueeze(0).to(device)
# # text = clip.tokenize(["a diagram", "a dog", "a cat"]).to(device)

# # with torch.no_grad():
# #     image_features = model.encode_image(image)
# #     text_features = model.encode_text(text)
    
# #     logits_per_image, logits_per_text = model(image, text)
# #     probs = logits_per_image.softmax(dim=-1).cpu().numpy()

# # print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]





# param_dict = torch.load("model/clip_visual_B_16.pth")

# # Print the state dict
# for param_tensor in param_dict:
#     print(param_tensor, "\t", param_dict[param_tensor].size())


# param_dict = torch.load("model/vit_base.pth")

# # Print the state dict
# for param_tensor in param_dict:
#     print(param_tensor, "\t", param_dict[param_tensor].size())

from collections import OrderedDict
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import math
from torch.nn import init



class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


# #####################################################################
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.zeros_(m.bias.data)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.01)
        init.zeros_(m.bias.data)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0, 0.001)
        if m.bias:
            init.zeros_(m.bias.data)


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class VisionTransformer(nn.Module):
    def __init__(self, class_num, dataset, h_resolution: int, w_resolution: int, patch_size: int, stride_size: int, width: int, layers: int, heads: int, output_dim: int):
        super().__init__()
        self.class_num = class_num
        self.dataset = dataset
        self.h_resolution = h_resolution
        self.w_resolution = w_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=stride_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn(h_resolution*w_resolution + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads)

        self.ln_post = LayerNorm(width)
        # self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

        pool_dim = 768
        # Bottleneck
        self.bottleneck = nn.BatchNorm1d(pool_dim)
        self.bottleneck.bias.requires_grad_(False)  # no shift
        self.bottleneck.apply(weights_init_kaiming)

        # Classifier
        self.classifier = nn.Linear(pool_dim, class_num, bias=False)
        self.classifier.apply(weights_init_classifier)

    def forward(self, x: torch.Tensor, cv_emb = None):
        # patch embedding
        x = self.conv1(x)  # shape = [*, width, grid, grid]  torch.Size([32, 768, 16, 8])
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]  torch.Size([32, 768, 128])
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]  torch.Size([32, 128, 768])
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width] torch.Size([32, 129, 768])
        if cv_emb != None: 
            x[:,0] = x[:,0] + cv_emb
        x = x + self.positional_embedding.to(x.dtype) #torch.Size([32, 129, 768])
        # LN 层
        x = self.ln_pre(x)
        
        x = x.permute(1, 0, 2)  # NLD -> LND #torch.Size([129, 32, 768])
        x = self.transformer(x) #torch.Size([129, 32, 768])
        x = x.permute(1, 0, 2)  # LND -> NLD #torch.Size([129, 32, 768])

        x = self.ln_post(x[:, 0, :]) #torch.Size([32, 768])

        x_att = self.bottleneck(x) #torch.Size([32, 768])
        out = self.classifier(x_att)
        # if self.proj is not None:
        #     x = x @ self.proj

        return x, x_att, out

    
    def load_param(self, model_path):
        param_dict = torch.load(model_path, map_location='cpu')
        count = 0
        if 'model' in param_dict:
            param_dict = param_dict['model']
        if 'state_dict' in param_dict:
            param_dict = param_dict['state_dict']
        
        for k, v in param_dict.items():
            if 'proj' == k:  #注意'in'和'=='的区别
                continue
            # if 'fc' in k:
            #     continue
            if 'head' in k or 'dist' in k:

                continue
            if 'conv1.weight' in k: 
                print(self.conv1.weight.shape)
                print(v.shape)
            if 'conv1.weight' in k and len(v.shape) < 4:
                # For old models that I trained prior to conv based patchification
                O, I, H, W = self.conv1.weight.shape
                v = v.reshape(O, -1, H, W)
            elif k == 'positional_embedding' and v.shape != self.positional_embedding.shape:
                # v.shape: torch.Size([197, 768]), self.positional_embedding.shape: torch.Size([129, 768])
                # To resize pos embedding when using model at different size from pretrained weights
                if 'distilled' in model_path:
                    print('distill need to choose right cls token in the pth')
                    v = torch.cat([v[:, 0:1], v[:, 2:]], dim=1)
                v = resize_pos_embed(v, self.positional_embedding, self.h_resolution, self.w_resolution)
            if k == 'transformer.resblocks.0.attn.in_proj_weight':
                print('transformer.resblocks.0.attn.in_proj_weight')
            if k == 'transformer.resblocks.0.attn.in_proj_bias':
                print('transformer.resblocks.0.attn.in_proj_bias')
            try:
                self.state_dict()[k].copy_(v)
                count += 1
            except:
                print('===========================ERROR=========================')
                print('shape do not match in k :{}: param_dict{} vs self.state_dict(){}'.format(k, v.shape, self.state_dict()[k].shape))
            print('Load %d / %d layers.'%(count,len(self.state_dict().keys())))
            print('k: ', k)
        print('=================================================================')
        print(len(param_dict))
        print(len(self.state_dict()))
        print(param_dict.keys())
        print(self.state_dict().keys())


def resize_pos_embed(posemb, posemb_new, hight, width):
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    # 给posemb和posemb_new加一个维度
    posemb = posemb.unsqueeze(0) #posemb:torch.Size([1, 197, 768])
    posemb_new = posemb_new.unsqueeze(0) #posemb_new:torch.Size([1, 129, 768])
    ntok_new = posemb_new.shape[1] #ntok_new: 129

    posemb_token, posemb_grid = posemb[:, :1], posemb[0, 1:]
    #posemb_token:torch.Size([1, 1, 768]), posemb_grid:torch.Size([196, 768])
    ntok_new -= 1

    gs_old = int(math.sqrt(len(posemb_grid))) #gs_old: 14
    print('Resized position embedding from size:{} to size: {} with height:{} width: {}'.format(posemb.shape, posemb_new.shape, hight, width))
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    # posemb_grid: torch.Size([1, 768, 14, 14])
    posemb_grid = F.interpolate(posemb_grid, size=(hight, width), mode='bilinear')
    # 进行采样，posemb_grid: torch.Size([1, 768, 16, 8])
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, hight * width, -1)
    # posemb_grid: torch.Size([1, 128, 768])
    posemb = torch.cat([posemb_token, posemb_grid], dim=1)
    # posemb: torch.Size([1, 129, 768])
    posemb = posemb.squeeze(0)
    # posemb: torch.Size([ 129, 768])
    return posemb

    
def vit_base_patch16_224_TransReID(class_num, dataset, h_resolution=256, w_resolution=128, patch_size=16, stride_size=16, width=768, layers=12, heads=12, output_dim=768):
    model = VisionTransformer(
        class_num = class_num,
        dataset =  dataset, 
        h_resolution = h_resolution//patch_size,
        w_resolution = w_resolution//patch_size,
        patch_size = patch_size,
        stride_size = patch_size,
        width = width,
        layers = layers,
        heads = heads,
        output_dim = output_dim)
    return model



