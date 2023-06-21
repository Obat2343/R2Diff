import torch
import torchvision
import torch.nn as nn

from einops import rearrange, reduce, repeat

from .base_module import Flatten, LinearBlock, Transformer_Block
from .feature_extractor import Image_feature_extractor_model

class Predictor(nn.Module):
    def __init__(self, extractor_name, predictor_name,  query_emb_dim, query_list, query_dims, drop=0., img_emb_dim=0, num_attn_block=4):
        super().__init__()
        
        self.ife = Image_feature_extractor_model(extractor_name, img_emb_dim, 1) # extract low dim feature from dense img feature
        
        if img_emb_dim == 0:
            img_emb_dim = query_emb_dim

        self.predictor = build_predictor(predictor_name, query_emb_dim, query_list, query_dims, attn_block=num_attn_block, head=4, act="gelu", norm="none")
        self.query_emb = HIBC_query_emb_model(query_list, query_dims, query_emb_dim, drop=drop, act="gelu")
        self.predictor_name = predictor_name
        
        if 'softargmax_feature' == extractor_name:
            self.pos_emb = softarg_posemb(query_emb_dim)
        else:
            self.pos_emb = nn.Identity()
    
    def forward(self, img_emb, query):
        """
        img_emb: torch.tensor -> input image feature, shape(B,C,H,W)
        y: dict[str: torch.tensor] -> query
        """
        device = img_emb.device
        uv = query['uv'].to(device)
        
        img_feature_dict, debug_info = self.ife(img_emb, uv)
        pos_emb = self.pos_emb([img_feature_dict, uv])
        query_emb = self.query_emb(query)
        
        output_dict, pred_info = self.predictor(img_feature_dict, query_emb, pos_emb=pos_emb, query=query)
        
        for key in pred_info.keys():
            if key not in debug_info.keys():
                debug_info[key] = pred_info[key]
        
        return output_dict, debug_info

def build_predictor(name, query_emb_dim, query_list, query_dims, attn_block=4, head=4, act="gelu", norm="none"):
    if name == "EBM_Transformer_with_img_and_pose_feature":
        predictor = EBM_Transformer_with_img_and_pose_feature(query_emb_dim, attn_block=attn_block, head=head, act=act, norm=norm)
    elif name == "EBM_Transformer_with_cat_feature":
        predictor = EBM_Transformer_with_cat_feature(query_emb_dim, attn_block=attn_block, head=head, act=act, norm=norm)
    elif name == "Regressor_Transformer_with_img_and_pose_feature":
        predictor = Regressor_Transformer_with_img_and_pose_feature(query_emb_dim, query_list, query_dims, attn_block=attn_block, head=head, act=act, norm=norm, grasp_activation="linear")
    elif name == "Regressor_Transformer_with_cat_feature":
        predictor = Regressor_Transformer_with_cat_feature(query_emb_dim, query_list, query_dims, attn_block=attn_block, head=head, act=act, norm=norm, grasp_activation="linear")
    else:
        raise ValueError("Invalid predictor")
    return predictor

class base(nn.Module):
    
    def pos_norm(self,pos,H,W):
        """
        norm position value from range(0, H) to range(-1,1)
        We assume H, W are image or feature width and height.
        """
        x_coords = pos[:,:,0]
        y_coords = pos[:,:,1]
        
        x_coords = (x_coords / W) * 2 - 1
        y_coords = (y_coords / H) * 2 - 1
        
        return torch.cat([torch.unsqueeze(x_coords, 2), torch.unsqueeze(y_coords, 2)], dim=2)
    
    def pos_denorm(self,pos_norm,H,W):
        x_coords = pos_norm[:,0]
        y_coords = pos_norm[:,1]
        
        x_coords = (x_coords + 1) / 2 * W
        y_coords = (y_coords + 1) / 2 * H
        
        return torch.cat([torch.unsqueeze(x_coords, 1), torch.unsqueeze(y_coords, 1)], dim=1)
    
    @staticmethod
    def activation_layer(name):
        if name == 'relu':
            layer = nn.ReLU()
        elif name == 'prelu':
            layer = nn.PReLU()
        elif name == 'lrelu':
            layer = nn.LeakyReLU(0.2)
        elif name == 'tanh':
            layer = nn.Tanh()
        elif name == 'sigmoid':
            layer = nn.Sigmoid()
        elif name == 'gelu':
            layer = nn.GELU()
        elif name == 'none':
            layer = nn.Identity()
        else:
            raise ValueError("Invalid activation")
        return layer

class EBM_Transformer_with_img_and_pose_feature(base):
    
    def __init__(self, query_emb_dim, attn_block=4, head=4, act="gelu", norm="none"):
        super().__init__()

        self.flat = Flatten()
        self.transformer = Transformer_Block(query_emb_dim, attn_block=attn_block, head=head)
        self.output_layer = torch.nn.Sequential(
                            LinearBlock(query_emb_dim, query_emb_dim, activation=act, norm=norm),
                            LinearBlock(query_emb_dim, query_emb_dim, activation=act, norm=norm),
                            LinearBlock(query_emb_dim, int(query_emb_dim / 2), activation=act, norm=norm),
                            LinearBlock(int(query_emb_dim / 2), int(query_emb_dim / 2), activation=act, norm=norm),
                            LinearBlock(int(query_emb_dim / 2), 1),
        )
        
    def forward(self, img_feature_dict, query_feature, pos_emb=None, query=None):
        img_feature = img_feature_dict["img_feature"]
        if img_feature.dim() == 3:
            img_feature = torch.unsqueeze(img_feature_dict["img_feature"], 2) # B, N, D -> B, N, 1, D
            S = 1
        elif img_feature.dim() == 4:
            _, _, S, _ = img_feature.shape
        else:
            raise ValueError(f"Invalid shape. Current shape is {img_feature.shape} but (Batch NumQuery Dim) is required")
        
        mixed_feature = torch.cat([img_feature, query_feature], 2) # B, N, S, D
        
        B, N, S, D = mixed_feature.shape
        mixed_feature = rearrange(mixed_feature, 'B N S D -> (B N) S D')
        mixed_feature = self.transformer(mixed_feature)
        mixed_feature = rearrange(mixed_feature, '(B N) S D -> B N S D',B=B,N=N)       
        mixed_feature = torch.mean(mixed_feature, 2) # B N D

        output = self.output_layer(mixed_feature) 
        output = torch.squeeze(output, 2)       
        return {"score": output}, {}

class EBM_Transformer_with_cat_feature(base):
    
    def __init__(self, query_emb_dim, attn_block=4, head=4, act="gelu", norm="none"):
        super().__init__()

        self.flat = Flatten()
        self.feature_mix_layer = torch.nn.Sequential(
                            LinearBlock(query_emb_dim * 2, query_emb_dim * 2, activation=act, norm=norm),
                            LinearBlock(query_emb_dim * 2, query_emb_dim * 2, activation=act, norm=norm),
                            LinearBlock(query_emb_dim * 2, query_emb_dim, activation=act, norm=norm),
        )
        self.transformer = Transformer_Block(query_emb_dim, attn_block=attn_block, head=head)
        self.output_layer = torch.nn.Sequential(
                            LinearBlock(query_emb_dim, query_emb_dim, activation=act, norm=norm),
                            LinearBlock(query_emb_dim, query_emb_dim, activation=act, norm=norm),
                            LinearBlock(query_emb_dim, int(query_emb_dim / 2), activation=act, norm=norm),
                            LinearBlock(int(query_emb_dim / 2), int(query_emb_dim / 2), activation=act, norm=norm),
                            LinearBlock(int(query_emb_dim / 2), 1, activation='none', norm='none')
        )
        
    def forward(self, img_feature_dict, query_feature, pos_emb=None, query=None):
        img_feature = img_feature_dict["img_feature"]
        if img_feature.dim() != 4:
            raise ValueError(f"Invalid shape. Current shape is {img_feature.shape} but (Batch NumQuery NumSequence Dim) is required")
        mixed_feature = torch.cat([img_feature, query_feature], 3) # B, N, S, 2D
        mixed_feature = self.feature_mix_layer(mixed_feature)
        
        B, N, S, D = mixed_feature.shape
        mixed_feature = rearrange(mixed_feature, 'B N S D -> (B N) S D')
        mixed_feature = self.transformer(mixed_feature)
        mixed_feature = rearrange(mixed_feature, '(B N) S D -> B N S D',B=B,N=N)       
        mixed_feature = torch.mean(mixed_feature, 2) # B N D

        output = self.output_layer(mixed_feature)  
        output = torch.squeeze(output, 2)       
        return {"score": output}, {}

class Regressor_Transformer_with_img_and_pose_feature(base):
    
    def __init__(self, query_emb_dim, query_list, query_dims, attn_block=4, head=4, act="gelu", norm="none", grasp_activation="linear"):
        super().__init__()

        self.transformer = Transformer_Block(query_emb_dim, attn_block=attn_block, head=head)
        self.output_layer = torch.nn.Sequential(
                            LinearBlock(query_emb_dim, query_emb_dim, activation=act, norm=norm),
                            LinearBlock(query_emb_dim, query_emb_dim, activation=act, norm=norm),
                            LinearBlock(query_emb_dim, int(query_emb_dim / 2), activation=act, norm=norm),
                            LinearBlock(int(query_emb_dim / 2), int(query_emb_dim / 2), activation=act, norm=norm),
        )
        
        module_dict = {}
        for key, dim in zip(query_list, query_dims):
            if key == "time":
                continue
            module_dict[key] = torch.nn.Sequential(
                            LinearBlock(int(query_emb_dim / 2), int(query_emb_dim / 2)),
                            LinearBlock(int(query_emb_dim / 2), dim))
        
        self.output_module_dict = torch.nn.ModuleDict(module_dict)

        if grasp_activation == "sigmoid":
            self.grasp_act = torch.nn.Sigmoid()
        elif grasp_activation == "linear":
            self.grasp_act = lambda input, min=0., max=1.: torch.clamp(input, min=min, max=max)

    def forward(self, img_feature_dict, query_feature, pos_emb=None, query=None):
        img_feature = img_feature_dict["img_feature"]
        if img_feature.dim() == 3:
            img_feature = torch.unsqueeze(img_feature_dict["img_feature"], 2) # B, N, D -> B, N, 1, D
            S = 1
        elif img_feature.dim() == 4:
            _, _, S, _ = img_feature.shape
        else:
            raise ValueError(f"Invalid shape. Current shape is {img_feature.shape} but (Batch NumQuery Dim) is required")
        mixed_feature = torch.cat([img_feature, query_feature], 2) # B, N, 1+S, D
        
        B, N, _, _ = mixed_feature.shape
        mixed_feature = rearrange(mixed_feature, 'B N S D -> (B N) S D')
        mixed_feature = self.transformer(mixed_feature)
        mixed_feature = rearrange(mixed_feature, '(B N) S D -> B N S D',B=B,N=N)       

        mixed_feature = self.output_layer(mixed_feature[:,:,S:]) 

        pred_dict = {}
        for key in self.output_module_dict.keys():
            pred_dict[key] = query[key] + self.output_module_dict[key](mixed_feature)
            if key == "grasp_state":
                pred_dict["grasp_state"] = self.grasp_act(pred_dict["grasp_state"])

        return pred_dict, {}

class Regressor_Transformer_with_cat_feature(base):
    
    def __init__(self, query_emb_dim, query_list, query_dims, attn_block=4, head=4, act="gelu", norm="none", grasp_activation="linear"):
        super().__init__()

        self.flat = Flatten()
        self.feature_mix_layer = torch.nn.Sequential(
                            LinearBlock(query_emb_dim * 2, query_emb_dim * 2, activation=act, norm=norm),
                            LinearBlock(query_emb_dim * 2, query_emb_dim * 2, activation=act, norm=norm),
                            LinearBlock(query_emb_dim * 2, query_emb_dim, activation=act, norm=norm),
        )
        self.transformer = Transformer_Block(query_emb_dim, attn_block=attn_block, head=head)
        self.output_layer = torch.nn.Sequential(
                            LinearBlock(query_emb_dim, query_emb_dim, activation=act, norm=norm),
                            LinearBlock(query_emb_dim, query_emb_dim, activation=act, norm=norm),
                            LinearBlock(query_emb_dim, int(query_emb_dim / 2), activation=act, norm=norm),
                            LinearBlock(int(query_emb_dim / 2), int(query_emb_dim / 2), activation=act, norm=norm),
        )

        module_dict = {}
        for key, dim in zip(query_list, query_dims):
            if key == "time":
                continue
            module_dict[key] = torch.nn.Sequential(
                            LinearBlock(int(query_emb_dim / 2), int(query_emb_dim / 2)),
                            LinearBlock(int(query_emb_dim / 2), dim))
        
        self.output_module_dict = torch.nn.ModuleDict(module_dict)

        if grasp_activation == "sigmoid":
            self.grasp_act = torch.nn.Sigmoid()
        elif grasp_activation == "linear":
            self.grasp_act = lambda input, min=0., max=1.: torch.clamp(input, min=min, max=max)
        
    def forward(self, img_feature_dict, query_feature, pos_emb=None, query=None):
        img_feature = img_feature_dict["img_feature"]
        if img_feature.dim() != 4:
            raise ValueError(f"Invalid shape. Current shape is {img_feature.shape} but (Batch NumQuery NumSequence Dim) is required")
        mixed_feature = torch.cat([img_feature, query_feature], 3) # B, N, S, 2D
        mixed_feature = self.feature_mix_layer(mixed_feature)
        
        B, N, S, D = mixed_feature.shape
        mixed_feature = rearrange(mixed_feature, 'B N S D -> (B N) S D')
        mixed_feature = self.transformer(mixed_feature)
        mixed_feature = rearrange(mixed_feature, '(B N) S D -> B N S D',B=B,N=N)       

        mixed_feature = self.output_layer(mixed_feature) 

        pred_dict = {}
        for key in self.output_module_dict.keys():
            pred_dict[key] = query[key] + self.output_module_dict[key](mixed_feature)
            if key == "grasp_state":
                pred_dict["grasp_state"] = self.grasp_act(pred_dict["grasp_state"])

        return pred_dict, {}

# class predictor(nn.Module):
#     def __init__(self, extractor_list, predictor_name, img_size, query_emb_dim, down_scale, query_list, query_dims, drop=0., linear_layer_num=3, img_emb_dim=0, obs_emb_dim=0):
#         super().__init__()
#         size = (img_size[0] / down_scale, img_size[1] / down_scale)
        
#         self.ife = Image_feature_extractor_model(extractor_list, img_emb_dim, down_scale) # extract low dim feature from dense img feature
        
#         if img_emb_dim == 0:
#             img_emb_dim = query_emb_dim
#         if obs_emb_dim == 0:
#             obs_emb_dim = query_emb_dim

#         self.predictor = build_predictor(predictor_name, img_emb_dim, query_emb_dim, obs_emb_dim, down_scale, size, query_list, act="gelu", linear_layer_num=linear_layer_num)
#         self.query_emb = query_emb_model(query_list, query_dims, query_emb_dim, drop=drop, act="gelu")
#         self.down_scale = down_scale
#         self.predictor_name = predictor_name
        
#         if 'softargmax_feature' in extractor_list:
#             self.pos_emb = softarg_posemb(query_emb_dim)
#         else:
#             self.pos_emb = nn.Identity()
    
#     def forward(self, img_emb, query, obs_emb):
#         """
#         img_emb: torch.tensor -> input image feature, shape(B,C,H,W)
#         y: dict[str: torch.tensor] -> query
#         """
#         device = img_emb.device
#         uv = query['uv'].to(device)
#         B, N, _ = uv.shape
        
#         extract_dict, debug_info = self.ife(img_emb, uv)
#         pos_emb = self.pos_emb([extract_dict, uv])
#         query_emb = self.query_emb(query)
        
#         obs_emb = repeat(obs_emb, "B D -> B N D", N=N)
#         output_dict, pred_info = self.predictor(extract_dict, pos_emb, query_emb, N, obs_emb)
        
#         for key in pred_info.keys():
#             if key not in debug_info.keys():
#                 debug_info[key] = pred_info[key]
        
#         return output_dict, debug_info

# def build_predictor(name, img_query_dim, query_emb_dim, obs_emb_dim, down_scale, size, query_list, act="gelu", linear_layer_num=3):
#     if name == 'coords_linear':
#         predictor = coords_linear(query_emb_dim, down_scale, size, query_list, act, linear_layer_num) # TODO add obs emb
#         raise ValueError("TODO: now coding")
#     elif name == 'coords_IBC_all':
#         predictor = coords_IBC_all(query_emb_dim, down_scale, size, query_list, act, linear_layer_num) # TODO add obs emb
#         raise ValueError("TODO: now coding")
#     elif name == 'coords_IBC_sep':
#         predictor = coords_IBC_sep(query_emb_dim, down_scale, size, query_list, act, linear_layer_num) # TODO add obs emb
#         raise ValueError("TODO: now coding")
#     elif name == 'feature_IBC_all':
#         predictor = feature_IBC_all(img_query_dim, query_emb_dim, obs_emb_dim, query_list, act, linear_layer_num)
#         raise ValueError("TODO: now coding")
#     elif name == 'feature_IBC_sep':
#         predictor = feature_IBC_sep(img_query_dim, query_emb_dim, obs_emb_dim, query_list, act, linear_layer_num)
#         raise ValueError("TODO: now coding")
#     elif name == 'feature_IBC_sep_dot':
#         predictor = feature_IBC_sep_dot(img_query_dim, query_emb_dim, query_list, act, linear_layer_num) # TODO add obs emb
#         raise ValueError("TODO: now coding")
#     elif name == 'softargmax_linear':
#         predictor = softargmax_feature_linear(img_query_dim, down_scale, size, act="gelu", linear_layer_num=3)
#         raise ValueError("TODO: now coding")
#     else:
#         raise ValueError("Invalid predictor")
        
#     return predictor

# class coords_linear(base):
    
#     def __init__(self, query_emb_dim, output_dim, down_scale, size, query_list, num_vec, act="gelu", linear_layer_num=3):
#         super().__init__()
#         input_dim = (query_emb_dim * len(query_list)) + (2 * num_vec)
#         self.flat = Flatten()
#         model_list = []
#         for _ in range(linear_layer_num - 1):
#             model_list.append(nn.Linear(input_dim, input_dim))
#             model_list.append(self.activation_layer(act))
#         model_list.append(nn.Linear(input_dim, output_dim))

#         self.linear = torch.nn.Sequential(*model_list)
        
#         self.size = size
#         self.down_scale = down_scale
        
#     def forward(self, extractor_dict, pos_emb, query_emb, N):
#         coords = extractor_dict["key_coords"]
#         coords = self.flat(coords)
#         coords = repeat(coords, 'B K -> B N K', N=N)
        
#         keys = sorted(list(query_emb.keys()))
#         query_vec = torch.cat([query_emb[key] for key in keys], 2)
        
#         input_vec = torch.cat([coords, query_vec], 2)
#         output = self.linear(input_vec)
#         output = self.pos_denorm(output, self.size[0], self.size[1])
        
#         return {"all": output * self.down_scale}, {}

# class coords_IBC_all(base):
    
#     def __init__(self, query_emb_dim, down_scale, size, query_list, num_vec, act="gelu", linear_layer_num=3):
#         super().__init__()
#         input_dim = (query_emb_dim * len(query_list)) + (2 * num_vec)
#         self.flat = Flatten()
#         model_list = []
#         for _ in range(linear_layer_num - 1):
#             model_list.append(nn.Linear(input_dim, input_dim))
#             model_list.append(self.activation_layer(act))
#         model_list.append(nn.Linear(input_dim, 1))

#         self.linear = torch.nn.Sequential(*model_list)
        
#         self.down_scale = down_scale
#         self.size = size
        
#     def forward(self, extractor_dict, pos_emb, query_emb, N):
#         coords = extractor_dict["key_coords"]
#         coords = self.flat(coords)
#         keys = sorted(list(query_emb.keys()))
#         query_vec = torch.cat([query_emb[key] for key in keys], 2)
        
#         coords = repeat(coords, 'B K -> B N K', N=N)
#         input_vec = torch.cat([coords, query_vec], 2)
#         output = self.linear(input_vec)
#         output = rearrange(output, 'B N V -> B (N V)')
        
#         return {"all": output}, {}
    
# class coords_IBC_sep(base):
    
#     def __init__(self, query_emb_dim, down_scale, size, query_list, num_vec, act="gelu"):
#         super().__init__()
#         input_dim = query_emb_dim + (2 * num_vec)
#         self.flat = Flatten()
        
#         model_dict = {}
#         for query in query_list:
#             model_dict[query] = self.build_linear(input_dim, 1, act)
#         self.inference_model = torch.nn.ModuleDict(model_dict)
        
#         self.down_scale = down_scale
#         self.size = size
        
#     def forward(self, extractor_dict, pos_emb, query_emb, N):
#         output_log_dict = {}
#         output_dict = {}
#         coords = extractor_dict["key_coords"] # B K 2
#         coords = self.flat(coords) # B (K 2)
#         coords = repeat(coords, 'B K -> B N K', N=N)
#         keys = sorted(list(query_emb.keys()))

#         for i, query_key in enumerate(keys):
#             input_vec = torch.cat([coords, query_emb[query_key]], 2)
#             output = self.inference_model[query_key](input_vec)
#             output = rearrange(output, 'B N V -> B (N V)')
#             output_dict[query_key] = output
#             output_log_dict[f"sep_map_{query_key}"] =  output.detach().cpu()
#             if i == 0:
#                 final_output = output
#             else:
#                 final_output = final_output + output
        
#         output_dict["all"] = final_output
#         return output_dict, output_log_dict
    
#     def build_linear(self, input_dim, output_dim, act, linear_layer_num=3):
#         model_list = []
#         for _ in range(linear_layer_num - 1):
#             model_list.append(nn.Linear(input_dim, input_dim))
#             model_list.append(self.activation_layer(act))
#         model_list.append(nn.Linear(input_dim, output_dim))

#         return torch.nn.Sequential(*model_list)

# class feature_IBC_all(base):
    
#     def __init__(self, img_query_dim, query_emb_dim, obs_emb_dim, query_list, act="gelu", linear_layer_num=3):
#         super().__init__()
        
#         input_dim = img_query_dim + (query_emb_dim * len(query_list)) + obs_emb_dim
#         self.flat = Flatten()

#         model_list = []
#         for _ in range(linear_layer_num - 1):
#             model_list.append(nn.Linear(input_dim, input_dim))
#             model_list.append(self.activation_layer(act))
#         model_list.append(nn.Linear(input_dim, 1))

#         self.linear = torch.nn.Sequential(*model_list)
        
#     def forward(self, extractor_dict, pos_emb, query_emb, N, obs_emb):
#         img_query = extractor_dict["img_query"] # B N C
#         keys = sorted(list(query_emb.keys()))
#         query_vec = torch.cat([query_emb[key] for key in keys], 2) # B N C
#         input_vec = torch.cat([img_query, query_vec, obs_emb], 2)
#         output = self.linear(input_vec)
#         output = rearrange(output, 'B N V -> B (N V)')
#         return {"all": output}, {}
    
# class feature_IBC_sep(base):
    
#     def __init__(self, img_query_dim, query_emb_dim, query_list, act="gelu"):
#         super().__init__()
        
#         input_dim = img_query_dim + query_emb_dim
#         self.flat = Flatten()
        
#         model_dict = {}
#         for query in query_list:
#             model_dict[query] = self.build_linear(input_dim, 1, act)
#         self.inference_model = torch.nn.ModuleDict(model_dict)
        
#     def forward(self, extractor_dict, pos_emb, query_emb, N, obs_emb):
#         output_log_dict = {}
#         output_dict = {}
#         img_query = extractor_dict["img_query"] # B N C
        
#         keys = sorted(list(query_emb.keys()))
#         for i, query_key in enumerate(keys):
#             input_vec = torch.cat([img_query, query_emb[query_key], obs_emb], 2)
#             output = self.inference_model[query_key](input_vec)
#             output = rearrange(output, 'B N V -> B (N V)')

#             output_dict[query_key] = output
#             output_log_dict[f"sep_map_{query_key}"] =  output.detach().cpu()
#             if i == 0:
#                 final_output = output
#             else:
#                 final_output = final_output + output
        
#         output_dict["all"] = final_output
#         return output_dict, output_log_dict
    
#     def build_linear(self, input_dim, output_dim, act, linear_layer_num=3):
#         model_list = []
#         for _ in range(linear_layer_num - 1):
#             model_list.append(nn.Linear(input_dim, input_dim))
#             model_list.append(self.activation_layer(act))
#         model_list.append(nn.Linear(input_dim, output_dim))

#         return torch.nn.Sequential(*model_list)

# class feature_IBC_sep_dot(base):
    
#     def __init__(self, img_query_dim, query_emb_dim, query_list, act="gelu"):
#         super().__init__()
        
#         self.flat = Flatten()
        
#         query_model_dict = {}
#         feature_model_dict = {}
#         for query in query_list:
#             query_model_dict[query] = self.build_linear(query_emb_dim, img_query_dim, act)
#             feature_model_dict[query] = self.build_linear(img_query_dim, img_query_dim, act)
#         self.query_model = torch.nn.ModuleDict(query_model_dict)
#         self.feature_model = torch.nn.ModuleDict(feature_model_dict)
        
#     def forward(self, extractor_dict, pos_emb, query_emb, N, obs_emb):
#         output_log_dict = {}
#         output_dict = {}
#         img_query = extractor_dict["img_query"] # B N C
        
#         keys = sorted(list(query_emb.keys()))
#         for i, query_key in enumerate(keys):
#             feature_vec = self.feature_model[query_key](img_query) # B N C
#             query_vec = self.query_model[query_key](query_emb[query_key]) # B N C
#             output = -(feature_vec*query_vec).sum(2)
#             output_dict[query_key] = output
#             output_log_dict[f"sep_map_{query_key}"] =  output.detach().cpu()
#             if i == 0:
#                 final_output = output
#             else:
#                 final_output = final_output + output
        
#         output_dict["all"] = final_output
#         return output_dict, output_log_dict
    
#     def build_linear(self, input_dim, output_dim, act, linear_layer_num=3):
#         model_list = []
#         for _ in range(linear_layer_num - 1):
#             model_list.append(nn.Linear(input_dim, input_dim))
#             model_list.append(self.activation_layer(act))
#         model_list.append(nn.Linear(input_dim, output_dim))

#         return torch.nn.Sequential(*model_list)


# class softargmax_feature_linear(base):

#     def __init__(self, feature_dim, down_scale, size, act="gelu", linear_layer_num=3):
#         super().__init__()

#         model_list = []
#         for _ in range(linear_layer_num):
#             model_list.append(nn.Linear(feature_dim, feature_dim))
#             model_list.append(self.activation_layer(act))
#         self.linear = torch.nn.Sequential(*model_list)
        
#         self.z = nn.Linear(feature_dim, 1)
#         self.rot = nn.Linear(feature_dim, 6)
#         self.grasp = nn.Linear(feature_dim, 1)

#         self.size = size
#         self.down_scale = down_scale
        
#     def forward(self, extractor_dict, pos_emb, query_emb, N, obs_emb):
#         coords = extractor_dict["key_coords"]
#         uv = self.pos_denorm(coords[:,0], self.size[0], self.size[1])

#         feature = extractor_dict["key_feature"]        
#         feature = self.linear(feature)

#         z = self.z(feature)
#         rot = self.rot(feature)
#         grasp = self.grasp(feature)
        
#         return {"uv": uv, "z": z, "rotation_6d": rot, "grasp_state": grasp}, {}

##### TODO #####

class feature_IABC_all(base):
    
    def __init__(self, img_query_dim, query_emb_dim, query_list, act="gelu"):
        super().__init__()
        
        input_dim = img_query_dim + (query_emb_dim * len(query_list))
        self.flat = Flatten()    
        self.linear = torch.nn.Sequential(
                nn.Linear(input_dim, input_dim),
                self.activation_layer(act),
                nn.Linear(input_dim, input_dim))
        
    def forward(self, extractor_dict, pos_emb, query_emb, N):
        img_query = extractor_dict["img_query"] # B N C

        keys = sorted(list(query_emb.keys()))
        query_vec = torch.cat([query_emb[key] for key in keys], 2) # B N C
        
        input_vec = torch.cat([img_query, query_vec], 2)
        key_feature = extractor_dict["key_feature"]
        
        value_pos_emb = self.pos_emb(self.pos_norm(coords,H,W),self.pos_norm(y,H,W))
        value_pos_emb = rearrange(value_pos_emb, 'B N K c -> (B N) c K')
        value_pos_emb = repeat(value_pos_emb, 'B c K -> B h c K', h=self.heads)
        
        for atten_block in self.attention_list:
            query, coef = atten_block(query, x, value_pos_emb) # coef shape: B N h k
        
        output = self.linear(input_vec)
        output = rearrange(output, 'B N V -> B (N V)')
        return output, {}
    
class feature_IABC_sep(base):
    
    def __init__(self, img_query_dim, query_emb_dim, query_list, act="gelu"):
        super().__init__()
        
        input_dim = img_query_dim + query_emb_dim
        self.flat = Flatten()
        
        model_dict = {}
        for query in query_list:
            model_dict[query] = self.build_linear(input_dim, 1, act)
        self.inference_model = torch.nn.ModuleDict(model_dict)
        
    def forward(self, extractor_dict, pos_emb, query_emb, N):
        output_dict = {}
        img_query = extractor_dict["img_query"] # B N C
        keys = sorted(list(query_emb.keys()))

        for i, query_key in enumerate(keys):
            input_vec = torch.cat([img_query, query_emb[query_key]], 2)
            output = self.inference_model[query_key](input_vec)
            output = rearrange(output, 'B N V -> B (N V)')
            if i == 0:
                final_output = output
            else:
                final_output = final_output + output
            output_dict[f"sep_map_{query_key}"] =  output.detach().cpu()
        return final_output, output_dict
    
    def build_linear(self, input_dim, output_dim, act):
        model = self.linear = torch.nn.Sequential(
                nn.Linear(input_dim, input_dim),
                self.activation_layer(act),
                nn.Linear(input_dim, input_dim),
                self.activation_layer(act),
                nn.Linear(input_dim, output_dim))
        return model

########################################################################

class softarg_posemb(torch.nn.Module):
    def __init__(self,emb_dim,act='gelu'):
        super().__init__()
        self.query_emb = torch.nn.Sequential(
                nn.Linear(2, emb_dim),
                self.activation_layer(act),
                nn.Linear(emb_dim, emb_dim),
                self.activation_layer(act),
                nn.Linear(emb_dim, emb_dim))
        
    def forward(self,x):
        """
        x: dict
        y: query coords: B, N, 2
        """
        x_dict, query_coords = x[0], x[1]
        key_coords = x_dict["key_coords"]  # B, K, 2
        B, K, _ = key_coords.shape
        _, N, _ = query_coords.shape
        
        x = repeat(key_coords, 'B K P -> B N K P', N=N)
        y = repeat(query_coords, 'B N P -> B N K P', K=K)
        
        diff = x - y
        return self.query_emb(diff) # B N K D
        
    @staticmethod
    def activation_layer(name):
        if name == 'relu':
            layer = nn.ReLU()
        elif name == 'prelu':
            layer = nn.PReLU()
        elif name == 'lrelu':
            layer = nn.LeakyReLU(0.2)
        elif name == 'tanh':
            layer = nn.Tanh()
        elif name == 'sigmoid':
            layer = nn.Sigmoid()
        elif name == 'gelu':
            layer = nn.GELU()
        elif name == 'none':
            layer = nn.Identity()
        else:
            raise ValueError("Invalid activation")
        return layer

class HIBC_query_emb_model(base):
    def __init__(self, query_keys, query_dims, emb_dim, drop=0., act="gelu"):
        """
        Input:
        query_keys: list of query keys you want to use. Other queries will be ignored in the forward process.
        query_dims: list of dim of each query that you want to use.
        emb_dim: dimension of output feature (embedded query)
        """
        super().__init__()

        self.register_query_keys = query_keys
        query_total_dim = sum(query_dims)
        self.query_emb_model = self.make_linear_model(query_total_dim, emb_dim, act, drop)

    def forward(self, querys):
        """
        Input
        querys: dict
            key:
                str
            value:
                torch.tensor: shape -> (B, N, S, D), B -> Batch Size, N, Num of query in each batch, S -> Sequence Length, D -> Dim of each values
        Output:
        query_emb: torch.tensor: shape -> (B, N, S, QD), QD -> emb_dim
        """
        keys = list(querys.keys())
        keys.sort()

        query_list = []
        for key in keys:
            if key in self.register_query_keys:
                query_list.append(querys[key])
        
        query_cat = torch.cat(query_list, 3)
        query_emb = self.query_emb_model(query_cat)
        return query_emb
        
    def make_linear_model(self, input_dim, output_dim, act, drop):
        model = torch.nn.Sequential(
            nn.Linear(input_dim, output_dim),
            self.activation_layer(act),
            nn.Dropout(drop),
            nn.Linear(output_dim, output_dim * 2),
            self.activation_layer(act),
            nn.Dropout(drop),
            nn.Linear(output_dim * 2, output_dim * 2),
            self.activation_layer(act),
            nn.Dropout(drop),
            nn.Linear(output_dim * 2, output_dim))
        return model

class query_emb_model(base):
    
    def __init__(self, query_keys, query_dims, emb_dim, drop=0., act="gelu"):
        super().__init__()
        query_dict = {}
            
        for i, key in enumerate(query_keys):
            query_dict[key] = self.make_linear_model(query_dims[i], emb_dim, act, drop)
        
        self.query_emb = torch.nn.ModuleDict(query_dict)
        self.register_query_keys = query_keys

    def forward(self, querys):
        query_emb = {}
        
        for key in querys.keys():
            if key in self.register_query_keys:
                query_emb[key] = self.query_emb[key](querys[key])
        
        return query_emb
        
    def make_linear_model(self, input_dim, output_dim, act, drop):
        model = torch.nn.Sequential(
            nn.Linear(input_dim, output_dim),
            self.activation_layer(act),
            nn.Dropout(drop),
            nn.Linear(output_dim, output_dim * 2),
            self.activation_layer(act),
            nn.Dropout(drop),
            nn.Linear(output_dim * 2, output_dim))
        return model