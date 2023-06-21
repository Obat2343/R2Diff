
import math
import copy
import random
import statistics
import torch
import numpy as np

from tqdm import tqdm
from einops import rearrange, repeat

from .base_module import LinearBlock
from .resnet_module import Resnet_Like_Decoder, Resnet_Like_Encoder
from ..dataset import RLBench_DMOEBM
from ..retrieval import Direct_Retrieval

class Forward_diffusion():
    
    def __init__(self, max_timesteps, start=1e-5, end=2e-2):
        self.max_timesteps = max_timesteps
        self.start_beta = start
        self.end_beta = end
        
        self.betas = torch.abs(torch.cat([torch.tensor([0.]), torch.linspace(start, end, max_timesteps)],0))
        alphas = 1. - self.betas
        alphas = torch.clip(alphas, 1e-8, 1)
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.nn.functional.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
        self.posterior_variance = self.betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

    def get_index_from_list(self, vals, t, x_shape):
        """ 
        Returns a specific index t of a passed list of values vals
        while considering the batch dimension.
        """
        batch_size = t.shape[0]
        out = vals.gather(-1, t.cpu())
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)
    
    def get_values_from_timestep_for_sampling(self, t, x_shape):
        """ 
        Returns a specific index t of a passed list of values vals
        while considering the batch dimension.
        """
        batch_size = t.shape[0]
        beta_t = self.get_index_from_list(self.betas, t, x_shape)
        sqrt_one_minus_alphas_cumprod_t = self.get_index_from_list(self.sqrt_one_minus_alphas_cumprod, t, x_shape)
        sqrt_recip_alphas_t = self.get_index_from_list(self.sqrt_recip_alphas, t, x_shape)
        posterior_variance_t = self.get_index_from_list(self.posterior_variance, t, x_shape)
        return beta_t, sqrt_one_minus_alphas_cumprod_t, sqrt_recip_alphas_t, posterior_variance_t
        
    def forward_sample(self, action_dict, t, device="cpu"):
        """ 
        Takes an image and a timestep as input and 
        returns the noisy version of it
        """
        noise_action_dict = {}
        shifted_action_dict = {}
        noise_dict = {}
        for key in action_dict.keys():
            noise = torch.randn_like(action_dict[key])
            
            sqrt_alphas_cumprod_t = self.get_index_from_list(self.sqrt_alphas_cumprod, t, action_dict[key].shape)
            sqrt_one_minus_alphas_cumprod_t = self.get_index_from_list(
                                                self.sqrt_one_minus_alphas_cumprod, t, action_dict[key].shape)
            
            shifted_action_dict[key] = sqrt_alphas_cumprod_t.to(device) * action_dict[key].to(device)
            noise_action_dict[key] = shifted_action_dict[key] + sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device)
            noise_dict[key] = noise.to(device)
        
        return noise_action_dict, noise_dict, shifted_action_dict

    def get_mean_and_std(self, action_dict, t, device="cpu"):
        std = {}
        mean = {}
        for key in action_dict.keys():
            sqrt_alphas_cumprod_t = self.get_index_from_list(self.sqrt_alphas_cumprod, t, action_dict[key].shape)
            sqrt_one_minus_alphas_cumprod_t = self.get_index_from_list(
                                                self.sqrt_one_minus_alphas_cumprod, t, action_dict[key].shape)
            
            mean[key] = sqrt_alphas_cumprod_t.to(device) * action_dict[key].to(device)
            std[key] = sqrt_one_minus_alphas_cumprod_t.to(device)
        
        return mean, std
    
    def get_step(self, diff):
        alpha_values = self.sqrt_one_minus_alphas_cumprod / self.sqrt_alphas_cumprod
        step = torch.where(alpha_values > diff)[0][0]
        return step

class Improved_Forward_diffusion():
    
    def __init__(self, max_timesteps, s=0.008, bias=0.0):
        self.max_timesteps = max_timesteps
        self.s = s
        t_div_T = torch.arange(0, max_timesteps+1) / max_timesteps
        f = torch.pow(torch.cos((t_div_T + self.s) / (1 + self.s) * (math.pi / 2)), 2)
        alphas_cumprod = bias + ((1 - bias) * (f / f[0]))
        
        temp = alphas_cumprod[:-1]
        temp = torch.cat([torch.tensor([1.]), temp],0 )
        self.betas = torch.clip(1 - (alphas_cumprod / temp), max=0.9999)

        alphas = 1. - self.betas
        alphas_cumprod_prev = torch.nn.functional.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
        self.posterior_variance = self.betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

    def get_index_from_list(self, vals, t, x_shape):
        """ 
        Returns a specific index t of a passed list of values vals
        while considering the batch dimension.
        """
        batch_size = t.shape[0]
        out = vals.gather(-1, t.cpu())
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)
    
    def get_values_from_timestep_for_sampling(self, t, x_shape):
        """ 
        Returns a specific index t of a passed list of values vals
        while considering the batch dimension.
        """
        batch_size = t.shape[0]
        beta_t = self.get_index_from_list(self.betas, t, x_shape)
        sqrt_one_minus_alphas_cumprod_t = self.get_index_from_list(self.sqrt_one_minus_alphas_cumprod, t, x_shape)
        sqrt_recip_alphas_t = self.get_index_from_list(self.sqrt_recip_alphas, t, x_shape)
        posterior_variance_t = self.get_index_from_list(self.posterior_variance, t, x_shape)
        return beta_t, sqrt_one_minus_alphas_cumprod_t, sqrt_recip_alphas_t, posterior_variance_t
        
    def forward_sample(self, action_dict, t, device="cpu"):
        """ 
        Takes an image and a timestep as input and 
        returns the noisy version of it
        """
        noise_action_dict = {}
        shifted_action_dict = {}
        noise_dict = {}
        for key in action_dict.keys():
            noise = torch.randn_like(action_dict[key])
            
            sqrt_alphas_cumprod_t = self.get_index_from_list(self.sqrt_alphas_cumprod, t, action_dict[key].shape)
            sqrt_one_minus_alphas_cumprod_t = self.get_index_from_list(
                                                self.sqrt_one_minus_alphas_cumprod, t, action_dict[key].shape)
            
            shifted_action_dict[key] = sqrt_alphas_cumprod_t.to(device) * action_dict[key].to(device)
            noise_action_dict[key] = shifted_action_dict[key] + sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device)
            noise_dict[key] = noise.to(device)
        
        return noise_action_dict, noise_dict, shifted_action_dict

    def get_mean_and_std(self, action_dict, t, device="cpu"):
        std = {}
        mean = {}
        for key in action_dict.keys():
            sqrt_alphas_cumprod_t = self.get_index_from_list(self.sqrt_alphas_cumprod, t, action_dict[key].shape)
            sqrt_one_minus_alphas_cumprod_t = self.get_index_from_list(
                                                self.sqrt_one_minus_alphas_cumprod, t, action_dict[key].shape)
            
            mean[key] = sqrt_alphas_cumprod_t.to(device) * action_dict[key].to(device)
            std[key] = sqrt_one_minus_alphas_cumprod_t.to(device)
        
        return mean, std
    
    def get_step(self, diff):
        alpha_values = self.sqrt_one_minus_alphas_cumprod / self.sqrt_alphas_cumprod
        step = torch.where(alpha_values > diff)[0][0]
        return step

class Forward_DSM():
    
    def __init__(self, max_timesteps=100, start=1e-2, end=2e-1):
        self.max_timesteps = max_timesteps
        self.start_beta = start
        self.end_beta = end
        
        self.stds = torch.cat([torch.tensor([0.]), torch.linspace(start, end, max_timesteps)],0)
        self.sigmas = torch.pow(self.stds, 2)
        
    def get_index_from_list(self, vals, t, x_shape):
        """ 
        Returns a specific index t of a passed list of values vals
        while considering the batch dimension.
        """
        batch_size = t.shape[0]
        out = vals.gather(-1, t.cpu())
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)
        
    def forward_sample(self, action_dict, t, device="cpu"):
        """ 
        Takes an image and a timestep as input and 
        returns the noisy version of it
        """
        noise_action_dict = {}
        scaled_noise_dict = {}
        stds_vec = self.get_index_from_list(self.stds, t, t.shape)
        for key in action_dict.keys():
            noise = torch.randn_like(action_dict[key]).to(device)
            stds = self.get_index_from_list(self.stds, t, action_dict[key].shape)
            sigmas = torch.pow(stds, 2)
            
            scaled_noise = stds.to(device) * noise
            noise_action_dict[key] = action_dict[key] + scaled_noise
            scaled_noise_dict[key] = scaled_noise
        
        return noise_action_dict, scaled_noise_dict, stds_vec

class Forward_Latent_DSM():
    
    def __init__(self, ae, max_timesteps=100, start=1e-2, end=2e-1):
        self.max_timesteps = max_timesteps
        self.start_beta = start
        self.end_beta = end
        
        self.stds = torch.cat([torch.tensor([0.]), torch.linspace(start, end, max_timesteps)],0)
        self.sigmas = torch.pow(self.stds, 2)
        self.ae = ae
        
    def get_index_from_list(self, vals, t, x_shape):
        """ 
        Returns a specific index t of a passed list of values vals
        while considering the batch dimension.
        """
        batch_size = t.shape[0]
        out = vals.gather(-1, t.cpu())
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)
        
    def forward_sample(self, action_dict, t, device="cpu"):
        """ 
        Takes an image and a timestep as input and 
        returns the noisy version of it
        """
        noised_action_dict = {}
        noise_dict = {}
        stds_vec = self.get_index_from_list(self.stds, t, t.shape)
        
        with torch.no_grad():
            z = self.ae.encode(action_dict)
            
            noise = torch.rand_like(z).to(device)
            stds = self.get_index_from_list(self.stds, t, z.shape)
            scaled_latent_noise = stds.to(device) * noise
        
            noised_z = z + scaled_latent_noise
            noised_action_dict = self.ae.decode(noised_z)
            
        for key in noised_action_dict.keys():
            noise_dict[key] = noised_action_dict[key] - action_dict[key]
        
        return noised_action_dict, noise_dict, stds_vec

def calculate_end(cfg, rank, max_step, start=1e-5, iteration=5000, rot_mode="6d", target_mode="max", sigma=1.):
    print("##############################")
    print("Calculate hyper-parameter for retrieval diffusion")
    print("loading data")
    dataset = RLBench_DMOEBM("train", cfg, save_dataset=False, num_frame=100, rot_mode=rot_mode, img_aug=False)
    dataset.without_image = True
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1000, shuffle=False, num_workers=8)

    print("setting retrieval")
    Retriever = Direct_Retrieval(dataset)   
    
    print("start getting kNN")
    for data in dataloader:
        _, query = data
        near_queries, _, _ = Retriever.retrieve_k_sample(query, k=cfg.DIFFUSION.MAX_RANK+1)
    print("end\n")

    print(f"task: {cfg.DATASET.RLBENCH.TASK_NAME} rank: {rank}")
    diff_dict = {}
    retrieved_query = {}
    for key in near_queries.keys():
        retrieved_query[key] = near_queries[key][:,rank]
        diff_dict[key] = torch.pow(retrieved_query[key] - query[key], 2)

    uv_diff_list = []
    z_diff_list = []
    rotation_diff_list = []
    grasp_diff_list = []

    for i in range(1000):
        uv_diff_list.append(torch.max(diff_dict["uv"][i]).item())
        z_diff_list.append(torch.max(diff_dict["z"][i]).item())
        rotation_diff_list.append(torch.max(diff_dict["rotation"][i]).item())
        grasp_diff_list.append(torch.max(diff_dict["grasp_state"][i]).item())

    if target_mode == "max":
        target = max([max(uv_diff_list), max(z_diff_list)])
    elif target_mode == "mean":
        target_list = uv_diff_list + z_diff_list
        target = sum(target_list) / len(target_list)
    elif target_mode == "median":
        target_list = uv_diff_list + z_diff_list
        target = statistics.median(target_list)
    elif "std" in target_mode:
        target_list = uv_diff_list + z_diff_list
        
        mean = sum(target_list) / len(target_list)
        std_dev = statistics.stdev(target_list)
        
        if target_mode == "std1":
            target = mean + (2 * std_dev)
        elif target_mode == "std2":
            target = mean + (2 * std_dev)
        else:
            raise NotImplementedError("TODO")
            
    print(f"diff_{target}")

    x = torch.nn.parameter.Parameter(torch.tensor(2e-5, requires_grad=True))
    optimizer = torch.optim.Adam([x], lr=1e-8)
    target = (sigma**2)/((sigma**2) + target)

    print("start optimizing hyper-params")
    for i in tqdm(range(iteration)):
        value = 1
        for t in range(0, max_step):
            value = value * (1 - start - x*t) 

        loss = torch.abs(value - target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print("done")

    with torch.no_grad():
        result = start + x*max_step

    print(f"target: {target} loss: {loss.item()} result: (start, end) = ({start}, {result}) \n")
    return result.item()

def calculate_end_for_cosine(cfg, rank, rot_mode="6d"):
    print("Calculate hyper-parameter for retrieval diffusion")
    print("loading data")
    dataset = RLBench_DMOEBM("train", cfg, save_dataset=False, num_frame=100, rot_mode=rot_mode, img_aug=False)
    dataset.without_image = True
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1000, shuffle=False, num_workers=8)

    print("setting retrieval")
    Retriever = Direct_Retrieval(dataset)   
    
    print("start getting kNN")
    for data in dataloader:
        _, query = data
        near_queries, _, _ = Retriever.retrieve_k_sample(query, k=cfg.DIFFUSION.MAX_RANK+1)
    print("end\n")

    print(f"task: {cfg.DATASET.RLBENCH.TASK_NAME} rank: {rank}")
    diff_dict = {}
    retrieved_query = {}
    for key in near_queries.keys():
        retrieved_query[key] = near_queries[key][:,rank]
        diff_dict[key] = torch.pow(retrieved_query[key] - query[key], 2)

    uv_diff_list = []
    z_diff_list = []
    rotation_diff_list = []
    grasp_diff_list = []

    for i in range(1000):
        uv_diff_list.append(torch.max(diff_dict["uv"][i]))
        z_diff_list.append(torch.max(diff_dict["z"][i]))
        rotation_diff_list.append(torch.max(diff_dict["rotation"][i]))
        grasp_diff_list.append(torch.max(diff_dict["grasp_state"][i]))

    target = max([max(uv_diff_list), max(z_diff_list)])
    target = 1/(1 + target)

    print(f"target: {target} \n")
    return target.item()

class Denoising_Diffusion(torch.nn.Module):
    def __init__(self, query_keys, query_dims, diff_cfg, img_size=256, seq_length=101, input_dim=4, dims=[96, 192, 384, 768], 
                    enc_depths=[3,3,9,3], enc_layers=['conv','conv','conv','conv'], enc_act="gelu", enc_norm="layer", 
                    dec_depths=[3,3,3], dec_layers=['conv','conv','conv'], dec_act="gelu", dec_norm="layer", drop_path_rate=0.1,
                    extractor_name="query_uv_feature", predictor_act="gelu",
                    predictor_drop=0., img_guidance_rate=1.0, query_emb_dim=256):
        """
        Args:
        img_size (int): Size of image. We assume image is square.
        input_dim (int): Size of channel. 3 for RGB image.
        enc_depths (list[int]): Number of blocks at each stage for encoder.
        dec_depths (list[int]): Number of blocks at each stage for decoder.
        predictor_depth (int): Number of blocks at predicotr. This value is for IABC predictor.
        dims (list[int]): The channel size of each feature map.
        enc_layers (list[str]): Name of layer at each stage of encoder.
        dec_layers (list[str]): Name of layer at each stage of decoder.
        predictor (str): Name of predictor
        predictor_prob_func (str): Function to change the vector to probability. This function is for IABC.
        act (str): Activation function.
        norm (str): Normalization function.
        atten (str): Name of attention. If layer name is atten, indicated atten layer is used.
        drop_path (float): Stochastic depth rate for encoder and decoder. Default: 0.1
        """
        super().__init__()
        self.img_size = img_size
        if query_emb_dim == 0:
            emb_dim = dims[0] // 2
        else:
            emb_dim = query_emb_dim
        
        # image encoder (UNet)
        self.enc = Resnet_Like_Encoder(img_size, in_chans=input_dim, depths=enc_depths, dims=dims, layers=enc_layers, drop_path_rate=drop_path_rate, activation=enc_act, norm=enc_norm)
        self.dec = Resnet_Like_Decoder(img_size, depths=dec_depths, enc_dims=dims, layers=dec_layers, drop_path_rate=drop_path_rate, emb_dim=emb_dim, activation=dec_act, norm=dec_norm)
        
        # extract important feature
        self.ife = Image_feature_extractor_model(extractor_name, emb_dim, 1)
        
        # predictor
        self.predictor = Diffusion_Predictor(query_keys, query_dims, emb_dim, dropout=predictor_drop, act=predictor_act, img_guidance_rate=img_guidance_rate)
        
        # forward_diffusion
        self.max_steps = diff_cfg.STEP
        self.query_keys = query_keys
        self.query_dims = query_dims
        self.sequence_length = seq_length
        if diff_cfg.TYPE == "normal":
            self.forward_diffusion_function = Forward_diffusion(self.max_steps, start=diff_cfg.START, end=diff_cfg.END)
        elif diff_cfg.TYPE == "improved":
            self.forward_diffusion_function = Improved_Forward_diffusion(self.max_steps, s=diff_cfg.S, bias=diff_cfg.BIAS)
        
    def forward(self, img, query, time_step, with_feature=False):
        device = img.device
        debug_info = {}
        
        # encode image
        if with_feature == False:
            img_feature = self.get_img_feature(img)
        else:
            img_feature = self.img_feature
        
        noised_query, noise_dict, shifted_query = self.forward_diffusion_function.forward_sample(query, time_step, device)
        
        # extract important features from image features
        uv = noised_query['uv'].to(device)
        img_feature_dict, info = self.ife(img_feature, uv) # TODO: dict?
        for key in info.keys():
            debug_info[key] = info[key]
        
        output_dict = self.predictor(img_feature_dict["img_feature"], noised_query, time_step) # TODO: dict?
            
        return output_dict, noise_dict, debug_info
        
    def get_img_feature(self, img):
        """
        input:
            img: torch.tensor -> shape: (B C H W), C=input_dim
        output:
            img_feature: torch.tensor -> shape: (B C H W), C=emb_dim
        """
        img_feature = self.enc(img)
        img_feature = self.dec(img_feature)
        self.img_feature = img_feature
        return img_feature
    
    def get_extracted_img_feature(self, img, query, with_feature=False):
        device = img.device
        if with_feature == False:
            img_feature = self.get_img_feature(img)
        else:
            img_feature = self.img_feature
        
        # extract important features from image features
        uv = query['uv'].to(device)
        img_feature_dict, _ = self.ife(img_feature, uv) # TODO: dict?
        
        return img_feature_dict["img_feature"]

    @torch.no_grad()
    def sample_timestep(self, img, query, time_step, with_feature=False):
        """
        Calls the model to predict the noise in the image and returns 
        the denoised image. 
        Applies noise to this image, if we are not in the last step yet.
        """
        device = img.device
        debug_info = {}
        
        # encode image
        if with_feature == False:
            img_feature = self.get_img_feature(img)
        else:
            img_feature = self.img_feature
        
        # extract important features from image features
        uv = query['uv'].to(device)
        img_feature_dict, _ = self.ife(img_feature, uv) # TODO: dict?        
        pred_noise_dict = self.predictor(img_feature_dict["img_feature"], query, time_step) # TODO: dict?
        
        pred_dict = {}
        for key in pred_noise_dict.keys():
            betas_t, sqrt_one_minus_alphas_cumprod_t, sqrt_recip_alphas_t, posterior_variance_t = self.forward_diffusion_function.get_values_from_timestep_for_sampling(time_step, pred_noise_dict[key].shape)

            # Call model (current image - noise prediction)
            model_mean = sqrt_recip_alphas_t * (
                query[key] - betas_t * pred_noise_dict[key] / sqrt_one_minus_alphas_cumprod_t
            )
            
            noise = torch.randn_like(pred_noise_dict[key])
            noise_mask = (time_step != 0.0)
            noise = torch.einsum("b, bsd -> bsd", noise_mask, noise)
            pred_dict[key] = model_mean + torch.sqrt(posterior_variance_t) * noise 

        return pred_dict
    
    @torch.no_grad()
    def sampling(self, image):
        action_dict = {}
        B,_,_,_ = image.shape
        device = image.device
        for key,dims in zip(self.query_keys, self.query_dims):
            action_dict[key] = torch.randn(B,self.sequence_length,dims).to(device)
            
        result_dict = {}
        
        for i in range(1, self.max_steps+1)[::-1]:
            result_dict[i] = action_dict
            time_step = torch.full((B,), i, device=device, dtype=torch.long)
            if i == (self.max_steps):
                action_dict = self.sample_timestep(image, action_dict, time_step)
            else:
                action_dict = self.sample_timestep(image, action_dict, time_step, with_feature=True)
            
        result_dict[0] = action_dict
            
        return result_dict
    
    @torch.no_grad()
    def reconstruct(self, image, query, t=10, div=1):
        B,_,_,_ = image.shape
        device = image.device
        time_step = torch.full((B,), t, device=device, dtype=torch.long)
        result_dict = {}

        if t == 0:
            pass
        else:
            query, _, _ = self.forward_diffusion_function.forward_sample(query, time_step, device)    
            for i in range(1,t+1)[::-1]:
                result_dict[i] = query
                time_step = torch.full((B,), i, device=device, dtype=torch.long)
                if i == t:
                    query = self.sample_timestep(image, query, time_step)
                else:
                    query = self.sample_timestep(image, query, time_step, with_feature=True)
                
                if i % div == 0:
                    result_dict[i] = query
        
        result_dict[0] = query
            
        return result_dict

    @torch.no_grad()
    def multiple_sample_timestep(self, img, query, time_step, with_feature=False):
        """
        Calls the model to predict the noise in the image and returns 
        the denoised image. 
        Applies noise to this image, if we are not in the last step yet.
        """
        device = img.device
        debug_info = {}
        
        # encode image
        if with_feature == False:
            img_feature = self.get_img_feature(img)
        else:
            img_feature = self.img_feature
        
        # extract important features from image features
        uv = query['uv'].to(device)
        N, _, _ = uv.shape
        # img_feature = img_feature.expand(B, *img_feature.shape[1:])
        img_feature = repeat(img_feature, "B C H W -> (N B) C H W", N=N)
        img_feature_dict, _ = self.ife(img_feature, uv) # TODO: dict?        
        pred_noise_dict = self.predictor(img_feature_dict["img_feature"], query, time_step) # TODO: dict?
        
        pred_dict = {}
        for key in pred_noise_dict.keys():
            betas_t, sqrt_one_minus_alphas_cumprod_t, sqrt_recip_alphas_t, posterior_variance_t = self.forward_diffusion_function.get_values_from_timestep_for_sampling(time_step, pred_noise_dict[key].shape)

            # Call model (current image - noise prediction)
            model_mean = sqrt_recip_alphas_t * (
                query[key] - betas_t * pred_noise_dict[key] / sqrt_one_minus_alphas_cumprod_t
            )
            
            noise = torch.randn_like(pred_noise_dict[key])
            noise_mask = (time_step != 0.0)
            noise = torch.einsum("b, bsd -> bsd", noise_mask, noise)
            pred_dict[key] = model_mean + torch.sqrt(posterior_variance_t) * noise 

        return pred_dict
    
    @torch.no_grad()
    def multiple_reconstruct(self, image, query, good_feature, t=10, n=32):
        """
        image: shape->(1 C H W)
        query: dict
            query["???"]: shape->(1 k S D) k=num of retrieved data, S=length of sequence(101), D=dim of data
            e.g., query["uv"]: shape->(1 k S 2)
        """
        B,k,_,_ = query["uv"].shape
        if B != 1:
            raise ValueError("multiple reconstruct is applicable for single image")
        
        device = image.device
        time_step = torch.full((k * n,), t, device=device, dtype=torch.long)
        original_query = copy.deepcopy(query)
        for key in query.keys():
            # query[key] = query[key].expand(n, *query[key].shape[1:])
            query[key] = repeat(query[key], "B ... -> (n B) ...", n=n)
            query[key] = rearrange(query[key], "n k S D -> (n k) S D")

        print("start denoising")
        if t == 0:
            pass
        else:
            query, _, _ = self.forward_diffusion_function.forward_sample(query, time_step, device)    
            for i in range(1,t+1)[::-1]:
                time_step = torch.full((k * n,), i, device=device, dtype=torch.long)
                if i == t:
                    query = self.multiple_sample_timestep(image, query, time_step)
                else:
                    query = self.multiple_sample_timestep(image, query, time_step, with_feature=True)
        print("end denoising")

        print("evaluate feature")
        pred_query = {}
        for key in query.keys():
            pred_query[key] = torch.cat([original_query[key][0].to(device), query[key]], 0)
        
        # img_feature = img_feature.expand(k * (n + 1), *img_feature.shape[1:])
        img_feature = repeat(self.img_feature, "B C H W -> (N B) C H W", N=k*(n+1))
        img_feature_dict, _ = self.ife(img_feature, pred_query["uv"]) # (n + 1) * k, S, D
        img_features = img_feature_dict["img_feature"] # S, (n + 1) * k, D
        img_features = rearrange(img_features, "S (n k) D -> n k (S D)",k=k)
        # good_feature = good_feature.expand(n, *good_feature.shape[1:])
        good_feature = repeat(good_feature, "B ... -> (n B) ...",n=(n+1))
        feature_diff = torch.mean(torch.abs(good_feature - img_features), 2)
        feature_diff = rearrange(feature_diff, "n k -> (n k)")

        print("sorting")
        dists, nears = torch.sort(feature_diff)
        sorted_query = {}
        for key in pred_query.keys():
            _, S, D = pred_query[key].shape
            index_nears = repeat(nears, "B -> B S D",S=S,D=D)
            
            sorted_query[key] = torch.gather(pred_query[key], 0, index_nears)

        dists, nears = dists, nears
        
        return sorted_query, dists

class Diffusion_Predictor(torch.nn.Module):
    def __init__(self, query_keys, query_dims, latent_dim, 
                    sequence_length=101, ff_size=1024, num_layers=8, num_heads=4, dropout=0.0,
                    activation="gelu", img_guidance_rate=1.0, **kargs):
        super().__init__()
        """
        This model is based of Motion Diffusion Model from https://arxiv.org/pdf/2209.14916.pdf
        The image features is fed into the model insted of the text features.
        """
        # configuration
        self.query_keys = query_keys
        self.query_dims = query_dims
        self.sequence_length = sequence_length
        self.latent_dim = latent_dim
        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.activation = activation
        self.img_guidance_rate = img_guidance_rate

        # embedding pose to vector
        self.query_emb_model = Query_emb_model(query_keys, query_dims, latent_dim)
        
        # embedding noise step
        self.steps_encoder = StepEncoding(latent_dim)
        
        # Transformer
        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)
        seqTransEncoderLayer = torch.nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                        nhead=self.num_heads,
                                                        dim_feedforward=self.ff_size,
                                                        dropout=self.dropout,
                                                        activation=self.activation)
        self.seqTransEncoder = torch.nn.TransformerEncoder(seqTransEncoderLayer,
                                                        num_layers=self.num_layers)
        
        # decoder
        module_dict = {}
        for key, dim in zip(query_keys, query_dims):
            if key == "time":
                continue
            module_dict[key] = torch.nn.Sequential(
                            LinearBlock(latent_dim, int(latent_dim / 2)),
                            torch.nn.GELU(),
                            LinearBlock(int(latent_dim / 2), int(latent_dim / 2)),
                            torch.nn.GELU(),
                            LinearBlock(int(latent_dim / 2), dim))
        self.output_module_dict = torch.nn.ModuleDict(module_dict)
        
    def forward(self, img_feature, query, time_step):
        pred_noise = self.pred_noise(img_feature, query, time_step)
        return pred_noise
    
    def pred_noise(self, img_feature, query, time_step):
        # embedding the pose to the vector
        query_feature = self.query_emb_model(query)
        bs, nframes, nfeats = query_feature.shape
        query_feature = rearrange(query_feature, "B S D -> S B D")
        
        # add time embedding 
        time_emb = self.steps_encoder(time_step)
        query_feature = torch.cat([query_feature, time_emb], 0) # (S+1) B D
        
        # add positional encoding
        query_feature = self.sequence_pos_encoder(query_feature)
        img_feature = self.sequence_pos_encoder(img_feature)
        
        if random.random() <= self.img_guidance_rate:
            cat_feature = torch.cat([query_feature, img_feature]) # (2S + 1) B D
        else:
            cat_feature = query_feature
        
        # transformer process
        motion_feature = self.seqTransEncoder(cat_feature) # (2S + 1) B D
        
        # decode
        motion_feature = rearrange(motion_feature[:nframes], "S B D -> B S D") # (2S + 1) B D -> B S D
        
        pred_noise_dict = {}
        for key in self.output_module_dict.keys():
            pred_noise_dict[key] = query[key] + self.output_module_dict[key](motion_feature)
        
        return pred_noise_dict

class Image_feature_extractor_model(torch.nn.Module):
    def __init__(self, extractor_name, img_feature_dim, down_scale, num_vec=8):
        super().__init__()
        
        if "query_uv_feature" == extractor_name:
            self.extractor = query_uv_extractor(down_scale)
        else:
            raise ValueError(f"Invalid key: {extractor_name} is invalid key for the Image-feature_extractor_model (in feature_extractory.py)")
        
    def forward(self, x, uv):
        extractor_dict, extractor_info = self.extractor(x, uv)
        return extractor_dict, extractor_info

class query_uv_extractor(torch.nn.Module):
    
    def __init__(self, down_scale, do_norm=False, pos_emb=False, dim=0):
        super().__init__()
        self.down_scale = down_scale
        self.norm = do_norm

        if pos_emb:
            self.pos_emb = PositionalEncoding(dim)
            if dim == 0:
                raise ValueError("Invalid dim")
            self.dim = dim
        else:
            self.dim = 0
            
    def forward(self,x,y):
        """
        Input 
        x: feature B,C,H,W
        y: pose B,S,2

        Output
        output_dict: dict
            key:
                img_feature: feature of image, shape(B, N, C)

        Note
        B: batch size
        C: Num channel
        H: Height
        W: Width
        N: Num query
        """
        debug_info = {}
        output_dict = {}
        B,C,H,W = x.shape
        
        y = torch.unsqueeze(y, 1)
        feature = torch.nn.functional.grid_sample(x, y, mode='bilinear', padding_mode='zeros', align_corners=True) # B, C, N, S
        feature = rearrange(feature, 'B C N S -> S N B C') # B, N, S, C
        feature = torch.squeeze(feature, 1)
        output_dict["img_feature"] = feature
        return output_dict, debug_info
    
    def pos_norm(self,pos,H,W):
        x_coords = pos[:,:,0]
        y_coords = pos[:,:,1]
        
        x_coords = (x_coords / W) * 2 - 1
        y_coords = (y_coords / H) * 2 - 1

class StepEncoding(torch.nn.Module):
    def __init__(self, dim, max_len=5000):
        super().__init__()
        self.dim = dim
        self.max_len = max_len
        self.linear_layer = self.make_linear_model(dim, dim, "gelu")

    def forward(self, time):
        """
        -----------------------------
        inputs
        x: torch.tensor(S, B, D)
        time: torch.tensor(B)
        -----------------------------
        S: length of sequence
        B: Batch size
        D: Dimension of feature
        """
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(self.max_len) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        embeddings = self.linear_layer(embeddings)
        embeddings = repeat(embeddings, "B D -> S B D", S=1)
        return embeddings
    
    def make_linear_model(self, input_dim, output_dim, act):
        model = torch.nn.Sequential(
            torch.nn.Linear(input_dim, output_dim),
            self.activation_layer(act),
            torch.nn.Linear(output_dim, output_dim * 2),
            self.activation_layer(act),
            torch.nn.Linear(output_dim * 2, output_dim))
        return model
    
    @staticmethod
    def activation_layer(name):
        if name == 'relu':
            layer = torch.nn.ReLU()
        elif name == 'prelu':
            layer = torch.nn.PReLU()
        elif name == 'lrelu':
            layer = torch.nn.LeakyReLU(0.2)
        elif name == 'tanh':
            layer = torch.nn.Tanh()
        elif name == 'sigmoid':
            layer = torch.nn.Sigmoid()
        elif name == 'gelu':
            layer = torch.nn.GELU()
        elif name == 'none':
            layer = torch.nn.Identity()
        else:
            raise ValueError("Invalid activation")
        return layer

class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)

    def forward(self, x):
        # not used in the final model
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)
    
class Query_emb_model(torch.nn.Module):
    def __init__(self, query_keys, query_dims, emb_dim, act="gelu"):
        """
        Input:
        query_keys: list of query keys you want to use. Other queries will be ignored in the forward process.
        query_dims: list of dim of each query that you want to use.
        emb_dim: dimension of output feature (embedded query)
        """
        super().__init__()

        self.register_query_keys = query_keys
        query_total_dim = sum(query_dims)
        self.query_emb_model = self.make_linear_model(query_total_dim, emb_dim, act)

    def forward(self, querys):
        """
        Input
        querys: dict
            key:
                str
            value:
                torch.tensor: shape -> (B, S, D), B -> Batch Size, N, Num of query in each batch, S -> Sequence Length, D -> Dim of each values
        Output:
        query_emb: torch.tensor: shape -> (B, S, QD), QD -> emb_dim
        """
        keys = list(querys.keys())
        keys.sort()

        query_list = []
        for key in keys:
            if key in self.register_query_keys:
                query_list.append(querys[key])
        
        query_cat = torch.cat(query_list, 2)
        query_emb = self.query_emb_model(query_cat)
        return query_emb
        
    def make_linear_model(self, input_dim, output_dim, act):
        model = torch.nn.Sequential(
            torch.nn.Linear(input_dim, output_dim),
            self.activation_layer(act),
            torch.nn.Linear(output_dim, output_dim * 2),
            self.activation_layer(act),
            torch.nn.Linear(output_dim * 2, output_dim))
        return model
    
    @staticmethod
    def activation_layer(name):
        if name == 'relu':
            layer = torch.nn.ReLU()
        elif name == 'prelu':
            layer = torch.nn.PReLU()
        elif name == 'lrelu':
            layer = torch.nn.LeakyReLU(0.2)
        elif name == 'tanh':
            layer = torch.nn.Tanh()
        elif name == 'sigmoid':
            layer = torch.nn.Sigmoid()
        elif name == 'gelu':
            layer = torch.nn.GELU()
        elif name == 'none':
            layer = torch.nn.Identity()
        else:
            raise ValueError("Invalid activation")
        return layer

class Diffusion_Loss(torch.nn.Module):
    
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.L1Loss()
    
    def forward(self, pred_dict, noise_dict, mode="train"):
        sum_loss = 0.
        loss_dict = {}
        for key in pred_dict.keys():
            loss = self.l1(pred_dict[key], noise_dict[key])
            loss_dict[f"{mode}/{key}"] = loss.item()
            sum_loss += loss

        loss_dict[f"{mode}/loss"] = sum_loss.item()
        
        return sum_loss, loss_dict

class DSM_Loss(torch.nn.Module):
    
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.L1Loss(reduction="none")
    
    def forward(self, pred_dict, scaled_noise_dict, stds, mode="train"):
        sum_loss = 0.
        loss_dict = {}
        for key in pred_dict.keys():
            loss = self.l1(pred_dict[key], -scaled_noise_dict[key])
            loss = torch.mean(loss.view(loss.shape[0], -1), 1)
            loss_dict[f"{mode}/{key}"] = torch.mean(loss).item()
            loss = torch.mean(loss * torch.reciprocal(stds))
            sum_loss += loss

        loss_dict[f"{mode}/loss"] = sum_loss.item()
        
        return sum_loss, loss_dict
    
class DSM_Loss_DiffusionLike(torch.nn.Module):
    
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.L1Loss()
    
    def forward(self, pred_dict, noise_dict, stds, mode="train"):
        sum_loss = 0.
        loss_dict = {}
        for key in pred_dict.keys():
            loss = self.l1(pred_dict[key], -noise_dict[key])
            loss_dict[f"{mode}/{key}"] = loss.item()
            sum_loss += loss

        loss_dict[f"{mode}/loss"] = sum_loss.item()
        
        return sum_loss, loss_dict

class Noise_Sampler():
    
    def __init__(self, cfg, max_noise_steps=20, vae="none"):
        self.max_steps = max_noise_steps
        
        # define noise scale
        if cfg.SAMR.NOISE.SCALE == "custom":
            self.min_std = cfg.SAMR.NOISE.CUSTOM.MIN
            self.max_std = cfg.SAMR.NOISE.CUSTOM.MAX
        elif cfg.SAMR.NOISE.SCALE == "auto":
            raise ValueError("TODO")
        else:
            raise ValueError("Invalid mode. cfg.NOISE.SCALE should be custom or auto")
        
        # get noise class
        self.noise_type = cfg.SAMR.NOISE.TYPE
        if self.noise_type == "gaussian":
            self.noise_sampler = Gaussian_Noise(max_noise_steps=self.max_steps, min_noise_std=self.min_std, max_noise_std=self.max_std)
        elif self.noise_type == "latent-gaussian":
            if vae == "none":
                raise ValueError("Please input pre-trained vae")
            self.noise_sampler = Latent_Gaussian_Noise(vae, max_noise_steps=self.max_steps, min_noise_std=self.min_std, max_noise_std=self.max_std)
            
    def get_noise(self, inputs):
        return self.noise_sampler.get_noise(inputs)

    def get_noised_query(self, query, time_step):
        return self.noise_sampler.get_noised_query(query, time_step)

class Gaussian_Noise():
    def __init__(self, max_noise_steps=20, min_noise_std=0.1, max_noise_std=2.0):
        self.max_noise_steps = max_noise_steps
        self.min_noise_std = min_noise_std
        self.max_noise_std = max_noise_std
        
        steps = torch.arange(0, max_noise_steps)
        steps = steps / (max_noise_steps - 1) # normalize
        steps = steps * (max_noise_std - min_noise_std)
        self.noise_std_list = steps + min_noise_std
        
    def get_noise(self, query):
        B, S, _ = query["uv"].shape
        device = query["uv"].device
        random_steps = torch.randint(0, self.max_noise_steps, (B,)).long()
        sampled_std = self.noise_std_list.gather(-1, random_steps).to(device)
        
        sampled_noise = {}
        noised_query = {}
        for key in query.keys():
            if key == "time":
                continue
                
            _, _, D = query[key].shape
            sampled_noise[key] = repeat(sampled_std, "B -> B S D",S=S,D=D) * torch.randn(B, S, D).to(device)
            noised_query[key] = query[key] + sampled_noise[key]
            
        return noised_query, sampled_noise, torch.unsqueeze(sampled_std, 1), random_steps.to(device)
    
    def get_noised_query(self, query, time_step):
        B, S, _ = query["uv"].shape
        device = query["uv"].device
        time_step = torch.ones(B).long() * time_step
        sampled_std = self.noise_std_list.gather(-1, time_step).to(device)
        
        sampled_noise = {}
        noised_query = {}
        for key in query.keys():
            if key == "time":
                continue
                
            _, _, D = query[key].shape
            sampled_noise[key] = repeat(sampled_std, "B -> B S D",S=S,D=D) * torch.randn(B, S, D).to(device)
            noised_query[key] = query[key] + sampled_noise[key]
            
        return noised_query

class Latent_Gaussian_Noise():
    def __init__(self, vae, max_noise_steps=20, min_noise_std=0.1, max_noise_std=2.0):
        self.max_noise_steps = max_noise_steps
        self.min_noise_std = min_noise_std
        self.max_noise_std = max_noise_std
        
        steps = torch.arange(0, max_noise_steps)
        steps = steps / (max_noise_steps - 1) # normalize
        steps = steps * (max_noise_std - min_noise_std)
        self.noise_std_list = steps + min_noise_std
        
        self.vae = vae

    def get_noise(self, query):
        B, _, _ = query["uv"].shape
        device = query["uv"].device
        self.vae = self.vae.to(device)

        with torch.no_grad():
            z = self.vae.encode(query)
            _, D = z.shape

        random_steps = torch.randint(0, self.max_noise_steps, (B,)).long()
        sampled_std = self.noise_std_list.gather(-1, random_steps).to(device)
        sampled_z_noise = repeat(sampled_std, "B -> B D", D=D) * torch.randn(B, D).to(device)
        noised_z = z + sampled_z_noise

        with torch.no_grad():
            noised_query = self.vae.decode(noised_z)

        sampled_noise = {}
        for key in noised_query.keys():
            sampled_noise[key] = noised_query[key] - query[key]

        return noised_query, sampled_noise, torch.unsqueeze(sampled_std, 1), random_steps.to(device)
    
    def get_noised_query(self, query, time_step):
        B, _, _ = query["uv"].shape
        device = query["uv"].device
        self.vae = self.vae.to(device)

        with torch.no_grad():
            z = self.vae.encode(query)
            _, D = z.shape

        time_step = torch.ones(B).long() * time_step
        sampled_std = self.noise_std_list.gather(-1, time_step).to(device)
        sampled_z_noise = repeat(sampled_std, "B -> B D", D=D) * torch.randn(B, D).to(device)
        noised_z = z + sampled_z_noise

        with torch.no_grad():
            noised_query = self.vae.decode(noised_z)

        sampled_noise = {}
        for key in noised_query.keys():
            sampled_noise[key] = noised_query[key] - query[key]

        return noised_query