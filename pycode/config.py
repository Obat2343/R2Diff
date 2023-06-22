from yacs.config import CfgNode as CN
import os

_C = CN()

##################
##### OUTPUT ##### 
##################
_C.OUTPUT = CN()
_C.OUTPUT.BASE_DIR = "../result"
_C.OUTPUT.NUM_TRIAL = 5
_C.OUTPUT.MAX_ITER = 100000
_C.OUTPUT.SAVE_ITER = 10000 # interval to save model and log eval loss
_C.OUTPUT.LOG_ITER = 100 # interval to log training loss
_C.OUTPUT.EVAL_ITER = 1000

###################
##### DATASET ##### 
###################

_C.DATASET = CN()
_C.DATASET.NAME = "RLBench"
_C.DATASET.BATCH_SIZE = 32
_C.DATASET.IMAGE_SIZE = 256

### RLBENCH ###
_C.DATASET.RLBENCH = CN()
_C.DATASET.RLBENCH.TASK_NAME = 'PickUpCup' # e.g. 'CloseJar', 'PickUpCup'
_C.DATASET.RLBENCH.PATH = os.path.abspath('../dataset/RLBench') # '../dataset/RLBench-Local'
_C.DATASET.RLBENCH.SEQ_LEN = 1000
_C.DATASET.RLBENCH.QUERY_LIST = ["uv", "time", "rotation_quat", "grasp", "z"]
_C.DATASET.RLBENCH.QUERY_DIMS = [2, 1, 4, 1, 1]

###################
###### MODEL ######
###################

_C.MODEL = CN()

_C.MODEL.CONV_DIMS = [96, 192, 384, 768]
_C.MODEL.ENC_LAYERS = ['convnext','convnext','convnext','convnext']
_C.MODEL.ENC_DEPTHS = [3,3,9,3]

_C.MODEL.DEC_LAYERS = ['convnext','convnext','convnext']
_C.MODEL.DEC_DEPTHS = [3,3,3]

_C.MODEL.EXTRACTOR_NAME = "query_uv_feature"
_C.MODEL.PREDICTOR_NAME = "Regressor_Transformer_with_cat_feature"

_C.MODEL.CONV_DROP_PATH_RATE = 0.1
_C.MODEL.ATTEN_DROPOUT_RATE = 0.1
_C.MODEL.QUERY_EMB_DIM = 128
_C.MODEL.NUM_ATTEN_BLOCK = 4

_C.VAE = CN()
_C.VAE.NAME = "VAE" # VAE, Transformer_VAE
_C.VAE.LATENT_DIM = 256
_C.VAE.KLD_WEIGHT = 0.01


###################
#### DIFFUSION ####
###################

_C.DIFFUSION = CN()

_C.DIFFUSION.TYPE = "normal" # normal, improved, sde
_C.DIFFUSION.STEP = 1000
_C.DIFFUSION.MAX_RANK = 1 #  used for calculating end
_C.DIFFUSION.SIGMA = 1.0
_C.DIFFUSION.TARGET_MODE = "max"

# for normal DDPM
_C.DIFFUSION.START = 1e-5
_C.DIFFUSION.END = 2e-2

# for Improved DDPM
_C.DIFFUSION.S = 8e-3
_C.DIFFUSION.BIAS = 0.

# for classifier free guidance
_C.DIFFUSION.IMG_GUIDE = 1.0

# for retrieval-based evaluation
_C.DIFFUSION.STEP_EVAL = 500

###################
#### RETREIVE  ####
###################

_C.RETRIEVAL = CN()
_C.RETRIEVAL.RANK = 1

###################
###### OPTIM ######
###################

_C.OPTIM = CN()

_C.OPTIM.LR = 1e-4

_C.OPTIM.SCHEDULER = CN()
_C.OPTIM.SCHEDULER.GAMMA = 0.99
_C.OPTIM.SCHEDULER.STEP = 1000