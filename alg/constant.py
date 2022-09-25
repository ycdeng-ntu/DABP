import torch

VAR_ID = [1, 0, 0, 0]
FUN_ID = [0, 1, 0, 0]
V2F_ID = [0, 0, 1, 0]
F2V_ID = [0, 0, 0, 1]
MAX_COLOR_NUM = 100
gpu_id = 0
device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
scale = 50
SPLIT_RATIO = .95