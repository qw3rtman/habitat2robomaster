import torch

C = 2#1
def make_onehot(semantic):
    """
        input:  torch (B,H,W,1), dtype: torch/np.uint8
        output: torch (B,H,W,C), dtype: torch.float
    """
    onehot = torch.zeros((*semantic.shape, C), dtype=torch.float)
    onehot[..., 0] = torch.as_tensor(semantic==2, dtype=torch.float)
    onehot[..., 1] = torch.as_tensor((semantic!=2)&(semantic!=17)&(semantic!=28), dtype=torch.float)
    return onehot
