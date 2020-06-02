import torch

C = 2
def make_onehot(semantic):
    """
        input:  torch (B,256,256,1), dtype: torch.uint8
        output: torch (B,256,256,C), dtype: torch.float
    """
    onehot = torch.zeros((*semantic.shape, C), dtype=torch.float)
    onehot[..., 0] = torch.as_tensor(semantic==2, dtype=torch.float)
    onehot[..., 1] = torch.as_tensor((semantic!=2)&(semantic!=17)&(semantic!=28), dtype=torch.float)
    return onehot
