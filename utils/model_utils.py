import torch

# Reference: https://github.com/facebookresearch/moco/blob/main/moco/builder.py
@torch.no_grad()
def copy_param(src, dest):
    # Assumes src and dest are the same model
    for src_param, dest_param in zip(src.parameters(), dest.parameters()):
        dest_param.data.copy_(src_param.data)  # initialize
        dest_param.requires_grad = False  # not updated by gradient

@torch.no_grad()
def copy_param_and_buffer(src, dest):
    # Assumes src and dest are the same model
    src_state_dict = src.state_dict()
    dest_state_dict = dest.state_dict()
    assert list(src_state_dict.keys()) == list(dest_state_dict.keys()), f"{list(src_state_dict.keys())} != {list(dest_state_dict.keys())}" 
    for key in src_state_dict.keys():
        dest_state_dict[key].data.copy_(src_state_dict[key])  # initialize
        dest_state_dict[key].requires_grad = False  # not updated by gradient

@torch.no_grad()
def set_no_grad(dest):
    for dest_param in dest.parameters():
        dest_param.requires_grad = False  # not updated by gradient

@torch.no_grad()
def momentum_update_param(src, dest, m):
    # Assumes src and dest are the same model
    for src_param, dest_param in zip(src.parameters(), dest.parameters()):
        dest_param.data = dest_param.data * m + src_param.data * (1.0 - m)

@torch.no_grad()
def momentum_update_param_and_buffer(src, dest, m):
    # Assumes src and dest are the same model
    src_state_dict = src.state_dict()
    dest_state_dict = dest.state_dict()
    for key in src_state_dict.keys():
        dest_state_dict[key].data.copy_(dest_state_dict[key].data * m + src_state_dict[key].data * (1.0 - m))  # initialize
