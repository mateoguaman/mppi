import torch

def dict_to_device(dict, device):
    '''Puts tensor elements of dictionary into device.

    Args:
        dict:
            Input dictionary
        device:
            String specifying device that tensor will be put into (usually "cuda", "cuda:0", or "cpu")
    '''
    for k, v in dict.items():
        if torch.is_tensor(v):
            dict[k] = v.to(device)
    return dict