"""
PyTorch Compatibility Layer for Jetson Nano

This module provides compatibility wrappers for code that needs to run
on both PyTorch 1.x (Jetson Nano) and PyTorch 2.x (modern systems).

JetPack 4.6.x ships with PyTorch 1.10-1.12, which uses:
- torch.cuda.amp.autocast() instead of torch.amp.autocast('cuda')
- torch.cuda.amp.GradScaler() instead of torch.amp.GradScaler('cuda')

Usage:
    from utils.torch_compat import get_autocast, get_grad_scaler
    
    scaler = get_grad_scaler() if use_amp else None
    with get_autocast(enabled=use_amp):
        output = model(input)
"""

import torch

# Check PyTorch version
PYTORCH_VERSION = tuple(int(x) for x in torch.__version__.split('.')[:2])
IS_PYTORCH_2 = PYTORCH_VERSION[0] >= 2


def get_autocast(device_type: str = 'cuda', enabled: bool = True):
    """
    Get appropriate autocast context manager for the PyTorch version.
    
    Args:
        device_type: Device type ('cuda' or 'cpu')
        enabled: Whether to enable autocast
        
    Returns:
        Autocast context manager
    """
    if IS_PYTORCH_2:
        return torch.amp.autocast(device_type, enabled=enabled)
    else:
        # PyTorch 1.x
        if device_type == 'cuda':
            return torch.cuda.amp.autocast(enabled=enabled)
        else:
            # CPU autocast not available in PyTorch 1.x
            return torch.cuda.amp.autocast(enabled=False)


def get_grad_scaler(device_type: str = 'cuda'):
    """
    Get appropriate GradScaler for the PyTorch version.
    
    Args:
        device_type: Device type ('cuda')
        
    Returns:
        GradScaler instance
    """
    if IS_PYTORCH_2:
        return torch.amp.GradScaler(device_type)
    else:
        # PyTorch 1.x
        return torch.cuda.amp.GradScaler()


def is_jetson() -> bool:
    """Check if running on Jetson platform."""
    try:
        with open('/etc/nv_tegra_release', 'r') as f:
            return True
    except FileNotFoundError:
        return False


def get_device_info() -> dict:
    """Get device information for logging."""
    info = {
        'pytorch_version': torch.__version__,
        'is_pytorch_2': IS_PYTORCH_2,
        'cuda_available': torch.cuda.is_available(),
        'is_jetson': is_jetson(),
    }
    
    if torch.cuda.is_available():
        info['cuda_version'] = torch.version.cuda
        info['device_name'] = torch.cuda.get_device_name(0)
        info['device_memory_gb'] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    
    return info


# Print info on import for debugging
if __name__ == "__main__":
    print("PyTorch Compatibility Layer")
    print("=" * 40)
    for k, v in get_device_info().items():
        print(f"  {k}: {v}")
