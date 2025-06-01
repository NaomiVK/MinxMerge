"""
LaMa Remove Object Node for ComfyUI
Based on https://github.com/advimman/lama
"""
import os
import torch
import torch.nn.functional as F
import hashlib
import urllib.request
import urllib.error
from tqdm import tqdm

import folder_paths
import comfy.model_management as model_management


lama = None
gpu = model_management.get_torch_device()
cpu = torch.device("cpu")
model_dir = os.path.join(folder_paths.models_dir, "lama")
model_url = "https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt"
model_sha = "344c77bbcb158f17dd143070d1e789f38a66c04202311ae3a258ef66667a9ea9"


def download_file(url: str, dst: str, sha256sum: str = None):
    """Downloads a file from URL to destination with optional SHA-256 verification."""
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    
    if os.path.isfile(dst):
        # Check if existing file has correct checksum
        if sha256sum:
            with open(dst, "rb") as f:
                file_hash = hashlib.sha256()
                while chunk := f.read(8192):
                    file_hash.update(chunk)
                if file_hash.hexdigest() == sha256sum:
                    return  # File exists and checksum matches
        else:
            return  # File exists and no checksum to verify
    
    # Download file
    print(f"Downloading LaMa model from {url} to {dst}")
    try:
        with tqdm(unit="B", unit_scale=True, unit_divisor=1024, miniters=1, desc="Downloading LaMa model") as t:
            def reporthook(blocknum, blocksize, totalsize):
                if t.total is None and totalsize > 0:
                    t.total = totalsize
                read_so_far = blocknum * blocksize
                t.update(max(0, read_so_far - t.n))
            
            urllib.request.urlretrieve(url, dst, reporthook=reporthook)
            
        # Verify checksum
        if sha256sum:
            with open(dst, "rb") as f:
                file_hash = hashlib.sha256()
                while chunk := f.read(8192):
                    file_hash.update(chunk)
                if file_hash.hexdigest() != sha256sum:
                    os.remove(dst)
                    raise ValueError(f"Downloaded file checksum mismatch. Expected: {sha256sum}")
                    
    except Exception as e:
        if os.path.isfile(dst):
            os.remove(dst)
        raise e


def ceil_modulo(x, mod):
    if x % mod == 0:
        return x
    return (x // mod + 1) * mod


def pad_tensor_to_modulo(img, mod):
    height, width = img.shape[-2:]
    out_height = ceil_modulo(height, mod)
    out_width = ceil_modulo(width, mod)
    return F.pad(img, pad=(0, out_width - width, 0, out_height - height), mode="reflect")


def load_model():
    global lama
    if lama is None:
        model_path = os.path.join(model_dir, "big-lama.pt")
        download_file(model_url, model_path, model_sha)
        
        lama = torch.jit.load(model_path, map_location="cpu")
        lama.eval()
    
    return lama


class LamaRemoveObject:
    """
    Remove objects from images using LaMa (Large Mask Inpainting) model.
    The model will be automatically downloaded on first use.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
            },
            "optional": {
                "device_mode": (["AUTO", "Prefer GPU", "CPU"],),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "remove_object"
    CATEGORY = "MinxMerge/Image"
    
    def remove_object(self, image: torch.Tensor, mask: torch.Tensor, device_mode="AUTO"):
        """
        Remove objects from image using LaMa inpainting.
        
        Args:
            image: Input image tensor (B, H, W, C)
            mask: Binary mask tensor (B, H, W) or (B, H, W, C)
            device_mode: Device selection mode
            
        Returns:
            Tuple containing the inpainted image tensor
        """
        if image.shape[0] != mask.shape[0]:
            raise ValueError("Image and mask must have the same batch size")
        
        # Select device based on mode
        device = gpu if device_mode != "CPU" else cpu
        
        # Load model
        model = load_model()
        model.to(device)
        
        try:
            inpainted = []
            orig_h = image.shape[1]
            orig_w = image.shape[2]
            
            for i, img in enumerate(image):
                # Convert image from (H, W, C) to (C, H, W) and add batch dimension
                img = img.permute(2, 0, 1).unsqueeze(0)
                
                # Process mask
                if len(mask.shape) == 4:
                    # If mask has channel dimension, take first channel
                    msk = mask[i, :, :, 0].detach().cpu()
                else:
                    # Mask is already (B, H, W)
                    msk = mask[i].detach().cpu()
                
                # Convert mask to binary (0 or 1)
                msk = (msk > 0) * 1.0
                msk = msk.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
                
                # Pad to multiple of 8 (required by model)
                src_image = pad_tensor_to_modulo(img, 8).to(device)
                src_mask = pad_tensor_to_modulo(msk, 8).to(device)
                
                # Run inpainting
                with torch.no_grad():
                    res = model(src_image, src_mask)
                
                # Convert back to (H, W, C) and crop to original size
                res = res[0].permute(1, 2, 0).detach().cpu()
                res = res[:orig_h, :orig_w]
                
                inpainted.append(res)
            
            return (torch.stack(inpainted),)
            
        finally:
            # Move model back to CPU if in AUTO mode to save GPU memory
            if device_mode == "AUTO":
                model.to(cpu)


NODE_CLASS_MAPPINGS = {
    "LamaRemoveObject": LamaRemoveObject,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LamaRemoveObject": "LaMa Remove Object",
}