
import io
import base64
import torch
import zlib
from safetensors.torch import save, load

def serialize_tensor(tensor_or_dict, use_quantization: bool = True) -> str:
    """
    Serialize a PyTorch tensor (or dict of tensors) to a compressed, base64 encoded string
    using SAFETENSORS for security (no pickle/RCE).
    """
    # If input is a single tensor, wrap it in a dict
    if isinstance(tensor_or_dict, torch.Tensor):
        data_dict = {"data": tensor_or_dict}
    elif isinstance(tensor_or_dict, dict):
        data_dict = tensor_or_dict
    else:
        raise ValueError("Input must be a Tensor or a Dict of Tensors")
        
    # Ensure all values are tensors (safetensors requirement)
    # If we are doing quantization logic, we need to handle it before saving
    
    final_dict = {}
    
    # Quantization Logic
    if isinstance(tensor_or_dict, torch.Tensor) and use_quantization and tensor_or_dict.dtype in [torch.float32, torch.float16]:
        # Symmetric Quantization to INT8
        max_val = tensor_or_dict.abs().max()
        scale = max_val / 127.0 if max_val > 0 else torch.tensor(1.0, device=tensor_or_dict.device)
        quantized = (tensor_or_dict / scale).round().to(torch.int8)
        
        # Safetensors only stores tensors, so we store scale as a 0-d tensor
        final_dict["t"] = quantized
        final_dict["s"] = scale.view(1) # 0-d to 1-d for safety sometimes
        final_dict["q"] = torch.tensor([1], dtype=torch.int8) # Flag
    elif isinstance(tensor_or_dict, torch.Tensor):
         final_dict["data"] = tensor_or_dict
    else:
         # It's already a dict (e.g. state_dict)
         final_dict = data_dict
         
    # Serialize with Safetensors
    # Safetensors returns bytes directly
    try:
        safetensors_bytes = save(final_dict)
    except Exception as e:
        print(f"Safetensors save error: {e}")
        # Fallback or re-raise. Safetensors might fail on some dtypes or non-contiguous.
        # Ensure contiguous
        final_dict = {k: v.contiguous() for k, v in final_dict.items()}
        safetensors_bytes = save(final_dict)
        
    # Compress
    compressed_data = zlib.compress(safetensors_bytes)
    
    return base64.b64encode(compressed_data).decode('utf-8')

def deserialize_tensor(data: str) -> torch.Tensor:
    """
    Deserialize a base64 encoded string back to a PyTorch tensor using SAFETENSORS.
    Secure against RCE.
    """
    try:
        decoded_data = base64.b64decode(data)
        try:
            decompressed_data = zlib.decompress(decoded_data)
        except zlib.error:
            decompressed_data = decoded_data
            
        # Load from bytes
        loaded_dict = load(decompressed_data)
        
        # Check if it was our quantized format
        if "q" in loaded_dict and "t" in loaded_dict:
            quantized = loaded_dict["t"]
            scale = loaded_dict["s"]
            # Dequantize
            return quantized.to(torch.float32) * scale.item()
            
        # Check if it was a single tensor wrapper
        if "data" in loaded_dict and len(loaded_dict) == 1:
            return loaded_dict["data"]
            
        # Otherwise return the full dict (e.g. for weights)
        return loaded_dict
            
    except Exception as e:
        print(f"Error deserializing tensor (Security or Format): {e}")
        raise e
