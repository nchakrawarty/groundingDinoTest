import torch
print(torch.__version__)           # Should indicate a CUDA-enabled version, e.g., 2.5.1+cu121
print(torch.cuda.is_available())   # Should return True
print(torch.version.cuda)          # Should match your CUDA version or nearby compatibility
print(torch.cuda.device_count())  # Should be > 0


# Test if CUDA is being used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Perform a simple tensor operation on GPU
x = torch.rand(1000, 1000).to(device)
y = torch.rand(1000, 1000).to(device)
z = x * y  # This operation will run on the GPU if available

print("CUDA is working and tensor operation is performed on:", device)
