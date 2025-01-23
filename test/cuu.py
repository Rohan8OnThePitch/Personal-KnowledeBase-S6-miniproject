import torch
import torchaudio
import torchvision

# Check if CUDA is available
cuda_available = torch.cuda.is_available()
cuda_device = torch.cuda.current_device() if cuda_available else None
print(f"CUDA Available: {cuda_available}")
print(f"CUDA Device: {cuda_device}")

# Check PyTorch version
print(f"PyTorch Version: {torch.__version__}")
print(f"torchaudio Version: {torchaudio.__version__}")
print(f"torchvision Version: {torchvision.__version__}")

# Check if your GPU is detected
print(f"CUDA Device Name: {torch.cuda.get_device_name(cuda_device) if cuda_available else 'N/A'}")
