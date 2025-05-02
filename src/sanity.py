import torch, os
print("CONDA_PREFIX =", os.environ.get("CONDA_PREFIX"))
print("CUDA available? ", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU count:", torch.cuda.device_count())
    print("Current device:", torch.cuda.current_device())
    print("Device name:", torch.cuda.get_device_name(0))
    x = torch.randn(3,3).cuda()
    print("Tensor on CUDA:", x.device)
else:
    print("Running on CPU only")
