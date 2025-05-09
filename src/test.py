import torch
print("Built with CUDA:", torch.version.cuda)             # e.g. “11.8” or “12.4” :contentReference[oaicite:11]{index=11}
print("CUDA available?", torch.cuda.is_available())       # should print True :contentReference[oaicite:12]{index=12}
print("GPU count:", torch.cuda.device_count())            # usually 1
print("GPU name:", torch.cuda.get_device_name(0))         # e.g. “NVIDIA GeForce …”
