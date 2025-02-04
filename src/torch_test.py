import torch
print(torch.cuda.is_available())  # Should return True
print(torch.cuda.device_count())  # Should return the number of GPUs
print(torch.cuda.current_device())  # Should return the current GPU ID
print(torch.cuda.get_device_name(0))  # Should return the GPU name