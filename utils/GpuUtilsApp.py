# utils/GPUUtilsApp.py
import torch

def limit_gpu_memory(gpu_memory_limit=0.65):
    if torch.cuda.is_available():
        total_memory = torch.cuda.get_device_properties(0).total_memory
        reserved_memory = int(total_memory * gpu_memory_limit)
        torch.cuda.set_per_process_memory_fraction(gpu_memory_limit)
    else:
        print("CUDA не доступна. Используется CPU.")

def get_device():
    if torch.cuda.is_available():
        return 'cuda'
    else:
        return 'cpu'
