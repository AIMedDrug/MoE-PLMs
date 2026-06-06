import torch

def check_gpus():

    if not torch.cuda.is_available():
        print("CUDA is not available. No GPUs detected.")
        return

    gpu_count = torch.cuda.device_count()
    print(f"Total GPUs detected: {gpu_count}")
    

    if gpu_count < 2:
        print(f"Need at least 2 GPUs, but found {gpu_count}.")
        return
    

    print("\nGPU details:")
    for i in range(2):
        # set GPU
        torch.cuda.set_device(i)

        torch.cuda.empty_cache()
        
        gpu_properties = torch.cuda.get_device_properties(i)
        
        available_mem, total_mem = torch.cuda.mem_get_info(i)
        
        available_mem_gb = available_mem / (1024 **3)
        total_mem_gb = total_mem / (1024** 3)
        
        print(f"\nGPU {i}:")
        print(f"  Name: {gpu_properties.name}")
        print(f"  Total memory: {total_mem_gb:.2f} GB")
        print(f"  Available memory: {available_mem_gb:.2f} GB")
        print(f"  Compute capability: {gpu_properties.major}.{gpu_properties.minor}")
        print(f"  CUDA device index: {i}")

if __name__ == "__main__":
    check_gpus()