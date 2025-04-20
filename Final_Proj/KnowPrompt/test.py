import torch

gpu_count = torch.cuda.device_count()
print("gpu_count: ", gpu_count)
print("cuda: ", torch.cuda.is_available())
# print gpu info and cuda memory
if torch.cuda.is_available():
    for i in range(gpu_count):
        print(torch.cuda.get_device_name(i))
