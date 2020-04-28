import time
import pyvips
from PIL import Image
import numpy as np
import cv2
import torch

img_f = '/u/nimit/Documents/robomaster/habitat2robomaster/test222/train/Maryhill-train_ddppo-000001/rgb_0010.png'

print('\nPIL')
for _ in range(10):
    start=time.time()
    x = torch.Tensor(np.uint8(Image.open(img_f)))
    print(time.time()-start)

print('\npyvips')
for _ in range(10):
    start=time.time()
    image = pyvips.Image.new_from_file(img_f, memory=True)
    mem_img = image.write_to_memory()
    #torch.Tensor(mem_img)
    print(time.time()-start)

print('\nopencv')
for _ in range(10):
    start=time.time()
    x = torch.Tensor(cv2.imread(img_f, cv2.IMREAD_UNCHANGED))
    #print(x.shape)
    print(time.time()-start)
