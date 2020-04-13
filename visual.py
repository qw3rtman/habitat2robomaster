import numpy as np
from pathlib import Path
import cv2

root = Path('test/000001')
for x in root.glob('depth_*.npy'):
    cv2.imshow('depth', np.load(x))
    cv2.waitKey(0)
