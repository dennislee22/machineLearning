import numpy as np
from numpy import asarray
from PIL import Image
import matplotlib.pylab as plt
import sys

image = str(sys.argv[1])
img = Image.open(image)
print("Size:" , img.size)
print("Format:" , img.format)
print("Mode:" , img.mode)
raw = asarray(img)
print(raw[400:403, 500:503])
#numpyraw = np.load(img)
#new_img = Image.fromarray(numpyraw)
