import numpy as np
import matplotlib.pyplot as plt
import sys

data = np.load(sys.argv[1])
img = plt.imshow(data, interpolation='nearest')
img.set_cmap('hot')
plt.axis('off')
plt.show()
