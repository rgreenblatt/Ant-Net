import numpy as np
import matplotlib.pyplot as plt
import sys

data = np.load(sys.argv[1])


new_data = np.roll(data, -25, axis=0)
data = np.roll(data, 26, axis=0)

print(np.array_equal(data, new_data))

img = plt.imshow(data, interpolation='nearest')
img.set_cmap('hot')
plt.axis('off')
plt.show()


#new_data = np.roll(new_data, 10, axis=1)

img = plt.imshow(new_data, interpolation='nearest')
img.set_cmap('hot')
plt.axis('off')
plt.show()
