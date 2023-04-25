import numpy as np
import random
import matplotlib.pyplot as plt
a = 3+5
b = 6*a^3
c = np.random.random((256,256))
print('old->', a, 'new->' ,b)
plt.imshow(c)
plt.savefig('/cluster/home/quintep/outputs/random.png')