import numpy as np



conv = np.array([[1.5,2.5,1.2],
                 [1.0,1.5,0.8],
                 [0.9,1.2,1.4]])

summation = np.sum(conv)
print(conv)
print(summation)

print(conv/summation)
stoc = conv/summation