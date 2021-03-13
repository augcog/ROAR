import numpy as np
from scipy.sparse import dok_matrix

S = dok_matrix((10000, 10000), dtype=np.float32)
S[0:10, 0:10] = 0.5
print(type(S))