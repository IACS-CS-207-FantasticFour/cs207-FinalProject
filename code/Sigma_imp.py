from AutoDiff import AutoDiff
import numpy as np


pi = 3.14
T_t = 10
S = 100
C = 100



K = AutoDiff(110, 1)

f = np.sqrt(2*pi/T_t) * (1/(S + K)) *  (  C - ((S - K)/2) + np.sqrt( (C - (S-K)/2)**2 - (S -K)**2/pi  ) )

print(f.val, f.derv)

