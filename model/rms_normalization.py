import numpy as np
from typing import List


class Solution:
    def rms_norm(self, x: List[float], gamma: List[float], eps: float) -> List[float]:
        # Implement RMS Normalization (similar to LayerNorm but without mean centering or beta)
        # Normalize x, then scale by gamma
        # Return result rounded to 4 decimal places as a list
        x_arr=np.array(x,dtype=np.float64)
        gamma_arr=np.array(gamma,dtype=np.float64)
        x_sqr=np.square(x_arr)
        x_mean=np.mean(x_sqr)
        rms=np.sqrt(x_mean+eps)
        x_hat=x_arr/rms
        print(x_hat)
        print(gamma_arr)
        output=x_hat*gamma_arr
        print(output)
        return np.round(output,4)