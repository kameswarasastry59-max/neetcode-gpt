import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)
        """
        Subtract max(z) to prevent overflow from very large exponentials
        and improve numerical stability without changing the result.

        Example:
        z = [1000, 999, 998]

        Without subtraction:
        exp(1000) → very large number → overflow error

        With subtraction:
        z - max(z) = [0, -1, -2]
        exp([0, -1, -2]) = [1, 0.367, 0.135] → safe to compute

        The final softmax values remain the same, but calculations
        become stable and safe.
        """
        # A better way of this vectorized operation 
        z = z - np.max(z)
        exp_z = np.exp(z)
        ans = exp_z / np.sum(exp_z)

        return np.round(ans, 4)


         
        
