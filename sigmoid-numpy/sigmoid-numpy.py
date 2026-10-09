import numpy as np
from typing import Union

def sigmoid(x: Union[list, float, np.ndarray]) -> Union[np.ndarray, float]:
    """
    Returns the sigmoid value for a scalar, list, or NumPy array.
    """
    # Converting to an array ensures uniform behavior for lists/arrays
    x_arr = np.asarray(x) 
    result = 1 / (1 + np.exp(-x_arr))
    
    # If the original input was a scalar float/int, return a scalar float
    if isinstance(x, (int, float)):
        return float(result)
        
    return result
