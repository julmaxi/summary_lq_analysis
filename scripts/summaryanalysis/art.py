import random
import numpy as np

def paired_approximate_randomization_test(x, y, n=1000):
    assert x.shape == y.shape
    x_sum = x.sum()
    y_sum = y.sum()
    original_difference = abs(x_sum - y_sum)
    
    num_successes = 0
    for _ in range(n):
        mask = np.random.choice(a=[True, False], size=x.shape)
        x_sample_sum = x[~mask].sum()
        y_sample_sum = y[mask].sum()
        
        sample_a_sum = (x_sum - x_sample_sum) + (y_sum - y_sample_sum)
        sample_b_sum = x_sample_sum + y_sample_sum
        
        sample_difference = abs(sample_a_sum - sample_b_sum)
        if sample_difference >= original_difference:
            num_successes += 1
    
    return (num_successes + 1) / (n + 1)
