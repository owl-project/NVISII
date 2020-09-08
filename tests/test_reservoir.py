#%%
import numpy as np
import random
import matplotlib.pyplot as plt

#%%
class Reservoir:
    def __init__(self):
        self.sample = None
        self.w_sum = 0.0
        self.M = 0.0
        self.W = 0.0

    def update(self, x_i, w_i, rnd):
        self.w_sum += w_i
        self.M = self.M + 1
        if rnd < w_i / self.w_sum:
            self.sample = x_i

reservoir = Reservoir()

# %%
lights = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 10000]
ground_truth = sum(lights)
reservoir = Reservoir()
samples = np.random.uniform(low=0.0, high = 1.0, size = 1000)

normal_result = [0]
reservoir_result = [0]
truth_result = [ground_truth]
for idx, x_i in enumerate(samples):
    addr = int(min(x_i * len(lights), len(lights) - 1))
    i = lights[addr] # the "reward" for that sample

    # p must be a value between 0 and 1, and represents the probability of getting i. 
    # Ideally it is proportional to the function we're sampling, but is usually a cheap approximation
    p = 1.0 / len(lights)

    # p_hat is perfectly proportional to the function we're sampling, but expensive to compute 
    p_hat = i
    
    # This is a running average that is not importance sampled
    intensity = i / p
    normal_result.append((intensity + idx * normal_result[-1]) / (idx + 1))

    # This is an importance sampled average using a weighted reservoir
    reservoir.update(x_i, p_hat / p, random.uniform(0, 1))
    
    # Note here that it's important that p_hat does not equal 0, as W would otherwise go to infinity
    addr = int(min(reservoir.sample * len(lights), len(lights) - 1))
    i = lights[addr]

    # Note here that p_hat must be updated to the new estimate before computing W
    p_hat = i
    reservoir.W = (1.0 / p_hat) * (1.0 / reservoir.M) * (reservoir.w_sum)
    intensity = i * reservoir.W
    reservoir_result.append((intensity + idx * reservoir_result[-1]) / (idx + 1)) 

    # ground truth
    truth_result.append(ground_truth)

plt.plot(normal_result)
plt.plot(reservoir_result)
plt.plot(truth_result)
plt.ylabel('Average of signal over time')
plt.show()
# %%
