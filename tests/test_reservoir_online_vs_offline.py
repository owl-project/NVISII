#%%
import numpy as np
import random
import matplotlib.pyplot as plt

#%%
class OfflineReservoir:
    def __init__(self):
        self.sample = None
        self.w_sum = 0.0
        self.M = 0.0
        # compute W using (1/p_hat) * (w_sum / M)

    def update(self, x_i, w_i, rnd):
        self.w_sum += w_i
        self.M = self.M + 1
        if rnd < w_i / self.w_sum:
            self.sample = x_i

class OnlineReservoir:
    def __init__(self, limit):
        self.sample = None
        self.w_avg = 0.0
        self.M = 0.0
        self.M_LIMIT = limit
        # compute W using (1/p_hat) * (w_avg)

    def update(self, x_i, w_i, rnd):
        self.M = min(self.M + 1, self.M_LIMIT)
        self.w_avg = self.w_avg * ((self.M-1.0) / self.M) + w_i * (1.0 / self.M) 
        if rnd < w_i / (self.w_avg * self.M):
            self.sample = x_i

# %%
lights = [1, 10]
ground_truth = sum(lights)
offline_reservoir = OfflineReservoir()
online_reservoir = OnlineReservoir(512)
samples = np.random.uniform(low=0.0, high = 1.0, size = 100000)

normal_result = [0]
offline_reservoir_result = [0]
online_reservoir_result = [0]
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
    offline_reservoir.update(x_i, p_hat / p, random.uniform(0, 1))
    online_reservoir.update(x_i, p_hat / p, random.uniform(0, 1))
    
    # Note here that it's important that p_hat does not equal 0, as W would otherwise go to infinity
    addr = int(min(offline_reservoir.sample * len(lights), len(lights) - 1))
    i = lights[addr]
    p_hat = i
    offline_reservoir.W = (1.0 / p_hat) * (1.0 / offline_reservoir.M) * (offline_reservoir.w_sum)
    intensity = i * offline_reservoir.W
    offline_reservoir_result.append((intensity + idx * offline_reservoir_result[-1]) / (idx + 1)) 

    # online test
    addr = int(min(online_reservoir.sample * len(lights), len(lights) - 1))
    i = lights[addr]
    p_hat = i
    online_reservoir.W = (1.0 / p_hat) * online_reservoir.w_avg
    intensity = i * online_reservoir.W
    online_reservoir_result.append((intensity + idx * online_reservoir_result[-1]) / (idx + 1)) 

    # ground truth
    truth_result.append(ground_truth)

plt.rcParams["figure.figsize"] = (10,3)
plt.ylim((10, 12))
plt.plot(normal_result, label='running average')
plt.plot(offline_reservoir_result, label='offline reservoir')
plt.plot(online_reservoir_result, label='online reservoir')
plt.plot(truth_result, label='ground truth')
plt.ylabel('Average of signal over time')
plt.legend()
plt.show()
# %%
