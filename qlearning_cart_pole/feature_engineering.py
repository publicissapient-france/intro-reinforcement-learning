import math

class KeepSame:
    def apply(self, obs):
        return obs

class Bucketizer:
    def __init__(self, state_bounds, num_buckets):
        self.state_bounds = state_bounds
        self.num_buckets = num_buckets
        self.state_width = [ bound.upper - bound.lower for bound in state_bounds]


    def apply(self, obs):
        bucket = []
        for i in range(len(obs)):
            if self.num_buckets[i] == 1:
                bucket.append(0)
            elif obs[i] <= self.state_bounds[i].lower:
                bucket.append(0)
            elif obs[i] > self.state_bounds[i].upper:
                bucket.append(self.num_buckets[i] - 1) #-1 cause index start at 0 in array
            else:
                zeroed = obs[i] - self.state_bounds[i].lower
                bucket_width = self.state_width[i] / float(self.num_buckets[i])
                bucket.append(math.floor(zeroed / bucket_width) - 1) #-1 cause index start at 0 in array
        return tuple(bucket)
