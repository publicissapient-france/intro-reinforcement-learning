import math
import matplotlib.pyplot as plt
import gym

from models import DummyCartPole
from feature_engineering import KeepSame
from simulation import *


sim = CartPoleSimulation()
states_dim = (1, 1, 6, 5) #(x, x_v, theta, theta_v)
num_actions = sim.env.action_space.n #(left, right)


obs_engineering = KeepSame()
model = DummyCartPole()


if __name__ == "__main__":
    sim.max_episode = 50
    steps = sim.run(model, obs_engineering)
    plt.plot(steps)
    plt.show()
