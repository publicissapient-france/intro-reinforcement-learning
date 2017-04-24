import math
import matplotlib.pyplot as plt
import gym

from models import QLearning
from feature_engineering import Bucketizer
from simulation import *


sim = CartPoleSimulation()
states_dim = (1, 1, 6, 5) #(x, x_v, theta, theta_v)
num_actions = sim.env.action_space.n #(left, right)

X_BOUND = Bound(0, 0)
X_V_BOUND = Bound(0, 0)
THETA_BOUND = Bound(-0.5, 0.5)
THETA_V_BOUND = Bound(-math.radians(50), math.radians(50))
states_bounds = [X_BOUND,X_V_BOUND, THETA_BOUND, THETA_V_BOUND]


alpha_bound = Bound(0.001, 0.5)
epsilon_bound = Bound(0.01, 1)
gamma = 0.99

obs_engineering = Bucketizer(states_bounds, states_dim)
model = QLearning(states_dim, num_actions, alpha_bound, epsilon_bound, gamma)

if __name__ == "__main__":
    steps = sim.run(model, obs_engineering)
    print(model.q_table)
    plt.plot(steps)
    plt.show()
