import numpy as np
import math
import random

class QLearning:
    def __init__(self, states_dim, num_actions, alpha_bound, epsilon_bound, gamma):
        self.num_actions = num_actions
        self.q_table = np.zeros(states_dim + (num_actions,))
        self.alpha_bound = alpha_bound #0.001 0.5
        self.epsilon_bound = epsilon_bound #0.01 1
        self.gamma = gamma
        self.alpha = alpha_bound.upper
        self.epsilon = epsilon_bound.upper
        self.current_state = [0] * len(states_dim)

    def update(self, next_state, action, reward):
        self.q_table[self.current_state + (action,)] += self.alpha * (reward + self.gamma * np.amax(self.q_table[next_state]) - self.q_table[self.current_state + (action,)])
        self.current_state = next_state

    def select_action(self):
        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        else:
            return np.argmax(self.q_table[self.current_state])

    def update_rate(self, episode):
        self.update_alpha(episode)
        self.update_epsilon(episode)

    def update_alpha(self, episode):
        self.alpha = max(self.alpha_bound.lower, min(self.alpha_bound.upper, 1.0 - math.log10(episode/25)))

    def update_epsilon(self, episode):
        self.epsilon = max(self.epsilon_bound.lower, min(self.epsilon_bound.upper, 1.0 - math.log10(episode/25)))


class DummyCartPole:
    def __init__(self):
        self.current_state = [0] * 4

    def update(self, next_state, action, reward):
        self.current_state = next_state

    def select_action(self):
        if self.current_state[2] < 0:
            return 0
        else:
            return 1

    def update_rate(self, episode):
        return 0
