import gym
class Bound:
    def __init__(self, lower, upper):
        self.lower = lower
        self.upper = upper

class CartPoleSimulation:
    def __init__(self):
        self.env = gym.make('CartPole-v0')
        self.solved_time = 200 - 1 #step needed to say the episode is solved
        self.streak_to_end = 100 #number consecutive success needed to consider the proble solved
        self.max_episode = 500

    def run(self, model, obs_engineering):
        steps = []
        num_streaks = 0
        episode = 0
        while 1:
            episode +=1
            model.update_rate(episode)
            # reset env
            obs = self.env.reset()

            # state of the pole
            model.current_state = obs_engineering.apply(obs)

            step = 0
            while 1:
                step += 1
                self.env.render()

                action = model.select_action()
                obs, reward, done, _ = self.env.step(action)
                next_state = obs_engineering.apply(obs)
                model.update(next_state, action, reward)

                if done:
                    steps.append(step)
                    if (step >= self.solved_time):
                        num_streaks += 1
                    else:
                        num_streaks = 0
                    break

            print("episode: {episode}, step: {step}, streak: {streak}".format(episode=episode, step=step, streak=num_streaks))

            if num_streaks > self.streak_to_end or episode > self.max_episode:
                break

        return steps
