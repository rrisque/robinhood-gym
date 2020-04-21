import gym
import numpy as np
import pandas as pandas
import robin_stocks
from gym import spaces

from robinhood_gym import config


class RobinhoodEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(RobinhoodEnv, self).__init__()

        robin_stocks.login(config.username, config.password)

        self.historicals = pandas.DataFrame(robin_stocks.stocks.get_historicals('MNST', span='year')).set_index(
            ['begins_at'])
        self.historicals.index = pandas.to_datetime(self.historicals.index)
        self.historicals = self.historicals[['open_price', 'close_price', 'high_price', 'low_price']].astype(
            dtype=float)

        self.stocks = 0
        self.net = 1000
        self.index = 0

        self.date = self.historicals.index[self.index]

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=self.historicals.min(), high=self.historicals.max())

    def step(self, action):
        self._take_action(action)
        self.index += 1

        obs = self._next_observation()
        reward = self.stocks * np.mean(obs) + self.net
        done = False

        if self.index >= self.historicals.index.shape[0]:
            done = True
        else:
            self.date = self.historicals.index[self.index]

        return obs, reward, done

    def _take_action(self, action):
        # buy
        if action == 1:
            print('buy')
            if self.net - self.price > 0:
                self.net -= self.price
                self.stocks += 1

        # sell
        if action == 2:
            print('sell')
            if self.stocks > 0:
                self.stocks -= 1
                self.net += self.price

    def reset(self):
        self.stocks = 0
        self.date = self.historicals.index[0]

        return self._next_observation()

    def _next_observation(self):
        obs = self.historicals.loc[self.date].values
        self.price = np.mean(obs)
        return obs

    def render(self, mode='human'):
        ...

    def close(self):
        ...


if __name__ == '__main__':
    env = RobinhoodEnv()
    env.reset()
    done = False
    while not done:
        obs, reward, done = env.step(env.action_space.sample())
        print(obs, reward)
