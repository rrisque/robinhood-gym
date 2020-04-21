import gym
import numpy as np
import pandas as pandas
import robin_stocks
from gym import spaces

from robinhood_gym import config


class RobinhoodEnv(gym.Env):
    """
    Gym environment for playing with the stock market. Requires a Robinhood account.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(RobinhoodEnv, self).__init__()

        robin_stocks.login(config.username, config.password)

        buying_power = 1000

        self.stocks = 0
        self.net = buying_power
        self.starting = buying_power
        self.index = 0

        self.action_space = spaces.Box(low=np.array([0]), high=np.array([0]), dtype=int)

    def step(self, action):
        self._take_action(action)
        self.index += 1

        obs = self._next_observation()
        reward = self.stocks * np.mean(obs) + self.net - self.starting
        done = False

        if self.index >= self.historicals.index.shape[0]:
            done = True
        else:
            self.date = self.historicals.index[self.index]

        return obs, reward, done

    def _take_action(self, action):
        self.stocks += action[0]
        self.net -= self.price * action[0]

        # update valid transactions possible after buying or selling
        self.action_space = spaces.Box(low=np.array([-self.stocks]), high=np.array([self.net // self.price]), dtype=int)

    def reset(self, symbol=None):
        self.symbol = symbol
        self.stocks = 0
        self.historicals = self._get_hitoricals()
        self.date = self.historicals.index[0]
        self.observation_space = spaces.Box(low=self.historicals.min(), high=self.historicals.max())

        return self._next_observation()

    def _next_observation(self):
        obs = self.historicals.loc[self.date].values
        self.price = np.mean(obs)
        return obs

    def _get_hitoricals(self):
        """
        Returns weekly historical pricing data as a pandas DataFrame.
        """
        historicals = pandas.DataFrame(robin_stocks.stocks.get_historicals(self.symbol, span='week')).set_index(
            ['begins_at'])
        historicals.index = pandas.to_datetime(historicals.index)
        historicals = historicals[['open_price', 'close_price', 'high_price', 'low_price']].astype(
            dtype=float)
        return historicals

    def _get_news(self):
        """
        Returns any news data.
        """
        news = robin_stocks.get_news(self.symbol)
        print(news)

    def render(self, mode='human'):
        ...

    def close(self):
        ...


if __name__ == '__main__':
    env = RobinhoodEnv()
    env.reset(symbol='BYND')
    done = False
    while not done:
        obs, reward, done = env.step(env.action_space.sample())
        print(reward)
