from gym.envs.registration import register

register(
    id='robinhood-v0',
    entry_point='robinhood_gym.envs:RobinhoodEnv',
)
