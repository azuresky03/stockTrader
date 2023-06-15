from gym.envs.registration import register

register(
    id='stock_env/StockTradingEnv-v0',
    entry_point='stock_env.env:StockTradingEnv',
    max_episode_steps=10000,
)

register(
    id='stock_env/StockTradingEnv-v1',
    entry_point='stock_env.env1:StockTradingEnv',
    max_episode_steps=15000,
)

register(
    id='stock_env/StockTradingEnv-v2',
    entry_point='stock_env.env2:StockTradingEnv',
    max_episode_steps=15000,
)

register(
    id='stock_env/StockTradingEnv-v3',
    entry_point='stock_env.env3:StockTradingEnv',
    max_episode_steps=15000,
)