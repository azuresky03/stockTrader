import pandas as pd

import optuna

from stable_baselines3 import A2C, PPO, DDPG, TD3, SAC
from stable_baselines3.common.logger import configure

from finrl.meta.preprocessor.preprocessors import data_split
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.agents.stablebaselines3.models import DRLAgent, DRLEnsembleAgent
from finrl.plot import backtest_stats
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader

from finrl import config
from finrl import config_tickers
from finrl.main import check_and_make_directories
from finrl.config import (
    DATA_SAVE_DIR,
    TRAINED_MODEL_DIR,
    TENSORBOARD_LOG_DIR,
    RESULTS_DIR,
)

check_and_make_directories(
    [DATA_SAVE_DIR, TRAINED_MODEL_DIR, TENSORBOARD_LOG_DIR, RESULTS_DIR]
)

TRAIN_START_DATE = "2014-01-01"
TRAIN_END_DATE = "2021-01-01"
TEST_START_DATE = "2021-01-02"
TEST_END_DATE = "2023-05-01"
TIME_INTERVAL = "1D"

INDICATORS = [
    "macd",
    "boll",
    "rsi",
    "ppo",
    "log-ret",
    "vwma",
    "adx",
    "mfi",
]


data = pd.read_csv(DATA_SAVE_DIR + "/processed_data.csv")
train = data_split(data, TRAIN_START_DATE, TRAIN_END_DATE)
test = data_split(data, TEST_START_DATE, TEST_END_DATE)


stock_dimension = len(train.tic.unique())
state_space = 1 + 2 * stock_dimension + len(INDICATORS) * stock_dimension
buy_cost_list = sell_cost_list = [0.001] * stock_dimension
num_stock_shares = [0] * stock_dimension

env_kwargs = {
    "hmax": 100,
    "initial_amount": 100000,
    "num_stock_shares": num_stock_shares,
    "buy_cost_pct": buy_cost_list,
    "sell_cost_pct": sell_cost_list,
    "state_space": state_space,
    "stock_dim": stock_dimension,
    "tech_indicator_list": INDICATORS,
    "action_space": stock_dimension,
    "reward_scaling": 1e-4,
}
stock_env = StockTradingEnv(df=train, **env_kwargs)


def model_train(model_name, env, model_kwargs):
    # model training
    agent = DRLAgent(env)
    model = agent.get_model(model_name=model_name, model_kwargs=model_kwargs)
    # set up logger
    tmp_path = RESULTS_DIR + f"/{model_name}"
    new_logger_ppo = configure(tmp_path, ["tensorboard", "stdout"])
    model.set_logger(new_logger_ppo)

    trained_model = agent.train_model(
        model=model, tb_log_name=model_name, total_timesteps=50000
    )
    return trained_model


def model_eval(model_name, env, trained_model):
    df_account_value, df_actions = DRLAgent.DRL_prediction(
        model=trained_model, environment=env
    )
    stats = backtest_stats(account_value=df_account_value)
    print(f"=============={model_name} Results===========")
    print(stats)

    # select the performance metric for tuning
    cumulative_returns = stats.loc["Cumulative returns"]
    sharpe_ratio = stats.loc["Sharpe ratio"]
    performance_metric = cumulative_returns + sharpe_ratio
    return performance_metric


def objective(trial):
    n_steps = trial.suggest_int("n_steps", 64, 256)
    ent_coef = trial.suggest_loguniform("ent_coef", 1e-3, 1e-1)
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-3)
    batch_size = trial.suggest_categorical(
        "batch_size", [32, 64, 128, 256, 512, 1024, 2048]
    )

    model_kwargs = {
        "n_steps": n_steps,
        "ent_coef": ent_coef,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
    }
    trained_model = model_train("ppo", stock_env, model_kwargs)
    env = StockTradingEnv(df=test, **env_kwargs)
    return model_eval("ppo", env, trained_model)


if __name__ == "__main__":
    study = optuna.create_study(
        direction="maximize",
        study_name="ppo",
        storage="sqlite:///stock.db",
        load_if_exists=True,
    )
    best_params = study.optimize(objective, n_trials=10)
    best_params = study.best_params
    print(f"Best params: {best_params}")

    trained_model = model_train("ppo", stock_env, best_params)
    env = StockTradingEnv(df=test, **env_kwargs)
    model_eval("ppo", env, trained_model)
