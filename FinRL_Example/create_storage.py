import sqlite3
import optuna

# 创建一个数据库文件
conn = sqlite3.connect("db.sqlite3")
conn.close()

# 创建一个RDBStorage对象
storage = optuna.storages.RDBStorage("sqlite:///db.sqlite3")

# 使用storage创建study
models = ["a2c", "ddpg", "td3", "sac", "ppo"]
for model in models:
    study = optuna.create_study(
        direction="maximize",
        study_name=model,
        storage="sqlite:///db.sqlite3",
        load_if_exists=True,
    )


