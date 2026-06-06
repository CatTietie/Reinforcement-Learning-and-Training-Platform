"""
超参数搜索空间定义与采样工具。
"""

DEFAULT_SEARCH_SPACE = {
    "lr": {"type": "float", "low": 5e-4, "high": 5e-3, "log": True},
    "clip_epsilon": {"type": "float", "low": 0.1, "high": 0.3, "log": False},
    "lstm_hidden_size": {"type": "categorical", "choices": [64, 128, 256]},
}


def sample_from_space(trial, search_space=None):
    """
    使用Optuna trial对象从搜索空间采样超参数。

    Args:
        trial: optuna.Trial实例
        search_space: 搜索空间字典，默认使用DEFAULT_SEARCH_SPACE

    Returns:
        dict: 采样得到的超参数组合
    """
    if search_space is None:
        search_space = DEFAULT_SEARCH_SPACE

    params = {}
    for name, spec in search_space.items():
        param_type = spec["type"]
        if param_type == "float":
            params[name] = trial.suggest_float(
                name,
                spec["low"],
                spec["high"],
                log=spec.get("log", False),
            )
        elif param_type == "int":
            params[name] = trial.suggest_int(
                name,
                spec["low"],
                spec["high"],
                log=spec.get("log", False),
            )
        elif param_type == "categorical":
            params[name] = trial.suggest_categorical(name, spec["choices"])
        else:
            raise ValueError(f"Unknown search space type: {param_type}")

    return params
