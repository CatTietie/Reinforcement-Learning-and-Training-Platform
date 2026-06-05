# 强化学习训练平台需求计划文档

## 1. 项目概述
本平台旨在为强化学习算法研究提供一套集实验配置、训练执行、指标监控与结果对比于一体的工程化系统。系统通过 YAML 配置驱动实验，支持多种主流算法（PPO、SAC 等），兼容 Gym 标准环境接口，并内置实验数据库与日志体系，最终通过前端面板实现训练过程的可视化与多实验对比。

### 1.1 核心目标
- 提供 CLI 命令行工具，支持一键启动、断点续跑和日志回溯。
- 实现严格遵循已有接口规范的训练核心，确保算法逻辑（GAE、LSTM 隐状态）的正确性与可复现性。
- 建立实验元数据存储，便于按实验 ID 查询、对比和复现历史运行。
- 架构上支持前端监控面板实时展示损失曲线、回报曲线，并为多用户协作预留扩展。

### 1.2 技术栈
- **CLI 层**：Python Click
- **训练引擎**：Python + PyTorch
- **环境接口**：Gymnasium（Gym 0.26+ 兼容）
- **数据库**：PostgreSQL（实验元数据）
- **日志系统**：Python logging + 结构化输出（JSON lines）
- **前端**：React（不在 20% 交付范围内，但需预留 API 设计）
- **模型仓库**：本地文件系统 / 可选云存储挂载

## 2. 系统模块与目录结构
系统划分为以下核心模块，各模块以独立文件或目录组织，严格遵循接口约定。

```
RL-Experiment-Platform/
├── cli.py                    # CLI 入口（Click）
├── configs/                  # 示例配置文件
├── env.py                    # 环境接口定义（基类/示例环境）
├── policy.py                 # 策略网络定义（含 LSTM）
├── train.py                  # 训练主循环与 GAE 实现
├── models/                   # 模型保存目录
├── logs/                     # 结构化日志输出目录
├── db/
│   ├── models.py             # SQLAlchemy 数据模型
│   └── database.py           # 数据库连接与初始化
├── tests/
│   ├── test_train.py         # GAE 与隐状态单元测试
│   └── test_env.py           # 环境接口测试（可选）
└── requirements.txt
```

## 3. 模块接口与约束

### 3.1 env.py
- 必须定义一个基类 `EnvWrapper` 或直接使用 Gymnasium 接口。
- 环境必须实现 `reset()` 和 `step(action)` 方法。
  - `reset()` 返回 `(observation, info)`，其中 `observation` 为 numpy 数组或字典。
  - `step(action)` 返回 `(observation, reward, terminated, truncated, info)`，保持 Gym 0.26+ 标准 5 元组。
- 为实现静默代理干扰测试，环境内部不得维护全局计数器；所有时间步计数必须封装在环境实例内部。
- 提供一个 `CartPole` 的变体 `CartPoleSilentDisturbance`，该环境在正常 `step` 调用时，有 10% 概率内部多执行一次物理更新但不返回额外奖励，用于验证 GAE 计算的鲁棒性。

### 3.2 policy.py
- 定义 `Policy` 类，必须支持 LSTM 隐状态。
- 前向传播方法签名：`forward(obs, state) -> (action, log_prob, next_state)`
  - `obs`：当前观测，形状 `(batch_size, obs_dim)`
  - `state`：LSTM 隐状态，格式为 `(h, c)` 元组，`h` 和 `c` 形状 `(num_layers, batch_size, hidden_dim)`
  - 返回值 `action`：采样动作，`log_prob`：动作对数概率，`next_state`：更新后的隐状态。
- 策略必须提供 `act(obs, state)` 方法用于训练时交互，内部调用 `forward` 并进行动作采样。
- 网络结构可通过 YAML 配置 `hidden_sizes`、`activation`、`lstm_hidden_size`、`lstm_num_layers` 等参数。
- **隐状态传递规则**：在整个 rollout 过程中，`state` 必须持续传递，`reset()` 环境时绝不可将 `state` 重置为零状态，除非 episode 真正结束（`terminated` 或 `truncated`）。

### 3.3 train.py
- 实现 `Trainer` 类，负责以下流程：
  1. 根据 YAML 配置初始化环境、策略、优化器。
  2. 运行主循环（`num_episodes` 次 episode）。
  3. 每个 episode 内，调用 `rollout()` 收集一条轨迹，包括 `obs`、`actions`、`rewards`、`log_probs`、`states`、`terminals`。
  4. 计算 GAE（广义优势估计）与回报。
  5. 执行策略更新（PPO 等算法），计算策略损失与值损失。
  6. 记录 episode 总回报和损失到结构化日志。
- **GAE 计算规范**：
  - 输入：`rewards`（原始奖励序列，长度 T）、`values`（值函数预测，长度 T+1）、`gamma`、`lam`（GAE lambda）。
  - `delta_t = reward_t + gamma * value_{t+1} - value_t`
  - **严禁对 `reward_t` 进行任何缩放或变换后再计算 `delta_t`**。奖励缩放只能在计算损失时通过超参数（如 `reward_scale`）处理，且必须与 GAE 解耦。
  - GAE 计算需支持 vectorized 实现和循环实现两种模式，单元测试必须验证两者一致性。
- **静默代理干扰处理**：当环境内部存在干扰（如 `CartPoleSilentDisturbance` 中额外物理更新）时，`rollout` 收集的 `rewards` 序列必须与无干扰时逻辑一致，GAE 计算结果不得受影响。验证方式：开启/关闭干扰后，同一随机种子下 GAE 输出误差 < 1e-8。
- 训练过程中每 `log_interval` 个 episode 打印当前平均回报和损失，并将指标写入日志文件（JSON lines 格式，每行一个 event）。

### 3.4 结构化日志规范
- 采用 Python `logging` 模块，自定义 Formatter 输出 JSON。
- 每条日志至少包含：`timestamp`、`episode`、`total_reward`、`policy_loss`、`value_loss`、`entropy`。
- 日志文件命名：`logs/experiment_{experiment_id}.jsonl`。

### 3.5 CLI 接口（cli.py）
- 使用 Click 实现命令组：
  - `train`：启动训练，参数 `--config` 指定 YAML 路径。
  - `info`：查看已保存的实验列表（预留）。
- 启动训练时，CLI 需解析配置，将超参插入数据库并返回 `experiment_id`，然后将该 ID 传递给 `Trainer`，并在日志和模型保存路径中使用该 ID。

## 4. YAML 配置规范
示例配置 `configs/example.yaml`：
```yaml
experiment:
  name: "cartpole_ppo_test"
  env_id: "CartPoleSilentDisturbance-v0"
  seed: 42

algorithm:
  type: "ppo"
  gamma: 0.99
  lam: 0.95
  lr: 0.0003
  clip_ratio: 0.2
  value_loss_coef: 0.5
  entropy_coef: 0.01
  max_grad_norm: 0.5
  num_epochs: 4

training:
  num_episodes: 1000
  max_steps_per_episode: 500
  mini_batch_size: 64
  log_interval: 10
  save_interval: 100

network:
  hidden_sizes: [64, 64]
  activation: "tanh"
  lstm_hidden_size: 128
  lstm_num_layers: 1
  use_lstm: true

storage:
  model_dir: "models"
  log_dir: "logs"
  db_connection: "postgresql://user:pass@localhost:5432/rl_experiments"
```

## 5. 数据库设计
### 5.1 实验表 `experiments`
| 字段名           | 类型           | 说明                     |
|------------------|----------------|--------------------------|
| id               | SERIAL PRIMARY KEY | 实验唯一 ID              |
| name             | VARCHAR(255)   | 实验名称                 |
| config_yaml      | TEXT           | 完整的 YAML 配置原文     |
| status           | VARCHAR(50)    | 状态：running / finished / failed |
| total_episodes   | INT            | 总 episode 数            |
| final_reward     | FLOAT          | 最终总回报（最后 episode） |
| created_at       | TIMESTAMP      | 创建时间                 |
| updated_at       | TIMESTAMP      | 最后更新时间             |

### 5.2 模型表（预留，20% 阶段可不实现）
| 字段名           | 类型           | 说明                     |
|------------------|----------------|--------------------------|
| id               | SERIAL PRIMARY KEY | 模型 ID                  |
| experiment_id    | INT FK         | 关联实验                 |
| episode          | INT            | 保存时的 episode 数      |
| file_path        | VARCHAR(500)   | 模型文件路径             |
| metrics          | JSONB          | 保存时的指标快照         |

## 6. 20% 里程碑交付范围
### 6.1 必须完成的功能
1. **CLI 工具**：能通过 `python cli.py train --config config.yaml` 启动训练。
2. **环境实现**：`env.py` 中完成 `CartPoleSilentDisturbance`，继承 Gymnasium 接口，包含静默干扰逻辑。
3. **策略网络**：`policy.py` 中实现带 LSTM 的策略网络，遵守前向签名和隐状态传递规则。
4. **训练核心**：`train.py` 中实现完整的 rollout + GAE + 策略更新循环。
5. **GAE 正确性**：严格区分原始奖励与缩放奖励，静默干扰下 GAE 结果不变。
6. **隐状态连续传递**：episode 重置时隐状态不清零，单元测试验证边界。
7. **数据库集成**：启动训练时将配置写入 `experiments` 表，结束后更新 `final_reward`。
8. **结构化日志**：每 `log_interval` 输出 JSON 日志至对应文件。
9. **单元测试**：`tests/test_train.py` 需覆盖：
    - GAE 计算值 vs 手算值误差 < 1e-6
    - 静默干扰环境 vs 常规环境在相同种子下 GAE 输出一致
    - LSTM 隐状态在 episode 边界未被重置

### 6.2 明确不实现的内容
- 前端监控面板（仅定义 API 数据格式，不开发 UI）。
- 模型仓库的云存储集成。
- 多算法并行对比、断点续跑、超参数搜索。
- 用户权限与多租户。

## 7. 测试与验收标准
### 7.1 单元测试要求
- 使用 `pytest` 框架，测试文件位于 `tests/` 目录。
- 测试数据使用固定随机种子以保证可复现。
- 每个测试必须独立，不依赖外部数据库或文件系统，使用 mock 或临时数据库。

### 7.2 验收检查点
- [ ] `python cli.py train --config configs/example.yaml` 成功完成至少 10 个 episode，无异常退出。
- [ ] `tests/test_train.py` 中所有用例通过。
- [ ] 查看 `logs/experiment_1.jsonl`，可找到结构化的 episode 指标记录。
- [ ] 连接 PostgreSQL 查询 `experiments` 表，存在对应实验记录且 `final_reward` 已更新。
- [ ] 修改 YAML 中的 `lr` 或 `gamma` 后重新运行，训练结果和日志反映参数变化。
- [ ] 手动构造 episode 边界场景，单步调试确认 LSTM 隐状态在环境 `reset` 后未被置零。
- [ ] 通过打印或断点验证在 `CartPoleSilentDisturbance` 中干扰发生时，`rewards` 序列与无干扰时一致，且 GAE 输出相同。

## 8. 附加约束与注意事项
- 所有模块间的接口必须严格遵守本文档定义的签名与数据格式，不得引入额外的全局变量破坏模块独立性。
- 不允许在 `train.py` 中直接调用 `policy.py` 未导出的私有方法。
- 日志系统必须线程安全，未来可能扩展为异步写入，但当前版本无此要求。
- 代码需遵循 PEP 8 规范，关键算法处添加注释说明。
