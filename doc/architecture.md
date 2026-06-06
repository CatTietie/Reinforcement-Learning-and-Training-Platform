# 项目架构与核心流程

## 目录结构

```
Reinforcement-Learning-and-Training-Platform/
├── cli.py                   # CLI入口（Click框架）
├── train.py                 # 训练引擎：Trainer、GAE计算、结构化日志
├── policy.py                # Actor-Critic策略网络（MLP + LSTM）
├── env.py                   # Gymnasium环境定义与工厂函数
├── conftest.py              # Pytest路径配置
├── requirements.txt         # Python依赖
├── README.md                # 项目说明
├── configs/
│   └── example.yaml         # YAML训练配置示例
├── db/
│   ├── __init__.py          # 包标记
│   ├── models.py            # SQLAlchemy ORM模型（experiments表）
│   └── database.py          # 数据库管理器（CRUD操作）
├── tests/
│   └── test_train.py        # 单元与集成测试
├── models/                  # PyTorch模型检查点（.pt文件）
├── logs/                    # 结构化JSONL训练日志
└── doc/
    └── rl_platform_plan.md  # 需求与设计文档
```

---

## 模块职责

### cli.py — 命令行接口

| 命令 | 用途 |
|------|------|
| `python cli.py train --config <path> [--no-db]` | 解析YAML配置，启动训练，记录实验到数据库 |
| `python cli.py info --db-connection <conn>` | 列出数据库中所有实验记录 |

`--no-db` 标志允许在无PostgreSQL的情况下进行本地训练。

### train.py — 训练引擎

包含三个核心组件：

- **GAE函数**：`compute_gae_loop` 和 `compute_gae_vectorized`，计算广义优势估计
- **StructuredLogger**：以JSON Lines格式写入训练日志（含UTC时间戳）
- **Trainer类**：训练主循环，协调环境交互、GAE计算、PPO更新和模型保存

### policy.py — 策略网络

`Policy(nn.Module)` 类实现 Actor-Critic 架构：

```
观测 → MLP特征提取器 [64, 64] → (可选)LSTM → Actor头(动作logits)
                                            → Critic头(状态价值)
```

关键方法：
- `act(obs, state)` — rollout时使用，返回动作、log_prob、隐状态、价值
- `evaluate(obs, state, actions)` — PPO更新时重新评估已采取动作
- `get_value(obs, state)` — 获取价值估计（用于GAE bootstrap）

### env.py — 环境层

- **CartPoleSilentDisturbance** — 带静默物理干扰的CartPole（不影响返回的观测/奖励）
- **CartPoleStandard** — 标准CartPole控制环境
- **make_env(name, **kwargs)** — 注册表工厂函数

### db/ — 数据持久化

**experiments表结构：**

| 字段 | 类型 | 说明 |
|------|------|------|
| id | Integer, PK | 实验ID |
| name | String(255) | 实验名称 |
| config_yaml | Text | 完整YAML配置 |
| status | String(50) | running / finished / interrupted / failed |
| total_episodes | Integer | 完成的episode数 |
| final_reward | Float | 最终奖励 |
| created_at | DateTime | 创建时间(UTC) |
| updated_at | DateTime | 更新时间(UTC) |

---

## 配置格式（YAML）

```yaml
experiment:
  name: "cartpole_ppo_lstm"
  seed: 42

env:
  name: "CartPoleSilentDisturbance-v0"
  max_steps: 500
  disturbance_prob: 0.1

network:
  hidden_sizes: [64, 64]
  activation: "tanh"
  lstm_hidden_size: 128
  use_lstm: true

algorithm:
  type: "ppo"
  lr: 3.0e-4
  gamma: 0.99
  gae_lambda: 0.95
  clip_epsilon: 0.2
  value_coef: 0.5
  entropy_coef: 0.01
  update_epochs: 4
  max_grad_norm: 0.5

training:
  num_episodes: 100
  batch_size: 64
  log_interval: 10
  save_interval: 100

storage:
  model_dir: "models"
  db_connection: "postgresql://..."

logging:
  level: "INFO"
  log_dir: "logs"
```

---

## 核心训练流程

```
┌─────────────────────────────────────────────────────────┐
│  python cli.py train --config configs/example.yaml      │
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼
              ┌────────────────────────┐
              │  解析YAML + 创建DB记录  │
              └───────────┬────────────┘
                          │
                          ▼
              ┌────────────────────────┐
              │  初始化Trainer         │
              │  ├─ 设置随机种子       │
              │  ├─ 创建环境(make_env) │
              │  ├─ 创建策略网络       │
              │  ├─ 创建Adam优化器     │
              │  └─ 初始化日志器       │
              └───────────┬────────────┘
                          │
                          ▼
         ┌────────────────────────────────────┐
         │  训练主循环 (episode 1..N)          │
         │                                    │
         │  ┌──────────────────────────────┐  │
         │  │ 1. Rollout                   │  │
         │  │    env.reset()               │  │
         │  │    循环:                      │
         │  │      policy.act() → 动作     │  │
         │  │      env.step() → 奖励/观测  │  │
         │  │    收集轨迹数据              │  │
         │  └──────────────┬───────────────┘  │
         │                 │                  │
         │                 ▼                  │
         │  ┌──────────────────────────────┐  │
         │  │ 2. GAE计算                   │  │
         │  │    δ_t = r + γV(t+1) - V(t)  │  │
         │  │    A_t = δ + γλ·A(t+1)       │  │
         │  │    returns = A + V            │  │
         │  └──────────────┬───────────────┘  │
         │                 │                  │
         │                 ▼                  │
         │  ┌──────────────────────────────┐  │
         │  │ 3. PPO更新                   │  │
         │  │    优势标准化                 │  │
         │  │    多轮mini-batch更新:       │  │
         │  │      ratio = exp(Δlog_prob)  │  │
         │  │      clip surrogate loss     │  │
         │  │      value loss (MSE)        │  │
         │  │      entropy bonus           │  │
         │  │      梯度裁剪 + 反向传播     │  │
         │  └──────────────┬───────────────┘  │
         │                 │                  │
         │                 ▼                  │
         │  ┌──────────────────────────────┐  │
         │  │ 4. 日志 & 保存               │  │
         │  │    JSONL记录episode指标       │  │
         │  │    控制台输出(每N个episode)   │  │
         │  │    模型检查点(奖励提升时保存) │  │
         │  └──────────────────────────────┘  │
         └────────────────────────────────────┘
                          │
                          ▼
              ┌────────────────────────┐
              │  更新DB: status=finished│
              │  输出最终奖励摘要       │
              └────────────────────────┘
```

---

## 模块依赖关系

```
cli.py
  ├── db.database.Database    (实验持久化)
  ├── train.Trainer           (训练引擎)
  │     ├── env.make_env      (环境创建)
  │     ├── policy.Policy     (策略网络)
  │     └── StructuredLogger  (JSONL日志)
  └── YAML config             (配置文件)
```

---

## 关键设计决策

| 决策 | 说明 |
|------|------|
| LSTM隐状态管理 | episode内连续传递，仅在真正终止时重置为零 |
| 数据库可选 | `--no-db`标志支持无DB本地开发 |
| 双重日志 | JSONL文件供机器解析 + 控制台输出供人阅读 |
| 检查点策略 | 奖励提升时保存 + 固定间隔保存 |
| 静默干扰环境 | 验证GAE在内部物理扰动下的鲁棒性 |
| 配置驱动 | 单一YAML文件控制所有超参和路径 |

---

## 测试覆盖

| 测试类 | 验证内容 |
|--------|----------|
| TestGAEComputation | GAE数学正确性、loop与vectorized一致性 |
| TestGAEHandComputed | 手算值对比，误差 < 1e-6 |
| TestHiddenStateContinuity | LSTM隐状态在episode边界正确处理 |
| TestSilentDisturbanceInvariance | 静默干扰不影响奖励序列和GAE输出 |
| TestPolicyNetwork | 网络前向传播输出形状验证 |
| TestTrainerIntegration | 端到端训练循环完成无报错 |
| TestDatabaseSQLite | 内存SQLite下的CRUD操作 |
