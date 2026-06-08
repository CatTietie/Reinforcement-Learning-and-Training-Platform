# 项目架构与核心流程

## 目录结构

```
Reinforcement-Learning-and-Training-Platform/
├── cli.py                   # CLI入口（Click框架，6个命令）
├── train.py                 # 训练引擎：Trainer、GAE计算、结构化日志
├── policy.py                # Actor-Critic策略网络（MLP + LSTM）
├── env.py                   # Gymnasium环境定义与工厂函数
├── conftest.py              # Pytest路径配置
├── requirements.txt         # Python依赖
├── README.md                # 项目说明
├── Dockerfile.benchmark     # 基准测试Docker镜像
├── docker-compose.benchmark.yml  # 基准测试Docker Compose栈
├── .gitignore               # Git忽略规则
├── configs/
│   ├── example.yaml         # YAML训练配置示例
│   ├── distributed.yaml     # 分布式训练配置
│   ├── tune.yaml            # 超参调优配置
│   └── benchmark/
│       ├── cartpole_standard.yaml      # 标准CartPole基准
│       ├── cartpole_disturbance.yaml   # 带干扰CartPole基准
│       └── cartpole_bad_clip.yaml      # 预期失败基准（验证检测能力）
├── benchmarks/
│   └── cartpole_suite.yaml  # 基准测试套件定义
├── db/
│   ├── __init__.py          # 包标记
│   ├── models.py            # SQLAlchemy ORM模型（4张表）
│   └── database.py          # 数据库管理器（CRUD操作）
├── distributed/
│   ├── __init__.py          # 包标记
│   ├── coordinator.py       # 分布式训练协调器
│   ├── worker.py            # 数据采集Worker进程
│   ├── learner.py           # 集中式Learner（PPO更新）
│   └── utils.py             # 共享工具函数
├── tuning/
│   ├── __init__.py          # 包标记
│   ├── optuna_tuner.py      # Optuna超参调优器
│   └── search_space.py      # 搜索空间定义
├── monitor/
│   ├── __init__.py          # 包标记
│   ├── app.py               # FastAPI实时监控服务
│   ├── callback.py          # 训练监控回调
│   └── static/
│       └── index.html       # Vue 3 + Chart.js监控面板
├── benchmark/
│   ├── __init__.py          # 包标记
│   ├── runner.py            # 基准测试运行器
│   ├── schema.py            # 套件配置数据类与解析
│   ├── threshold.py         # 阈值判定逻辑
│   └── reporter.py          # Markdown报告与图表生成
├── tests/
│   ├── test_train.py        # 训练引擎测试
│   ├── test_distributed.py  # 分布式训练测试
│   ├── test_tuning.py       # 超参调优测试
│   ├── test_monitor.py      # 实时监控测试
│   └── test_benchmark.py    # 基准测试套件测试
├── .github/workflows/
│   └── benchmark.yml        # CI：基准回归检测
├── models/                  # PyTorch模型检查点（.pt文件）
├── logs/                    # 结构化JSONL训练日志
├── reports/                 # 基准测试报告与图表
└── doc/
    ├── architecture.md      # 本文档
    └── rl_platform_plan.md  # 需求与设计文档
```

---

## 模块职责

### cli.py — 命令行接口

| 命令 | 用途 |
|------|------|
| `python cli.py train --config <path> [--no-db] [--monitor-url <url>]` | 单进程PPO训练，可选监控上报 |
| `python cli.py train-distributed --config <path> [--workers N] [--no-db]` | 分布式异步PPO训练 |
| `python cli.py tune --config <path> [--n-trials N] [--distributed] [--workers N] [--no-db]` | Optuna超参自动调优 |
| `python cli.py info --db-connection <conn>` | 列出数据库中所有实验记录 |
| `python cli.py monitor [--host <host>] [--port <port>]` | 启动实时训练监控面板 |
| `python cli.py benchmark --suite <path> [--verbose] [--db <conn>]` | 运行基准测试套件 |

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

### distributed/ — 分布式异步PPO训练

基于参数服务器（Parameter Server）模式，使用共享内存实现零拷贝权重同步：

| 模块 | 职责 |
|------|------|
| `coordinator.py` | `DistributedTrainer`：编排spawn上下文、共享内存张量、Worker与Learner |
| `worker.py` | `worker_process()`：采集episode数据，检查权重版本号，通过Queue提交轨迹 |
| `learner.py` | `Learner`：消费轨迹队列，计算GAE，执行PPO更新，广播权重到共享内存 |
| `utils.py` | 共享工具：`create_policy_from_config`、`create_env_from_config`、`collect_episode`、`ppo_update_from_batch` |

**架构示意：**

```
┌────────────────────────────────────────────────────────┐
│               Shared Memory (权重 + 版本号)              │
└───────┬───────────────────────────────────┬────────────┘
        │ 读取权重                           │ 写入权重
        ▼                                   │
┌───────────────┐                   ┌───────┴───────┐
│  Worker 1..N  │ ──── Queue ────▶  │    Learner    │
│  (采集episode) │    (轨迹批次)      │  (PPO更新)    │
└───────────────┘                   └───────────────┘
```

### tuning/ — 超参数自动调优

基于Optuna框架实现贝叶斯超参数优化：

| 模块 | 职责 |
|------|------|
| `search_space.py` | 定义默认搜索空间（lr, clip_epsilon, lstm_hidden_size等），支持float/int/categorical类型 |
| `optuna_tuner.py` | `PPOTuner`：基线评估 → Optuna Study优化 → 最优参数全量训练；支持分布式模式与DB持久化 |

**默认搜索空间：**

| 超参数 | 类型 | 范围 |
|--------|------|------|
| lr | log-float | [5e-4, 5e-3] |
| clip_epsilon | float | [0.1, 0.3] |
| lstm_hidden_size | categorical | [64, 128, 256] |

### monitor/ — 实时训练监控

基于FastAPI + WebSocket + Vue 3的实时监控系统：

| 模块 | 职责 |
|------|------|
| `app.py` | FastAPI服务：REST API + WebSocket推送 + 内存状态缓存（deque, maxlen=1000） |
| `callback.py` | `MonitorCallback`：训练回调，通过HTTP POST上报指标，传递停止信号 |
| `static/index.html` | Vue 3 + Chart.js单页面板：实验选择、奖励曲线、损失曲线、紧急停止按钮 |

**API端点：**

| 端点 | 方法 | 用途 |
|------|------|------|
| `/api/experiments` | GET | 列出所有活跃实验 |
| `/api/experiments/{id}/metrics` | GET | 获取实验指标历史 |
| `/api/experiments/{id}/metrics` | POST | 上报新指标 |
| `/api/experiments/{id}/hyperparams` | GET | 获取超参数 |
| `/api/experiments/{id}/stop` | POST | 发送停止信号 |
| `/api/experiments/{id}/stop` | GET | 查询停止状态 |
| `/ws/{experiment_id}` | WebSocket | 实时指标推送 |

### benchmark/ — 基准测试与回归检测

自动化性能回归检测系统，集成CI流水线：

| 模块 | 职责 |
|------|------|
| `schema.py` | 数据类定义（`ThresholdConfig`、`BenchmarkConfig`、`SuiteSettings`、`BenchmarkSuite`）；YAML解析与校验 |
| `threshold.py` | 阈值判定：`check_threshold()`（比率制通过/失败）、`compute_effective_pass()`（预期失败反转） |
| `runner.py` | `BenchmarkRunner`：遍历套件基准，实例化Trainer运行训练，收集结果 |
| `reporter.py` | `BenchmarkReporter`：生成Markdown报告 + matplotlib柱状图 + DB历史趋势 |

### db/ — 数据持久化

**数据模型：**

| 表 | 模型 | 说明 |
|----|------|------|
| `experiments` | `Experiment` | 实验记录 |
| `trials` | `Trial` | 超参调优Trial记录 |
| `benchmark_runs` | `BenchmarkRun` | 基准套件运行记录 |
| `benchmark_results` | `BenchmarkResultRecord` | 单项基准结果 |

**experiments表结构：**

| 字段 | 类型 | 说明 |
|------|------|------|
| id | Integer, PK | 实验ID |
| name | String(255) | 实验名称 |
| config_yaml | Text | 完整YAML配置 |
| status | String(50) | running / finished / interrupted / failed |
| total_episodes | Integer | 完成的episode数 |
| final_reward | Float | 最终奖励 |
| training_mode | String(50) | 训练模式（single/distributed/tuning） |
| worker_count | Integer | Worker数量（分布式模式） |
| trial_id | Integer, FK | 关联的Trial ID |
| created_at | DateTime | 创建时间(UTC) |
| updated_at | DateTime | 更新时间(UTC) |

**trials表结构：**

| 字段 | 类型 | 说明 |
|------|------|------|
| id | Integer, PK | Trial ID |
| study_name | String(255) | Optuna Study名称 |
| trial_number | Integer | Trial序号 |
| hyperparams_json | Text | 超参数JSON |
| objective_value | Float | 目标值（最终奖励） |
| status | String(50) | complete / pruned / failed |
| experiment_id | Integer, FK | 关联的实验ID |
| created_at | DateTime | 创建时间(UTC) |

**benchmark_runs表结构：**

| 字段 | 类型 | 说明 |
|------|------|------|
| id | Integer, PK | 运行ID |
| suite_name | String(255) | 套件名称 |
| run_at | DateTime | 运行时间(UTC) |
| overall_status | String(50) | passed / failed |
| passed_count | Integer | 通过数 |
| failed_count | Integer | 失败数 |

**benchmark_results表结构：**

| 字段 | 类型 | 说明 |
|------|------|------|
| id | Integer, PK | 结果ID |
| run_id | Integer, FK | 关联的运行ID |
| benchmark_name | String(255) | 基准名称 |
| baseline_reward | Float | 基线奖励 |
| actual_reward | Float | 实际奖励 |
| ratio | Float | 实际/基线比率 |
| threshold_ratio | Float | 阈值比率 |
| passed | Boolean | 是否通过 |

**Database类方法：**

| 方法 | 用途 |
|------|------|
| `create_experiment` / `update_experiment` / `list_experiments` | 实验CRUD |
| `create_trial` / `update_trial` / `get_best_trial` / `list_trials` | Trial CRUD |
| `create_benchmark_run` / `list_benchmark_runs` | 基准运行CRUD |

---

## 配置格式

### 训练配置（YAML）

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

### 分布式训练配置

```yaml
distributed:
  num_workers: 4
  queue_size: 16
  sync_interval: 10
```

### 超参调优配置

```yaml
tuning:
  n_trials: 20
  study_name: "ppo_cartpole"
  search_space:
    lr: { type: "log_float", low: 5.0e-4, high: 5.0e-3 }
    clip_epsilon: { type: "float", low: 0.1, high: 0.3 }
    lstm_hidden_size: { type: "categorical", choices: [64, 128, 256] }
```

### 基准测试套件配置

```yaml
suite:
  name: "CartPole Regression Suite"
  defaults:
    num_episodes: 50
    threshold_ratio: 0.8

benchmarks:
  - name: "cartpole_standard"
    config: "configs/benchmark/cartpole_standard.yaml"
    baseline_reward: 200.0
  - name: "cartpole_disturbance"
    config: "configs/benchmark/cartpole_disturbance.yaml"
    baseline_reward: 180.0
  - name: "cartpole_bad_clip"
    config: "configs/benchmark/cartpole_bad_clip.yaml"
    baseline_reward: 200.0
    expected_failure: true
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
         │  │    循环:                      │  │
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
         │  │ 4. 日志 & 保存 & 监控上报    │  │
         │  │    JSONL记录episode指标       │  │
         │  │    控制台输出(每N个episode)   │  │
         │  │    MonitorCallback上报(可选)  │  │
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

## 分布式训练流程

```
┌──────────────────────────────────────────────────────────────┐
│  python cli.py train-distributed --config configs/distributed.yaml  │
└────────────────────────────┬─────────────────────────────────┘
                             │
                             ▼
              ┌──────────────────────────────┐
              │  DistributedTrainer 初始化    │
              │  ├─ 创建共享内存(权重+版本)   │
              │  ├─ 创建轨迹Queue            │
              │  ├─ spawn Worker进程 × N     │
              │  └─ 启动Learner线程          │
              └──────────────┬───────────────┘
                             │
               ┌─────────────┼─────────────┐
               ▼             │             ▼
     ┌──────────────┐       │    ┌──────────────┐
     │  Worker 1..N │       │    │   Learner    │
     │              │       │    │              │
     │  循环:       │       │    │  循环:       │
     │  1.读取权重  │       │    │  1.从Queue取 │
     │  2.采集episode│──Queue──▶│  2.计算GAE   │
     │  3.提交轨迹  │       │    │  3.PPO更新   │
     │  4.检查版本号│◀──共享内存──│  4.写入权重  │
     └──────────────┘            └──────────────┘
```

---

## 超参调优流程

```
┌────────────────────────────────────────────────────┐
│  python cli.py tune --config configs/tune.yaml     │
└──────────────────────────┬─────────────────────────┘
                           │
                           ▼
              ┌────────────────────────┐
              │  PPOTuner 初始化        │
              │  ├─ 加载搜索空间       │
              │  └─ 创建Optuna Study   │
              └───────────┬────────────┘
                          │
                          ▼
              ┌────────────────────────┐
              │  基线评估（默认超参）   │
              └───────────┬────────────┘
                          │
                          ▼
         ┌────────────────────────────────────┐
         │  Optuna优化循环 (trial 1..N)       │
         │                                    │
         │  1. 采样超参数                     │
         │  2. 构建配置 → 训练              │
         │  3. 返回目标值(final_reward)       │
         │  4. 持久化Trial到DB(可选)         │
         └────────────────┬───────────────────┘
                          │
                          ▼
              ┌────────────────────────┐
              │  最优超参全量训练       │
              │  输出最终结果           │
              └────────────────────────┘
```

---

## 基准测试与CI流程

```
┌─────────────────────────────────────────────────────────────┐
│  python cli.py benchmark --suite benchmarks/cartpole_suite.yaml  │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
              ┌──────────────────────────────┐
              │  BenchmarkRunner             │
              │  遍历suite中每个benchmark:   │
              │  1. 加载配置                 │
              │  2. 实例化Trainer并训练      │
              │  3. 收集final_reward         │
              │  4. 阈值判定(pass/fail)     │
              └──────────────┬───────────────┘
                             │
                             ▼
              ┌──────────────────────────────┐
              │  BenchmarkReporter           │
              │  ├─ 生成Markdown报告         │
              │  ├─ 生成matplotlib柱状图     │
              │  └─ 查询DB历史趋势(可选)    │
              └──────────────┬───────────────┘
                             │
                             ▼
              ┌──────────────────────────────┐
              │  CI集成 (GitHub Actions)     │
              │  ├─ PR触发基准测试           │
              │  ├─ 上传报告Artifact         │
              │  ├─ PR评论中展示结果         │
              │  └─ 回归则失败流水线         │
              └──────────────────────────────┘
```

---

## 模块依赖关系

```
cli.py
  ├── db.database.Database         (实验持久化)
  ├── train.Trainer                (单进程训练引擎)
  │     ├── env.make_env           (环境创建)
  │     ├── policy.Policy          (策略网络)
  │     ├── StructuredLogger       (JSONL日志)
  │     └── monitor.callback.MonitorCallback (可选，实时上报)
  ├── distributed.coordinator.DistributedTrainer (分布式训练)
  │     ├── distributed.worker     (数据采集)
  │     ├── distributed.learner    (集中更新)
  │     └── distributed.utils      (共享工具)
  ├── tuning.optuna_tuner.PPOTuner (超参调优)
  │     └── tuning.search_space    (搜索空间)
  ├── monitor.app                  (FastAPI监控服务)
  ├── benchmark.runner.BenchmarkRunner  (基准运行器)
  │     ├── benchmark.schema       (配置解析)
  │     ├── benchmark.threshold    (阈值判定)
  │     └── benchmark.reporter     (报告生成)
  └── YAML config                  (配置文件)
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
| 参数服务器模式 | 分布式训练使用共享内存零拷贝同步，避免网络开销 |
| Windows spawn兼容 | Worker使用顶层函数（非lambda/闭包）以兼容spawn上下文 |
| 贝叶斯调优 | Optuna TPE采样器，比网格搜索效率更高 |
| 监控解耦 | MonitorCallback静默吞噬网络错误，训练不受监控故障影响 |
| 预期失败基准 | 验证检测系统能正确捕获性能劣化 |
| CI回归门禁 | 基准测试失败阻止PR合并 |

---

## 技术栈与依赖

| 组件 | 技术 |
|------|------|
| 深度学习 | PyTorch |
| 强化学习环境 | Gymnasium |
| CLI框架 | Click |
| 数据库ORM | SQLAlchemy |
| 超参调优 | Optuna ≥ 3.0 |
| 监控后端 | FastAPI ≥ 0.100 + Uvicorn |
| 监控前端 | Vue 3 + Chart.js |
| HTTP客户端 | httpx ≥ 0.24 |
| 图表生成 | matplotlib |
| CI/CD | GitHub Actions |
| 容器化 | Docker + Docker Compose |
| 测试 | pytest + pytest-anyio |

---

## 测试覆盖

| 测试文件 | 验证内容 |
|----------|----------|
| `test_train.py` | GAE数学正确性、loop与vectorized一致性、手算值对比、LSTM隐状态连续性、静默干扰不变性、策略网络形状、端到端训练、DB CRUD |
| `test_distributed.py` | 策略创建、环境实例化、episode采集、PPO更新、共享内存同步、Coordinator Mock、Worker进程、Learner单元、调度集成、端到端分布式 |
| `test_tuning.py` | 搜索空间采样、PPOTuner目标函数、完整调优运行、DB集成、Trial CRUD |
| `test_monitor.py` | WebSocket消息格式、断开处理、停止命令传递、回调隔离、多实验隔离、WebSocket端点、完整停止链集成 |
| `test_benchmark.py` | 阈值判定、报告生成、运行器集成、套件YAML解析、BenchmarkRun持久化与历史 |
