# RL Experiment Platform

强化学习实验训练与追踪平台，旨在以工程化方式管理 RL 算法开发全生命周期。平台严格遵循已有的 `env.py`、`policy.py`、`train.py` 接口规范，确保 GAE 计算、LSTM 隐状态传递等算法细节的正确性，避免因随意简化而引入逻辑错误。

## 核心功能
- **YAML 配置驱动**：通过统一配置文件定义环境、算法、网络结构与超参数。
- **CLI 工具**：基于 Click 的命令行入口，一键启动训练、断点续跑和日志回溯。
- **训练核心**：集成经典强化学习算法，内置 GAE 优势估计与 LSTM 隐状态管理，保证在多代理干扰下轨迹不变性。
- **实验追踪**：自动将超参与最终回报存入 PostgreSQL，便于实验对比。
- **结构化日志**：按时间戳输出每个 episode 的损失、回报，方便监控。

## 技术栈
- Python ≥ 3.9
- Click（CLI）、PyTorch（网络与训练）
- Gymnasium（环境接口）
- PostgreSQL（实验元数据）
- YAML 配置解析

## 快速开始
1. 克隆仓库并安装依赖：
   ```bash
   pip install -r requirements.txt
   ```
2. 准备配置文件 `config.yaml`（参考 `configs/example.yaml`）。
3. 启动训练：
   ```bash
   python cli.py train --config config.yaml
   ```
4. 运行单元测试验证核心算法：
   ```bash
   pytest tests/test_train.py
   ```

## 目录结构
```
├── cli.py                # 命令行入口
├── env.py                # 环境接口
├── policy.py             # 策略网络（含 LSTM）
├── train.py              # 训练循环与 GAE
├── configs/              # 配置样例
├── models/               # 模型保存目录
├── tests/                # 单元测试
└── logs/                 # 训练日志
```
