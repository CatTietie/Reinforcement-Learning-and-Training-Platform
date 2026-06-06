"""
CLI 入口：使用 Click 实现命令组，支持 train、train-distributed、tune 和 info 命令。
"""

import sys

import click
import yaml

from db.database import Database
from train import Trainer


@click.group()
def cli():
    """强化学习训练平台 CLI 工具。"""
    pass


@cli.command()
@click.option(
    "--config",
    "-c",
    required=True,
    type=click.Path(exists=True),
    help="YAML 配置文件路径",
)
@click.option(
    "--no-db",
    is_flag=True,
    default=False,
    help="跳过数据库集成（本地测试模式）",
)
def train(config, no_db):
    """启动训练：解析配置、写入数据库、执行训练循环。"""
    with open(config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    with open(config, "r", encoding="utf-8") as f:
        config_raw = f.read()

    # 打印配置摘要确认参数透传
    algo = cfg.get("algorithm", {})
    net = cfg.get("network", cfg.get("policy", {}))
    click.echo(
        f"Config loaded: lr={algo.get('lr')}, gamma={algo.get('gamma')}, "
        f"gae_lambda={algo.get('gae_lambda')}, "
        f"hidden_sizes={net.get('hidden_sizes', net.get('hidden_size'))}, "
        f"lstm_hidden={net.get('lstm_hidden_size')}"
    )

    experiment_name = cfg.get("experiment", {}).get("name", "unnamed_experiment")
    experiment_id = None

    db = None
    if not no_db:
        db_connection = cfg.get("storage", {}).get("db_connection")
        if db_connection:
            try:
                db = Database(db_connection)
                experiment_id = db.create_experiment(
                    name=experiment_name,
                    config_yaml=config_raw,
                )
                click.echo(f"Experiment created in DB with ID: {experiment_id}")
            except Exception as e:
                click.echo(
                    f"Warning: Database unavailable ({e}). Continuing without DB.",
                    err=True,
                )
                db = None

    if experiment_id is None:
        experiment_id = "local"

    try:
        trainer = Trainer(config=cfg, experiment_id=experiment_id)
        result = trainer.train()

        if db and experiment_id != "local":
            db.update_experiment(
                experiment_id=experiment_id,
                status="finished",
                total_episodes=result["total_episodes"],
                final_reward=result["final_reward"],
            )
            click.echo(f"Experiment {experiment_id} marked as finished in DB.")

        click.echo(
            f"Training complete. Final reward: {result['final_reward']:.2f}, "
            f"Avg last 10: {result['avg_final_10']:.2f}"
        )

    except KeyboardInterrupt:
        click.echo("\nTraining interrupted by user.")
        if db and experiment_id != "local":
            db.update_experiment(experiment_id=experiment_id, status="interrupted")
        sys.exit(1)

    except Exception as e:
        click.echo(f"Training failed: {e}", err=True)
        if db and experiment_id != "local":
            db.update_experiment(experiment_id=experiment_id, status="failed")
        raise


@cli.command(name="train-distributed")
@click.option(
    "--config",
    "-c",
    required=True,
    type=click.Path(exists=True),
    help="YAML 配置文件路径",
)
@click.option(
    "--workers",
    "-w",
    default=4,
    type=int,
    help="Worker进程数",
)
@click.option(
    "--no-db",
    is_flag=True,
    default=False,
    help="跳过数据库集成",
)
def train_distributed(config, workers, no_db):
    """启动分布式异步PPO训练。"""
    from distributed.coordinator import DistributedTrainer

    with open(config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    with open(config, "r", encoding="utf-8") as f:
        config_raw = f.read()

    # 注入worker数量到配置
    cfg.setdefault("distributed", {})["num_workers"] = workers

    algo = cfg.get("algorithm", {})
    click.echo(
        f"Distributed training: workers={workers}, "
        f"lr={algo.get('lr')}, gamma={algo.get('gamma')}, "
        f"episodes={cfg.get('training', {}).get('num_episodes', 100)}"
    )

    experiment_name = cfg.get("experiment", {}).get("name", "unnamed_experiment")
    experiment_id = None

    db = None
    if not no_db:
        db_connection = cfg.get("storage", {}).get("db_connection")
        if db_connection:
            try:
                db = Database(db_connection)
                experiment_id = db.create_experiment(
                    name=experiment_name,
                    config_yaml=config_raw,
                    worker_count=workers,
                    training_mode="distributed",
                )
                click.echo(f"Experiment created in DB with ID: {experiment_id}")
            except Exception as e:
                click.echo(
                    f"Warning: Database unavailable ({e}). Continuing without DB.",
                    err=True,
                )
                db = None

    if experiment_id is None:
        experiment_id = "distributed"

    try:
        trainer = DistributedTrainer(config=cfg, experiment_id=experiment_id)
        result = trainer.train()

        if db and experiment_id != "distributed":
            db.update_experiment(
                experiment_id=experiment_id,
                status="finished",
                total_episodes=result["total_episodes"],
                final_reward=result["final_reward"],
            )
            click.echo(f"Experiment {experiment_id} marked as finished in DB.")

        click.echo(
            f"Distributed training complete. Final reward: {result['final_reward']:.2f}, "
            f"Avg last 10: {result['avg_final_10']:.2f}"
        )

    except KeyboardInterrupt:
        click.echo("\nTraining interrupted by user.")
        if db and experiment_id != "distributed":
            db.update_experiment(experiment_id=experiment_id, status="interrupted")
        sys.exit(1)

    except Exception as e:
        click.echo(f"Distributed training failed: {e}", err=True)
        if db and experiment_id != "distributed":
            db.update_experiment(experiment_id=experiment_id, status="failed")
        raise


@cli.command()
@click.option(
    "--config",
    "-c",
    required=True,
    type=click.Path(exists=True),
    help="基础YAML配置文件路径",
)
@click.option(
    "--n-trials",
    default=20,
    type=int,
    help="Optuna搜索trial数",
)
@click.option(
    "--distributed/--no-distributed",
    default=False,
    help="每个trial是否使用分布式训练",
)
@click.option(
    "--workers",
    "-w",
    default=1,
    type=int,
    help="分布式模式下的Worker数",
)
@click.option(
    "--no-db",
    is_flag=True,
    default=False,
    help="跳过数据库集成",
)
def tune(config, n_trials, distributed, workers, no_db):
    """Optuna超参数自动调优：搜索最佳超参组合并完成最终训练。"""
    from tuning.optuna_tuner import PPOTuner

    with open(config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    click.echo(
        f"Starting hyperparameter tuning: {n_trials} trials, "
        f"distributed={distributed}, workers={workers}"
    )

    db_connection = None
    if not no_db:
        db_connection = cfg.get("storage", {}).get("db_connection")

    tuner = PPOTuner(
        base_config=cfg,
        n_trials=n_trials,
        db_connection=db_connection,
        use_distributed=distributed,
        num_workers=workers,
    )

    try:
        result = tuner.run()

        click.echo("\n" + "=" * 60)
        click.echo("TUNING COMPLETE")
        click.echo("=" * 60)
        click.echo(f"Baseline avg reward: {result['baseline_reward']:.2f}")
        click.echo(f"Best hyperparameters: {result['best_params']}")
        click.echo(f"Best trial avg reward: {result['best_trial_value']:.2f}")
        click.echo(
            f"Final training (best params): reward={result['final_training_result']['final_reward']:.2f}, "
            f"avg_last_10={result['final_training_result']['avg_final_10']:.2f}"
        )
        click.echo(f"Improvement over baseline: {result['improvement_pct']:.1f}%")

    except KeyboardInterrupt:
        click.echo("\nTuning interrupted by user.")
        sys.exit(1)

    except Exception as e:
        click.echo(f"Tuning failed: {e}", err=True)
        raise


@cli.command()
@click.option(
    "--db-connection",
    "-d",
    default=None,
    help="数据库连接字符串",
)
def info(db_connection):
    """查看已保存的实验列表。"""
    if db_connection is None:
        click.echo("请通过 --db-connection 指定数据库连接字符串。")
        return

    try:
        db = Database(db_connection)
        experiments = db.list_experiments()
        if not experiments:
            click.echo("No experiments found.")
            return

        click.echo(
            f"{'ID':<5} {'Name':<30} {'Status':<12} {'Mode':<12} "
            f"{'Workers':<8} {'Episodes':<10} {'Reward':<10}"
        )
        click.echo("-" * 87)
        for exp in experiments:
            click.echo(
                f"{exp.id:<5} {exp.name:<30} {exp.status:<12} "
                f"{(exp.training_mode or 'single'):<12} "
                f"{(exp.worker_count or 1):<8} "
                f"{exp.total_episodes or '-':<10} "
                f"{exp.final_reward or '-':<10}"
            )
    except Exception as e:
        click.echo(f"Failed to connect to database: {e}", err=True)


if __name__ == "__main__":
    cli()
