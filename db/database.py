"""
数据库连接与初始化：管理数据库连接及表创建。
支持 PostgreSQL（生产）和 SQLite（测试）。
"""

import json
from datetime import datetime, timezone

from sqlalchemy import create_engine
from sqlalchemy.orm import joinedload, sessionmaker

from db.models import Base, BenchmarkResultRecord, BenchmarkRun, Experiment, Trial


class Database:
    """数据库管理器：负责连接、建表、实验和Trial记录的 CRUD。"""

    def __init__(self, connection_string):
        self.engine = create_engine(connection_string, echo=False)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)

    def create_experiment(self, name, config_yaml, worker_count=1,
                          trial_id=None, training_mode="single"):
        """创建实验记录，返回实验 ID。"""
        session = self.Session()
        try:
            now = datetime.now(timezone.utc)
            experiment = Experiment(
                name=name,
                config_yaml=config_yaml,
                status="running",
                worker_count=worker_count,
                trial_id=trial_id,
                training_mode=training_mode,
                created_at=now,
                updated_at=now,
            )
            session.add(experiment)
            session.commit()
            exp_id = experiment.id
            return exp_id
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def update_experiment(self, experiment_id, status=None, total_episodes=None,
                          final_reward=None):
        """更新实验记录，同步刷新 updated_at。"""
        session = self.Session()
        try:
            experiment = session.query(Experiment).filter_by(id=experiment_id).first()
            if experiment is None:
                raise ValueError(f"Experiment {experiment_id} not found")
            if status is not None:
                experiment.status = status
            if total_episodes is not None:
                experiment.total_episodes = total_episodes
            if final_reward is not None:
                experiment.final_reward = final_reward
            experiment.updated_at = datetime.now(timezone.utc)
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def get_experiment(self, experiment_id):
        """查询实验记录。"""
        session = self.Session()
        try:
            experiment = session.query(Experiment).filter_by(id=experiment_id).first()
            if experiment:
                session.expunge(experiment)
            return experiment
        finally:
            session.close()

    def list_experiments(self):
        """列出所有实验。"""
        session = self.Session()
        try:
            experiments = session.query(Experiment).all()
            for exp in experiments:
                session.expunge(exp)
            return experiments
        finally:
            session.close()

    def create_trial(self, study_name, trial_number, hyperparams_dict,
                     experiment_id=None):
        """创建Trial记录，返回trial ID。"""
        session = self.Session()
        try:
            trial = Trial(
                study_name=study_name,
                trial_number=trial_number,
                hyperparams_json=json.dumps(hyperparams_dict, ensure_ascii=False),
                status="running",
                experiment_id=experiment_id,
                created_at=datetime.now(timezone.utc),
            )
            session.add(trial)
            session.commit()
            trial_id = trial.id
            return trial_id
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def update_trial(self, trial_id, status=None, objective_value=None,
                     experiment_id=None):
        """更新Trial状态和目标值。"""
        session = self.Session()
        try:
            trial = session.query(Trial).filter_by(id=trial_id).first()
            if trial is None:
                raise ValueError(f"Trial {trial_id} not found")
            if status is not None:
                trial.status = status
            if objective_value is not None:
                trial.objective_value = objective_value
            if experiment_id is not None:
                trial.experiment_id = experiment_id
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def get_best_trial(self, study_name):
        """获取指定study中目标值最高的Trial。"""
        session = self.Session()
        try:
            trial = (
                session.query(Trial)
                .filter_by(study_name=study_name, status="completed")
                .order_by(Trial.objective_value.desc())
                .first()
            )
            if trial:
                session.expunge(trial)
            return trial
        finally:
            session.close()

    def list_trials(self, study_name=None):
        """列出Trial记录，可按study_name筛选。"""
        session = self.Session()
        try:
            query = session.query(Trial)
            if study_name:
                query = query.filter_by(study_name=study_name)
            trials = query.all()
            for t in trials:
                session.expunge(t)
            return trials
        finally:
            session.close()

    def create_benchmark_run(self, suite_name, overall_status, passed_count,
                             failed_count, result_records):
        """创建 BenchmarkRun 及其关联的结果记录，返回 run ID。

        result_records: list of dict with keys:
            benchmark_name, baseline_reward, actual_reward, ratio, threshold_ratio, passed
        """
        session = self.Session()
        try:
            run = BenchmarkRun(
                suite_name=suite_name,
                run_at=datetime.now(timezone.utc),
                overall_status=overall_status,
                passed_count=passed_count,
                failed_count=failed_count,
            )
            session.add(run)
            session.flush()

            for rec in result_records:
                record = BenchmarkResultRecord(
                    run_id=run.id,
                    benchmark_name=rec["benchmark_name"],
                    baseline_reward=rec["baseline_reward"],
                    actual_reward=rec["actual_reward"],
                    ratio=rec["ratio"],
                    threshold_ratio=rec["threshold_ratio"],
                    passed=int(rec["passed"]),
                )
                session.add(record)

            session.commit()
            run_id = run.id
            return run_id
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def list_benchmark_runs(self, suite_name, limit=3):
        """查询指定 suite 最近 N 次 BenchmarkRun（含关联结果），按时间倒序。"""
        session = self.Session()
        try:
            runs = (
                session.query(BenchmarkRun)
                .options(joinedload(BenchmarkRun.results))
                .filter_by(suite_name=suite_name)
                .order_by(BenchmarkRun.run_at.desc())
                .limit(limit)
                .all()
            )
            session.expunge_all()
            return runs
        finally:
            session.close()
