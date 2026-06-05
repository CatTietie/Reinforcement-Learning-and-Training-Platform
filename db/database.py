"""
数据库连接与初始化：管理数据库连接及表创建。
支持 PostgreSQL（生产）和 SQLite（测试）。
"""

from datetime import datetime, timezone

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from db.models import Base, Experiment


class Database:
    """数据库管理器：负责连接、建表、实验记录的 CRUD。"""

    def __init__(self, connection_string):
        self.engine = create_engine(connection_string, echo=False)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)

    def create_experiment(self, name, config_yaml):
        """创建实验记录，返回实验 ID。"""
        session = self.Session()
        try:
            now = datetime.now(timezone.utc)
            experiment = Experiment(
                name=name,
                config_yaml=config_yaml,
                status="running",
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

    def update_experiment(self, experiment_id, status=None, total_episodes=None, final_reward=None):
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
