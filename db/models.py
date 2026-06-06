"""
SQLAlchemy 数据模型：实验元数据表 + 调优Trial表。
"""

import json
from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class Experiment(Base):
    """实验表：存储每次训练的配置与结果。"""

    __tablename__ = "experiments"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), nullable=False)
    config_yaml = Column(Text, nullable=False)
    status = Column(String(50), nullable=False, default="running")
    total_episodes = Column(Integer, nullable=True)
    final_reward = Column(Float, nullable=True)
    worker_count = Column(Integer, nullable=True, default=1)
    trial_id = Column(Integer, ForeignKey("trials.id"), nullable=True)
    training_mode = Column(String(50), nullable=True, default="single")
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    trial = relationship("Trial", back_populates="experiment", foreign_keys=[trial_id])

    def __repr__(self):
        return (
            f"<Experiment(id={self.id}, name='{self.name}', "
            f"status='{self.status}', final_reward={self.final_reward})>"
        )


class Trial(Base):
    """调优Trial表：存储每次Optuna试验的超参组合和结果。"""

    __tablename__ = "trials"

    id = Column(Integer, primary_key=True, autoincrement=True)
    study_name = Column(String(255), nullable=False)
    trial_number = Column(Integer, nullable=False)
    hyperparams_json = Column(Text, nullable=False)
    objective_value = Column(Float, nullable=True)
    status = Column(String(50), nullable=False, default="running")
    experiment_id = Column(Integer, ForeignKey("experiments.id"), nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    experiment = relationship("Experiment", foreign_keys=[experiment_id])

    @property
    def hyperparams(self):
        return json.loads(self.hyperparams_json) if self.hyperparams_json else {}

    def __repr__(self):
        return (
            f"<Trial(id={self.id}, study='{self.study_name}', "
            f"number={self.trial_number}, value={self.objective_value})>"
        )
