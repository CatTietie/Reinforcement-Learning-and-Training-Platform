"""
SQLAlchemy 数据模型：实验元数据表。
"""

from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, Float, Integer, String, Text
from sqlalchemy.orm import declarative_base

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
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    def __repr__(self):
        return (
            f"<Experiment(id={self.id}, name='{self.name}', "
            f"status='{self.status}', final_reward={self.final_reward})>"
        )
