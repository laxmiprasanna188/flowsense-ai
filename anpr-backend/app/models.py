
from sqlalchemy import Column, Integer, String, Float, DateTime
from datetime import datetime, timezone

from app.database import Base


class Detection(Base):
    __tablename__ = "detections"

    id = Column(Integer, primary_key=True, index=True)

    event_id = Column(String, unique=True, index=True)

    camera_id = Column(String, nullable=False)

    plate_text = Column(String, nullable=False)

    confidence = Column(Float, nullable=False)

    timestamp = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc)
    )
