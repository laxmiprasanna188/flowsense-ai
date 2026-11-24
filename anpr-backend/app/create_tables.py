
from app.database import engine, Base
from app.models import Detection

Base.metadata.create_all(bind=engine)

print("Database tables created successfully!")
