from pydantic import BaseModel, Field

class DetectionCreate(BaseModel):
    camera_id: str
    plate_text: str
    confidence: float = Field(ge=0, le=1)
