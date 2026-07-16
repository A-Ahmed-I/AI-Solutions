from pydantic import BaseModel


class PredictionResponse(BaseModel):
    """Response schema for model prediction output."""

    label: str
    probability: float
