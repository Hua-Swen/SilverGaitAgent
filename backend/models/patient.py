from pydantic import BaseModel, Field
from typing import Optional
from datetime import date


class Patient(BaseModel):
    id: Optional[int] = None
    name: str
    date_of_birth: date
    gender: str  # "male" | "female" | "other"
    created_at: Optional[str] = None

    @property
    def age(self) -> int:
        today = date.today()
        return today.year - self.date_of_birth.year - (
            (today.month, today.day) < (self.date_of_birth.month, self.date_of_birth.day)
        )
