from pydantic import BaseModel


class AddPayload(BaseModel):
    first_number: int
    second_number: int
