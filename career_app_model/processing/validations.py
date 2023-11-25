from typing import List, Optional, Dict, Tuple, Union
from pydantic import BaseModel, ValidationError, validator, field_validator


def validate_inputs(*, input_data: Dict) -> Tuple[Dict, Optional[dict]]:
    """Check model inputs for wrong values."""

    validated_data = input_data.copy()

    errors = None
    try:
        SingleResponseDataInputSchema(**input_data)
    except ValidationError as error:
        errors = error.json()

    return validated_data, errors


class SingleResponseDataInputSchema(BaseModel):
    profileId: str
    selectedIndustries: List[str]
    selectedInterests: List[str]
    responses: List[Dict[str, Union[str, int]]]

