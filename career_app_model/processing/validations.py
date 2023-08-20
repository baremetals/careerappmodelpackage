from typing import List, Optional, Tuple
import pandas as pd
from pydantic import BaseModel, ValidationError
import numpy as np


def validate_inputs(*, input_data: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[dict]]:
    """Check model inputs for unprocessable values."""

    # Assuming you've pre-processed and filtered the columns you need in your data
    validated_data = input_data.copy()

    errors = None
    try:
        # replace numpy nans so that pydantic can validate
        UserDataInputs(
            inputs=validated_data.replace({np.nan: None}).to_dict(orient="records")
        )
    except ValidationError as error:
        errors = error.json()

    return validated_data, errors


class SingleUserDataInputSchema(BaseModel):
    userId: str
    selectedIndustry1: str
    selectedIndustry2: str
    selectedIndustry3: str
    selectedIndustry4: str
    selectedIndustry5: str
    responseToQuestion1: str
    responseToQuestion2: str
    responseToQuestion3: str
    responseToQuestion4: str
    responseToQuestion5: str
    responseToQuestion6: str
    responseToQuestion7: str
    responseToQuestion8: str
    responseToQuestion9: str
    responseToQuestion10: str
    responseToQuestion11: str
    responseToQuestion12: str
    responseToQuestion13: str
    responseToQuestion14: str
    responseToQuestion15: str
    responseToQuestion16: str
    responseToQuestion17: str
    responseToQuestion18: str
    responseToQuestion19: str
    responseToQuestion20: str


class UserDataInputs(BaseModel):
    inputs: List[SingleUserDataInputSchema]