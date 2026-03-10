## Structured output
- Models can be requested to provide their response in a format matching a given schema. This is useful for ensuring the output can be easily parsed and used in subsequent processing. LangChain supports multiple schema types and methods for enforcing structured output.
Pydantic models provide the richest feature set with field validation, descriptions, and nested structures.
- main7_structured_output.py 
```python
...
from pydantic import BaseModel, Field

...

class Movie(BaseModel):
    """A movie with details."""
    title: str = Field(..., description="The title of the movie")
    year: int = Field(..., description="The year the movie was released")
    director: str = Field(..., description="The director of the movie")
    rating: float = Field(..., description="The movie's rating out of 10")

model_with_structure = model.with_structured_output(Movie)
response = model_with_structure.invoke("Provide details about the movie Terminator-2")
print(response)
```
- Run
```bash
uv --project uv_env/ run python week_05_langchain/01_core_components/02_models/main7_structured_output.py 
```
- Output
```
title='Terminator-2' year=1991 director='James Cameron' rating=8.3
```