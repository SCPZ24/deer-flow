from pydantic import BaseModel, ConfigDict, Field


class ImageModelConfig(BaseModel):
    name: str = Field(
        default = None,
        description = "Unique identifier for the image model",
    )
    model: str = Field(
        default = None,
        description = "Name of the image model",
    )
    api_base: str = Field(
        default = None,
        description = "Base URL for the image model API",
    )
    api_key: str = Field(
        default = None,
        description = "API key for authenticating with the image model API",
    )