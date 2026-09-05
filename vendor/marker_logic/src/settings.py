from typing import Optional

from dotenv import find_dotenv
from pydantic import computed_field
from pydantic_settings import BaseSettings
import torch
import os


class Settings(BaseSettings):
    # Paths
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    OUTPUT_DIR: str = os.path.join(BASE_DIR, "conversion_results")
    FONT_DIR: str = os.path.join(BASE_DIR, "static", "fonts")
    DEBUG_DATA_FOLDER: str = os.path.join(BASE_DIR, "debug_data")
    ARTIFACT_URL: str = "https://models.datalab.to/artifacts"
    FONT_NAME: str = "GoNotoCurrent-Regular.ttf"
    FONT_PATH: str = os.path.join(FONT_DIR, FONT_NAME)
    LOGLEVEL: str = "INFO"

    # General
    OUTPUT_ENCODING: str = "utf-8"
    OUTPUT_IMAGE_FORMAT: str = "JPEG"

    # LLM
    GOOGLE_API_KEY: Optional[str] = ""

    # General models
    TORCH_DEVICE: Optional[str] = (
        None  # Device for the local torch models (ocr error, fast layout); the VLM runs on the inference server
    )

    @computed_field
    @property
    def TORCH_DEVICE_MODEL(self) -> str:
        if self.TORCH_DEVICE is not None:
            return self.TORCH_DEVICE

        if torch.cuda.is_available():
            return "cuda"

        if torch.backends.mps.is_available():
            return "mps"

        return "cpu"

    class Config:
        env_file = find_dotenv("local.env")
        extra = "ignore"


settings = Settings()
