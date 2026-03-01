from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_PATH: Path = Path(__file__).parent.parent


class CustomBaseSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=f'{BASE_PATH}/.env',
        extra='allow',
        case_sensitive=False,
        env_prefix='',
        env_file_encoding='utf-8',
    )


class AppSettings(CustomBaseSettings):
    host: str = Field(validation_alias='APP_HOST', default='0.0.0.0')
    port: int = Field(validation_alias='APP_PORT', default=8000)
    reload: bool = Field(validation_alias='APP_RELOAD')


class LlmModelsSettings(CustomBaseSettings):
    text_model: str = Field(validation_alias='LLM_TEXT_MODEL')
    image_model: str = Field(validation_alias='LLM_IMAGE_MODEL')
    api_token: str = Field(validation_alias='LLM_API_TOKEN')
    base_url: str = Field(validation_alias='LLM_BASE_URL')
    max_retries: int = Field(validation_alias='LLM_MAX_RETRIES', ge=1, default=3)


class DeviceSettings(CustomBaseSettings):
    device: str = Field(validation_alias='DEVICE', default='cpu')


class EmbeddingModelSettings(CustomBaseSettings):
    model_name: str = Field(validation_alias='EMBEDDING_MODEL_NAME')


class ChromaSettings(CustomBaseSettings):
    collection_name: str = Field(validation_alias='CHROMA_COLLECTION_NAME')
    collection_description: str = Field(validation_alias='CHROMA_COLLECTION_DESCRIPTION')
    collection_hnsw_space: str = Field(validation_alias='CHROMA_HNSW_SPACE')


class Config(BaseSettings):
    app_config: AppSettings = AppSettings()
    llm_config: LlmModelsSettings = LlmModelsSettings()
    device_config: DeviceSettings = DeviceSettings()
    embedding_config: EmbeddingModelSettings = EmbeddingModelSettings()
    chroma_config: ChromaSettings = ChromaSettings()

    APP_PATH: Path = Path(__file__).parent


config = Config()
