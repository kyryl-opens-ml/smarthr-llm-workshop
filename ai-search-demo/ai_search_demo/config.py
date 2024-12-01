from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    colpali_base_url: str = "https://truskovskiyk--colpali-embedding-serve.modal.run"
    colpali_token: str = "super-secret-token"
    qdrant_uri: str = "https://qdrant.up.railway.app"
    qdrant_port: int = 443

    class Config:
        env_file = ".env"

settings = Settings()