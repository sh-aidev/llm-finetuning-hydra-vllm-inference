import uvicorn
from omegaconf import DictConfig
from fastapi import FastAPI, APIRouter
from fastapi.middleware.cors import CORSMiddleware

from src.utils.logger import logger
from src.core.inference import LLMInference, LLMInferenceHF

v1Router = APIRouter()

def get_router(cfg: DictConfig) -> APIRouter:
    llm = LLMInferenceHF(cfg)
    @v1Router.post("/llm-infer", status_code=200)
    def generate(
        text: str
        ) -> dict:
        return {"output": llm.run(text)}

    @v1Router.get("/health")
    def health():
        return {"message": "ok"}
    return v1Router

class LLMServer:
    def __init__(self, cfg: DictConfig, router: APIRouter) -> None:
        self.cfg = cfg
        self.port = cfg.server.port
        self.host = cfg.server.host

        self.server = FastAPI()
        self.server.add_middleware(
            CORSMiddleware,
            allow_origins = ["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
            
        )
        self.server.include_router(router, prefix="/v1")
    
    def run(self):
        uvicorn.run(self.server, port=self.port, host=self.host)