from fastapi import FastAPI
from app.api.routes_health import router as health_router

app = FastAPI(tittle = 'Movie reccomender API', version = '1.0.0')
app.include_router(health_router)