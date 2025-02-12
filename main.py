from fastapi import FastAPI
from api.routes import router

app = FastAPI()

# Register API Routes
app.include_router(router)

# Run API Server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
