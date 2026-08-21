import uvicorn
from src.api.endpoint import app

def main():
    """Run FastAPI application using Uvicorn server."""
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()