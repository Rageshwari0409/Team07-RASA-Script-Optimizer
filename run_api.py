"""Script to run the FastAPI application."""
import uvicorn
from src.utils.config_loader import get_config


def main():
    """Run the FastAPI application."""
    # Load configuration
    config = get_config()
    
    # Get server settings
    host="localhost"
    port = config.get('fastapi.port', 8000)
    reload = True 
    
    print(f"""
    ╔══════════════════════════════════════════════════════════════╗
    ║   Sales Transcript Analysis API                              ║
    ║                                                              ║
    ║   Server starting at: http://{host}:{port}              ║
    ║   API Documentation: http://{host}:{port}/docs          ║
    ║                                                              ║
    ║   Press CTRL+C to stop the server                           ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    # Run the server
    uvicorn.run(
        "src.api.main:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )


if __name__ == "__main__":
    main()

