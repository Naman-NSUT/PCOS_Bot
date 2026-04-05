import os
import sys
import subprocess
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("PCOS-Runner")

def run():
    """Run ingestion followed by the FastAPI server."""
    
    # 1. Check for .env file
    if not os.path.exists(".env"):
        logger.error(".env file not found! Please create one based on .env.example with your GEMINI_API_KEY.")
        sys.exit(1)

    # 2. Run Ingestion
    logger.info("Step 1: Running knowledge base ingestion...")
    try:
        # Run ingestion as a separate process
        subprocess.run([sys.executable, "-m", "src.ingestion.run_ingest"], check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Ingestion failed with exit code {e.returncode}. Stopping.")
        sys.exit(1)

    # 3. Start API Server
    logger.info("Step 2: Starting FastAPI server...")
    try:
        # uvicorn.run is blocking, so we use subprocess to keep the runner simple 
        # or we could import and call it, but uvicorn's CLI handles reloads and signals better.
        subprocess.run([
            "uvicorn", "src.api.main:app", 
            "--host", "0.0.0.0", 
            "--port", "8001", 
            "--reload"
        ])
    except KeyboardInterrupt:
        logger.info("Stopping server...")
    except Exception as e:
        logger.error(f"Server crashed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run()
