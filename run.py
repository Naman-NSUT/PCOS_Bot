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
        logger.error(".env file not found! Please create one based on .env.example with your OPENAI_API_KEY.")
        sys.exit(1)

    # 1b. Check the API key is actually set
    from dotenv import load_dotenv
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY is empty in .env. Add your key before starting.")
        sys.exit(1)

    python_exe = sys.executable
    if os.path.exists("venv/bin/python"):
        python_exe = "venv/bin/python"

    uvicorn_exe = "uvicorn"
    if os.path.exists("venv/bin/uvicorn"):
        uvicorn_exe = "venv/bin/uvicorn"

    # 2. Run Ingestion
    logger.info("Step 1: Running knowledge base ingestion...")
    try:
        # Run ingestion as a separate process
        subprocess.run([python_exe, "-m", "src.ingestion.run_ingest"], check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Ingestion failed with exit code {e.returncode}. Stopping.")
        sys.exit(1)

    # 3. Start API Server
    logger.info("Step 2: Starting FastAPI server...")
    try:
        subprocess.run([
            uvicorn_exe, "src.api.main:app", 
            "--host", "0.0.0.0", 
            "--port", "8003", 
            "--reload"
        ])
    except KeyboardInterrupt:
        logger.info("Stopping server...")
    except Exception as e:
        logger.error(f"Server crashed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run()
