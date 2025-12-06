import asyncio
import logging
from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.firebase_client import init_firebase
from app.model_loader import load_model_into_memory
from app.services import process_active_sessions

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Main")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- STARTUP ---
    logger.info("Initializing System...")
    init_firebase()         # Connect DB
    load_model_into_memory() # Load Model ONCE
    
    # Start Background Task
    task = asyncio.create_task(background_poller())
    yield
    # --- SHUTDOWN ---
    task.cancel()
    logger.info("System Shutdown")

app = FastAPI(lifespan=lifespan)

async def background_poller():
    """
    Robust loop that won't die on error.
    """
    logger.info("Poller Started")
    while True:
        try:
            # Process logic
            process_active_sessions()
            
            # Wait for 30 seconds before checking again
            # We check often (30s) to catch the 5-min intervals quickly
            await asyncio.sleep(30) 
            
        except asyncio.CancelledError:
            logger.info("Poller stopping...")
            break
        except Exception as e:
            # CRITICAL: Catch generic errors so the loop doesn't die
            logger.error(f"Poller crashed (restarting in 10s): {e}")
            await asyncio.sleep(10)

@app.get("/health")
def health_check():
    return {"status": "running", "mode": "production"}