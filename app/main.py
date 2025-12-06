import asyncio
import logging
from fastapi import FastAPI
from contextlib import asynccontextmanager

# Local imports
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
    
    # Start Background Poller
    task = asyncio.create_task(background_poller())
    
    yield
    
    # --- SHUTDOWN ---
    logger.info("Shutting down...")
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

app = FastAPI(lifespan=lifespan)

async def background_poller():
    """
    Robust loop that won't die on error.
    """
    logger.info("Poller Started")
    while True:
        try:
            # Run the business logic
            process_active_sessions()
            
            # Wait for 30 seconds before checking again
            await asyncio.sleep(30) 
            
        except asyncio.CancelledError:
            logger.info("Poller stopping request received.")
            break
        except Exception as e:
            # CRITICAL: Catch generic errors so the loop doesn't die
            logger.error(f"Poller crashed (restarting in 10s): {e}")
            await asyncio.sleep(10)

@app.get("/")
def health_check():
    return {"status": "running", "mode": "production"}