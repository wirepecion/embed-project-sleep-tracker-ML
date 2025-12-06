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
    init_firebase()          # 1. Connect to Firestore
    load_model_into_memory() # 2. Load ML Model into RAM (Critical!)
    
    # 3. Start the Background Loop
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
    Robust infinite loop. It never dies, even if Firebase errors out.
    """
    logger.info("Poller Started - Monitoring Active Sessions")
    while True:
        try:
            # The Brain: Checks DB, Predicts, Writes Scores
            process_active_sessions()
            
            # Wait 30 seconds before next check
            await asyncio.sleep(30) 
            
        except asyncio.CancelledError:
            logger.info("Poller stopping request received.")
            break
        except Exception as e:
            # CRITICAL: If a crash happens, log it and restart loop in 10s
            logger.error(f"Poller crashed (restarting in 10s): {e}")
            await asyncio.sleep(10)

@app.get("/")
def health_check():
    return {"status": "running", "mode": "production"}

# Inside background_poller() loop:

async def background_poller():
    logger.info("Poller Started - Monitoring Active & Finished Sessions")
    while True:
        try:
            # Task A: Predict for sleeping users
            process_active_sessions()
            
            # Task B: Summarize for woke users
            # (We run this new function now)
            from app.services import process_finished_sessions
            process_finished_sessions()
            
            await asyncio.sleep(30) 
            
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Poller crashed: {e}")
            await asyncio.sleep(10)