import unittest
from unittest.mock import patch, MagicMock
import sys
import os

# Allow importing from the parent directory
sys.path.append(os.getcwd())

# Import your service logic
from app import services

class TestSleepServices(unittest.TestCase):

    def setUp(self):
        """Setup the 'Fake' Database before every test."""
        self.mock_db = MagicMock()
        self.mock_batch = MagicMock()
        # Wire up the batch: db.batch() returns our mock_batch
        self.mock_db.batch.return_value = self.mock_batch

    @patch("app.services.predict_batch") # Fake the ML model
    def test_end_to_end_flow(self, mock_predict):
        """
        Scenario:
        1. We find 1 ACTIVE session ("session_123").
        2. That session has 1 NEW reading (5 mins of data).
        3. We expect the code to Predict and Write to DB.
        """
        # --- ARRANGE (The Setup) ---
        services.db = self.mock_db

        # 1. Mock the "Active Session" Query
        # Simulates: db.collection("sleep_sessions").where("status"=="recording")
        mock_session = MagicMock()
        mock_session.id = "session_123"
        (self.mock_db.collection.return_value
             .where.return_value
             .stream.return_value) = [mock_session]

        # 2. Mock the "New Data" Query
        # Simulates: db.collection("sensor_readings").where(...).where("is_processed"==False)
        mock_reading = MagicMock()
        mock_reading.id = "reading_5min_point"
        mock_reading.to_dict.return_value = {
            "temperature": 25.0, 
            "humidity": 60.0, 
            "light": 10.0, 
            "sound_level": 45.0
        }
        
        # Chain the mocks so the query returns our fake reading
        (self.mock_db.collection.return_value
             .where.return_value
             .where.return_value
             .limit.return_value
             .stream.return_value) = [mock_reading]

        # 3. Mock the AI Prediction
        # We tell the fake model to return "88.5" when asked
        mock_predict.return_value = [88.5]

        # --- ACT (Run the Code) ---
        services.process_active_sessions()

        # --- ASSERT (The Proof) ---
        
        # PROOF 1: Did it predict?
        # Check if model was called with the data from the reading
        mock_predict.assert_called_once_with([[25.0, 60.0, 10.0, 45.0]])
        print("✅ Model was called with correct sensor data.")

        # PROOF 2: Did it Write the Score?
        # Check if batch.set() was called
        self.assertTrue(self.mock_batch.set.called)
        
        # Dig into the arguments to see WHAT was written
        args, _ = self.mock_batch.set.call_args
        written_data = args[1] # The second arg is the data dict
        
        self.assertEqual(written_data["session_id"], "session_123")
        self.assertEqual(written_data["score"], 88.5)
        print(f"✅ Database Write Verified: {written_data}")

        # PROOF 3: Did it Mark the Reading as Processed?
        # Check if batch.update() was called
        self.assertTrue(self.mock_batch.update.called)
        print("✅ Reading marked as processed.")

        # PROOF 4: Did it Commit?
        self.mock_batch.commit.assert_called_once()
        print("✅ Transaction committed successfully.")

if __name__ == "__main__":
    unittest.main()