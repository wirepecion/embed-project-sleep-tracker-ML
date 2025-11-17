#!/usr/bin/env python3
"""
Example script demonstrating how to use the Sleep Quality Prediction API
"""

import requests
import json

# API base URL
BASE_URL = "http://localhost:8000"

def test_api():
    """Test the Sleep Quality Prediction API"""
    
    print("=" * 60)
    print("Sleep Quality Prediction API - Example Usage")
    print("=" * 60)
    
    # Test 1: Optimal conditions
    print("\n1. Testing OPTIMAL sleep conditions:")
    print("-" * 40)
    optimal_data = {
        "temperature": 20,
        "humidity": 40,
        "light": 5,
        "sound": 30
    }
    print(f"Input: {json.dumps(optimal_data, indent=2)}")
    
    response = requests.post(f"{BASE_URL}/predict", json=optimal_data)
    result = response.json()
    print(f"\nOutput:")
    print(f"  Sleep Quality: {result['sleep_quality_percent']}%")
    print(f"  Reasoning: {result['reasoning']}")
    
    # Test 2: Poor conditions
    print("\n\n2. Testing POOR sleep conditions:")
    print("-" * 40)
    poor_data = {
        "temperature": 30,
        "humidity": 70,
        "light": 300,
        "sound": 85
    }
    print(f"Input: {json.dumps(poor_data, indent=2)}")
    
    response = requests.post(f"{BASE_URL}/predict", json=poor_data)
    result = response.json()
    print(f"\nOutput:")
    print(f"  Sleep Quality: {result['sleep_quality_percent']}%")
    print(f"  Reasoning: {result['reasoning']}")
    
    # Test 3: Moderate conditions
    print("\n\n3. Testing MODERATE sleep conditions:")
    print("-" * 40)
    moderate_data = {
        "temperature": 18,
        "humidity": 45,
        "light": 25,
        "sound": 50
    }
    print(f"Input: {json.dumps(moderate_data, indent=2)}")
    
    response = requests.post(f"{BASE_URL}/predict", json=moderate_data)
    result = response.json()
    print(f"\nOutput:")
    print(f"  Sleep Quality: {result['sleep_quality_percent']}%")
    print(f"  Reasoning: {result['reasoning']}")
    
    # Test 4: Health check
    print("\n\n4. Checking API health:")
    print("-" * 40)
    response = requests.get(f"{BASE_URL}/health")
    health = response.json()
    print(f"Status: {health['status']}")
    print(f"Firebase: {health['firebase']}")
    print(f"Timestamp: {health['timestamp']}")
    
    print("\n" + "=" * 60)
    print("All tests completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    try:
        test_api()
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to API.")
        print("Please make sure the server is running with: python main.py")
    except Exception as e:
        print(f"Error: {e}")
