import asyncio
import aiohttp
import json
import time
import random
from datetime import datetime, timedelta

# API URL
url = "https://uchyzes6unae7t6ngihhqeuneq0ezcvk.lambda-url.us-east-2.on.aws/forecast"

# HTTP Headers
headers = {
    'Auth': '3b14bf4897ef89cd08b10a0a5dfbc206',
    'Content-Type': 'application/json'
}

# Semaphore to limit concurrent requests
semaphore = asyncio.Semaphore(50)  # Adjusted for better concurrency

# Store response times
response_times = []

def generate_random_payload():
    """Generate a random payload with random values in the series."""
    start_date = datetime(2023, 1, 1)
    series = []
    
    for i in range(25):  # Generate 25 months of data
        random_value = random.randint(100, 600)  # Random value between 100 and 600
        date_str = (start_date + timedelta(days=30 * i)).strftime('%Y-%m-%d')
        series.append({"Fecha": date_str, "value": random_value})
    
    payload = {
        "parameters": {
            "ts": random.randint(10, 15),  # Random TS value
            "df": "MS",
            "sp": 12,
            "fp": random.randint(10, 15)  # Random FP value
        },
        "series": series
    }
    return json.dumps(payload)

async def send_request(session, request_id):
    """Send a single request to the API with a semaphore limit."""
    async with semaphore:
        try:
            random_payload = generate_random_payload()
            start_time = time.time()
            async with session.post(url, headers=headers, data=random_payload) as response:
                result = await response.text()
                end_time = time.time()
                elapsed_time = end_time - start_time
                response_times.append(elapsed_time)
                print(f"Request {request_id}: {response.status} - {result[:100]} - Time: {elapsed_time:.2f}s")  # Print partial response
        except Exception as e:
            print(f"Request {request_id} failed: {str(e)}")

async def stress_test():
    """Send multiple requests with random data."""
    start_time = time.time()
    
    async with aiohttp.ClientSession() as session:
        tasks = [send_request(session, i) for i in range(1, 1000)]
        await asyncio.gather(*tasks)
    
    total_time = time.time() - start_time
    avg_response_time = sum(response_times) / len(response_times) if response_times else 0
    print(f"Test completed in {total_time:.2f} seconds.")
    print(f"Average response time: {avg_response_time:.2f} seconds.")

# Run the test
asyncio.run(stress_test())
