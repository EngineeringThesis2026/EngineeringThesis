"""Wait for services to start and open browser."""
import time
import webbrowser

URL = "http://localhost:8501"
WAIT_SECONDS = 5

print(f"Waiting {WAIT_SECONDS} seconds for services to start...")
time.sleep(WAIT_SECONDS)
print(f"Opening {URL}...")
webbrowser.open(URL)
