
import json
from pathlib import Path

# Location of our JSON database file
DATA_FILE = Path(__file__).with_name("detections.json")


# Load existing detections
def load_detections():

    if DATA_FILE.exists():
        with DATA_FILE.open("r", encoding="utf-8") as file:
            return json.load(file)

    return []


# Save detections to JSON file
def save_detections(detections):

    with DATA_FILE.open("w", encoding="utf-8") as file:
        json.dump(detections, file, indent=4)
