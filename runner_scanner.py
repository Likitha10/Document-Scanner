# run_scanner.py
import yaml
from Document_Scanner import document_scanner

# Load the YAML file
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

# Get the input image path from the YAML file
image_path = config.get("input_image", None)

if not image_path:
    raise ValueError("No input image specified in the configuration file.")

# Call the document scanner with the provided image path
document_scanner(image_path)
