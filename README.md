# Document-Scanner
Building a Document Scanner Using Open CV

## Use Case
The Document Scanner application facilitates easy scanning and storing of photos and important documents online. This application uses Python libraries to process images and convert them into scanned documents, mimicking the output of a traditional scanner.

The application allows users to:

a. Select a targeted portion of a document for scanning.

b. Process the selected area and output it as a clean, scanned-like image.

c. Export the scanned image for further use or storage.

# Project Structure

## 1. Jupyter Notebook

The project includes a Jupyter Notebook that serves as an interactive and exploratory tool for building and debugging the Document Scanner pipeline. It contains:

a. Code for preprocessing the image.

b. Functions for detecting document contours.

c erspective transformation to get a top-down view of the document.

d. Visualization of intermediate and final results.

# How to use the Jupyter Notebook

1. Open the notebook (Document_Scanner.ipynb) in JupyterLab or Jupyter Notebook.

2. Run the cells sequentially to execute the pipeline step by step.

3. Update the input image file path in the respective cell to use a different image.

4. View and analyze intermediate results such as edge detection, contour mapping, and final scanned output.

## 2. Python Script

he Python script (Document_Scanner.py) consolidates all the functionalities from the notebook into reusable functions. This script can be executed from another script for automation.

The script includes:

a. Core functions for preprocessing, contour detection, and perspective transformation.

b. A unified document_scanner function for end-to-end scanning.

## How to use the Script

A separate runner script (runner_scanner.py) automates the scanning process using input specified in a configuration file (config.yaml). This makes the pipeline user-friendly and easy to configure.

Example config.yaml:
input_image: "Images/image1.jpeg"

How to run:

1. Update the config.yaml file with the path to your input image.
2. Execute the runner script as : python runner_scanner.py

The script will read the image path from the YAML file, process the document, and display the scanned output.

## Contributor

Likitha Madhav Mohan Rekha

For questions or issues feel free to contact me.


