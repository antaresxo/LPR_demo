# LPR_demo
This project implements a real-time license plate recognition system using deep learning models for text detection and recognition. The pipeline is designed for modularity, allowing easy replacement of models or adaptation to different datasets.

## Dependencies
To perform a test run, install dependencies and run main from project folder.
```pip install -r requirements.txt```
Most imports in main.py are built-in, but these external libraries are required:
    numpy
    opencv-python

## Folder structure
```
    project_root/
    │── models/                  # Contains detection and recognition ONNX models
    │   ├── DB_TD500_resnet50.onnx
    │   ├── ResNet_CTC.onnx
    │── reference_material/      # Stores sample footage and test images
    │── numberplates.csv         # Log of recognized license plates (generated at runtime)
    │── numberplates.db          # SQLite database storing recognized plates (generated at runtime)
    │── main.py                  # Core pipeline execution
    │── requirements.txt         # Dependencies
    │── README.txt               # This file
```

## Pipeline overview
Frame Extraction: Reads frames from a video feed or static images.

Text Detection (DB Model):

    Uses DB_TD500_resnet50.onnx for detecting text regions.

    The detection model predicts bounding boxes around potential text.

    Preprocessing ensures the model gets correctly scaled inputs.

Text Recognition (ResNet CTC):
    Preprocessing using a four points perspective transform, grayscale and histogram equalization. 

    Extracted text regions are passed to ResNet_CTC.onnx.

    This model predicts sequences of characters using a CTC-based decoding strategy.

Filtering & Validation:
    Checks if the possible plate is in line with common formats for swedish numberplates.

    A detected plate is only considered valid if read multiple times (simple frequency-based filtering, no confidence matching yet).

    The final result is stored in numberplates.csv and numberplates.db along with time logged.

## Future improvements
Confidence-Based Filtering: Instead of just counting repeated reads, implement confidence-based filtering using OCR scores.

Better preprocessing to function even in snow heavy enviroments. Could include augmented datasets, different model structures or another approach for character recognition.

GUI Integration: To display real-time logs and allow manual validation along with being more userfriendly.

Actually check active tickets through some parking API. 

