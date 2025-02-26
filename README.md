### LPR_demo
This project implements a real-time license plate recognition system using deep learning models for text detection and recognition. The pipeline is designed for modularity, allowing easy replacement of models or adaptation to different datasets.

## Folder structure
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

# Pipeline overview
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


