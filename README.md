# LPR_demo
This project implements a real-time license plate recognition system using deep learning models for text detection and recognition. The pipeline is designed for modularity, allowing easy replacement of models or adaptation to different datasets.

## Dependencies, setup and usage
It is recommended to use git with support for files larger than 100MB,

```git clone https://github.com/antaresxo/LPR_demo```

```git install git-lfs```

```git pull```

and make sure everything is pulled with ResNet_CTC being 170MB!

To perform a test run, install dependencies and run main from project folder.
    ```pip install -r requirements.txt```
Most imports in main.py are built-in, but these external libraries are required:
    numpy
    opencv-python

To run, just do
    ````python main.py```

To close press Q !

NOTE: If you want to set webcam as the video input, change the path of the cv videostream to 0.

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

Some limitations:
    Very rudimentary skipping of frames to speed up testing
    Handle edgecases with custom license plates
    Handle up close reads of plates where two text boxes are recognized


## Troubleshooting
If opencv cant open window try: ```pip install opencv-contrib-python```

If networks cant be initialised after downloading as zip, 
try downloading the models seperately from below, 

these are the pretrained opencv examples which are taken from 
(https://github.com/clovaai/deep-text-recognition-benchmark),

(https://drive.google.com/drive/folders/1cTbQ3nuZG-EKWak6emD_s8_hHXWz7lAr)

and the DB detection models here

(https://drive.google.com/drive/folders/1qzNCHfUJOS0NEUOIKn69eCtxdlNPpWbq)

Or just download everything through git clone.

OBS !!! Really make sure to ```git install git-lfs``` 

verify ```git lfs install```

## Final words
This concludes the Norrspect.ai X T4 Innovation Group collaboration!
Good work everyone involved!!!

