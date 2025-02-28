import cv2 as cv
import numpy as np

import re
from collections import defaultdict
from datetime import datetime, timedelta

import csv
import sqlite3



def log_numberplate_read(plate):
    """
    Logs a number plate read into the database. If the plate is new, it stores the first and latest read times.
    If the plate already exists, it updates the latest read time.
    """
    conn = sqlite3.connect("numberplates.db")  # Connect to the SQLite database
    cursor = conn.cursor()
    
    cursor.execute('''CREATE TABLE IF NOT EXISTS numberplates (
                        plate TEXT PRIMARY KEY,
                        first_seen TEXT,
                        last_seen TEXT)''')  # Create table if it doesn't exist
    
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # Get the current timestamp
    cursor.execute("SELECT first_seen FROM numberplates WHERE plate = ?", (plate,))
    result = cursor.fetchone()
    
    if result is None:
        cursor.execute("INSERT INTO numberplates (plate, first_seen, last_seen) VALUES (?, ?, ?)", (plate, now, now))
    else:
        cursor.execute("UPDATE numberplates SET last_seen = ? WHERE plate = ?", (now, plate))
    
    conn.commit()
    conn.close()

def export_to_csv(filename="numberplates.csv"):
    """
    Exports the database contents to a CSV file.
    The default file name is "numberplates.csv", but this can be changed by passing a different filename.
    The file is saved in the same directory as the script.
    """
    conn = sqlite3.connect("numberplates.db")  # Connect to the SQLite database
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM numberplates")  # Fetch all records
    data = cursor.fetchall()
    conn.close()
    
    with open(filename, mode='w', newline='') as file:  # Open CSV file for writing
        writer = csv.writer(file)
        writer.writerow(["Plate", "First Seen", "Last Seen"])  # Write headers
        writer.writerows(data)  # Write data rows



def validate_license_plate(plate):
    # Forbidden letter combinations
    FORBIDDEN_COMBINATIONS = {
        "APA", "ARG", "ASS", "BAJ", "BSS", "CUC", "CUK", "CUM", "DUM",
        "ETA", "ETT", "FAG", "FAN", "FEG", "FEL", "FEM", "FES", "FET",
        "FNL", "FUC", "FUK", "FUL", "GAM", "GAY", "GEJ", "GEY", "GHB",
        "GUD", "GYN", "HAT", "HBT", "HKH", "HOR", "HOT", "KGB", "KKK",
        "KUC", "KUF", "KUG", "KUK", "KYK", "LAM", "LAT", "LEM", "LOJ",
        "LSD", "LUS", "MAD", "MAO", "MEN", "MES", "MLB", "MUS", "NAZ",
        "NRP", "NSF", "NYP", "OND", "OOO", "ORM", "PAJ", "PKK", "PLO",
        "PMS", "PUB", "RAP", "RAS", "ROM", "RPS", "RUS", "SEG", "SEX",
        "SJU", "SOS", "SPY", "SUG", "SUP", "SUR", "TBC", "TOA", "TOK",
        "TRE", "TYP", "UFO", "USA", "WAM", "WAR", "WWW", "XTC", "XTZ",
        "XXL", "XXX", "ZEX", "ZOG", "ZPY", "ZUG", "ZUP", "ZOO"
    }

    # Allowed letters (I, Q, V, Å, Ä, Ö are forbidden)
    ALLOWED_LETTERS = "ABCDEFGHJKLMNOPRSTUWXYZ"

    # Regular expression to match license plate format
    PLATE_PATTERN = re.compile(rf"^([{ALLOWED_LETTERS}]{{3}}) ?(\d{{3}}|\d{{2}}[{ALLOWED_LETTERS}])$")

    plate = plate.replace(" ", "")  # Remove spaces and convert to uppercase

    match = PLATE_PATTERN.fullmatch(plate)
    if not match:
        return False

    letters, numbers_or_letter = match.groups()

    if letters in FORBIDDEN_COMBINATIONS:  # Check forbidden letter combinations
        return False

    if plate[-1] == "O" or plate[2] == "O":  # The letter O cannot be the last character (both numbers and letters, 3rd or 6th character).
        return False

    if numbers_or_letter.isdigit():
        return numbers_or_letter != "000"  # 000 is invalid

    return True

# think buffer..
# Maximum number of license plates to store
MAX_PLATES = 1000

# Dictionary to store license plates and their scan times
registration_dict = defaultdict(list)

# Function to add a license plate and check if it meets the criteria
def data_filtering(plate, max_time_minutes=2, min_scans=3):
    plate = plate.replace(" ", "")  # Remove all spaces from the plate
    timestamp = datetime.now()
    registration_dict[plate].append(timestamp)

    # If the dictionary exceeds MAX_PLATES, remove the oldest plate
    if len(registration_dict) > MAX_PLATES:
        oldest_plate = min(registration_dict, key=lambda k: min(registration_dict[k]))
        registration_dict.pop(oldest_plate)

    # Check if the plate meets the conditions to be considered valid
    timestamps = registration_dict[plate]
    timestamps.sort()  # Sort timestamps to compare correctly
    
    # Check if there are at least 3 scans within the specified time interval
    for i in range(len(timestamps) - min_scans + 1):
        if timestamps[i + min_scans - 1] - timestamps[i] <= timedelta(minutes=max_time_minutes):
            return plate  # Return the plate if it is valid
    
    return None  # Return None if the plate is not valid

def main():

    # Create a new named window
    kWinName = "DB_TD500_resnet50 + ResNet_CTC"
    cv.namedWindow(kWinName, cv.WINDOW_NORMAL)

    # videostream selection (crude)
    isUsingWebcam : bool = True
    stream : cv.VideoCapture
    if isUsingWebcam:
        stream = cv.VideoCapture(0)
    else: 
        stream = cv.VideoCapture("./reference_material/VID_20250214114317927.mp4")

    ### DETECTOR ###
    detector = cv.dnn.TextDetectionModel_DB("./models/DB_TD500_resnet50.onnx")
    
    # Post-processing parameters
    binThresh : float = 0.3
    polyThresh : float = 0.5
    maxCandidates : int = 200
    unclipRatio : float = 2.0
    detector.setBinaryThreshold(binThresh)
    detector.setPolygonThreshold(polyThresh)
    detector.setMaxCandidates(maxCandidates)
    detector.setUnclipRatio(unclipRatio)

    # Normalization parameters
    detScale : float = 1.0 / 255.0
    detMean : cv.typing.Scalar = (122.67891434, 116.66876762, 104.00698793)

    # The input shape
    detInputSize : cv.typing.Size = (736, 736)
    detector.setInputParams(detScale, detInputSize, detMean)

    ### RECOGNIZER ### 
    recognizer = cv.dnn.TextRecognitionModel("./models/ResNet_CTC.onnx")

    recognizer.setDecodeType("CTC-greedy")
    recognizer.setVocabulary("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    
    # Normalization parameters
    recScale : float = 1.0 / 127.5
    recMean : cv.typing.Scalar = (127.5, 127.5, 127.5)

    # The input shape
    recInputSize : cv.typing.Size = (100, 32)

    recognizer.setInputParams(recScale, recInputSize, recMean)

    # gui?
    # duct tape fix for detecting two boxes on a plate!! 
    lastRecResult = ""
    # fps
    tickmeter = cv.TickMeter()
    count = 0 # speedup
    while cv.waitKey(1) < 0:
        # Read frame
        hasFrame, frame = stream.read()
        if not hasFrame:
            cv.waitKey()
            break

        # for faster playback of material
        count = count + 1
        if count % 20 != 0 and not isUsingWebcam:
            continue

        # just making sure that drawn elements dont get sent to models 
        frame_with_elements = frame.copy()

        tickmeter.start()
        detResults = detector.detect(frame)
        tickmeter.stop()

        cv.polylines(frame_with_elements, detResults[0], True, (0, 255, 0), 2)

        for quadrangle in detResults[0]:
            quadrangle_np = np.array(quadrangle, dtype=np.float32)

            # skip small detections
            height = np.linalg.norm(quadrangle_np[0] - quadrangle_np[1])
            width = np.linalg.norm(quadrangle_np[1] - quadrangle_np[2])
            #print("differential ", width , height) # for adjusting threshhold!
            if width < 100 or height < 30:
                cv.putText(frame_with_elements, "NOT A PLATE", (int(quadrangle[1][0]), int(quadrangle[1][1])) , cv.FONT_HERSHEY_SIMPLEX, 1, (255, 127, 0))
                continue

            cropped = fourPointsTransform(frame, quadrangle_np)
            cropped = cv.cvtColor(cropped, cv.COLOR_BGR2GRAY)
            cropped = cv.equalizeHist(cropped)
            # Resize to (100, 32) before CRNN
            cropped = cv.resize(cropped, (100, 32))

            #cv.imshow("fourPointsTransform + Grayscale", cropped)

            tickmeter.start()
            recResult = recognizer.recognize(cropped) # or blob 
            tickmeter.stop()
          
            cv.putText(frame_with_elements, recResult, (int(quadrangle[1][0]), int(quadrangle[1][1])) , cv.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0))

            print(recResult)
            
            # "handle" edgecase: if demo recognizes a numberplate as two text boxes
            if isUsingWebcam and len(recResult)==3 and len(lastRecResult)==3:
                formatedResult = lastRecResult + recResult
                print("edgecase detected: ", formatedResult)
                if validate_license_plate(formatedResult):
                    if (data_filtering(formatedResult) is not None):
                        log_numberplate_read(formatedResult)
                        export_to_csv("numberplates.csv")     
            else:
                # pass it along the rest of the pipeline
                if validate_license_plate(recResult):
                    if (data_filtering(recResult) is not None):
                        log_numberplate_read(recResult)
                        export_to_csv("numberplates.csv")
                        # gui 
            lastRecResult = recResult

            

        # Put efficiency information
        label = 'Inference time: %.2f ms' % (tickmeter.getTimeMilli())
        cv.putText(frame_with_elements, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))

        # Display the frame
        cv.imshow(kWinName, frame_with_elements)
        tickmeter.reset()


############ Utility functions ############

def fourPointsTransform(frame, vertices):
    vertices = np.asarray(vertices)
    outputSize = (100, 32)
    targetVertices = np.array([
        [0, outputSize[1] - 1],
        [0, 0],
        [outputSize[0] - 1, 0],
        [outputSize[0] - 1, outputSize[1] - 1]], dtype="float32")

    rotationMatrix = cv.getPerspectiveTransform(vertices, targetVertices)
    result = cv.warpPerspective(frame, rotationMatrix, outputSize)
    return result

def decodeText(scores):
    text = ""
    #alphabet = "0123456789abcdefghijklmnopqrstuvwxyz"
    #alphabet = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    alphabet  = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    for i in range(scores.shape[0]):
        c = np.argmax(scores[i][0])
        if c != 0:
            text += alphabet[c - 1]
        else:
            text += '-'

    # adjacent same letters as well as background text must be removed to get the final output
    char_list = []
    for i in range(len(text)):
        if text[i] != '-' and (not (i > 0 and text[i] == text[i - 1])):
            char_list.append(text[i])
    return ''.join(char_list)


if __name__ == '__main__':
    main()