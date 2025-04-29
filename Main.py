import subprocess
import cv2
import numpy as np
from imutils.perspective import four_point_transform
import serial

cap = cv2.VideoCapture(1)

count = 0
scale = 0.5
done = False

WIDTH, HEIGHT = 1920, 1080
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

def image_processing(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)

    return threshold

def scan_detection(image):
    global document_contour

    document_contour = np.array([[0, 0], [WIDTH, 0], [WIDTH, HEIGHT], [0, HEIGHT]])

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, threshold = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(threshold, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    max_area = 0
    for c in contours:
        area = cv2.contourArea(c)
        if area > 1000:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.015 * peri, True)
            if area > max_area and len(approx) == 4:
                document_contour = approx
                max_area = area

    cv2.drawContours(frame, [document_contour], -1, (0, 255, 0), 3)



while True:
    while not done:
        #Image Scanning
        _, frame = cap.read()
        frame_copy = frame.copy()

        scan_detection(frame_copy)

        cv2.imshow("Original", cv2.resize(frame, (int(scale * WIDTH), int(scale * HEIGHT))))

        warped = four_point_transform(frame_copy, document_contour.reshape(4, 2))

        processed = image_processing(warped)
        processed = processed[10:processed.shape[0] - 10, 10:processed.shape[1] - 10]
        width = processed.shape[0]
        height = processed.shape[1]
        if width > 0 and height > 0:
            cv2.imshow("Processed", processed)

        pressed_key = cv2.waitKey(1) & 0xFF
        if pressed_key == ord('q'):
            break

        elif pressed_key == ord('s'):
            processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2RGB)
            cv2.imwrite("img" + str(count) + ".jpg", processed)
            count += 1
            done = True
    cv2.destroyAllWindows()

    #Send to convert to GCode
    subprocess.run(["python","image_to_gcode.py","--input", "img" + str(count-1) + ".jpg","--output", "img" + str(count-1) + ".gcode", "--threshold", "100"])

    #Add code to send to the STM32
    GVec = []
    XVec = []
    YVec = []

    ser = serial.Serial('COM3', 115200)

    file = open('GATECHLOGO.nc', 'r')
    read = file.readlines()

    for line in read:
        arr = str.split(line, " ")
        G = int(arr[0].replace("G", ""))
        X = int(arr[1].replace("X", ""))
        Y = -int(arr[2].replace("Y", ""))

        # GVec.append(G)
        # XVec.append(X)
        # YVec.append(Y)

        ser.write(b"G" + G.to_bytes(1))
        ser.write(b"X" + X.to_bytes(1))
        ser.write(b"Y" + Y.to_bytes(1))

    print("done")
    ser.write(b"D")
    ser.close()

    done = False