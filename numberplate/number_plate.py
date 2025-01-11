import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import cv2
import easyocr
import mysql.connector


def get_db_connection():
    return mysql.connector.connect(
        host="127.0.0.1",
        user="root",
        password="Sahil@04",
        database="number_plate_db"
    )

conn = get_db_connection()
cursor = conn.cursor()

def insert_number_plate(scanned_text):
    sql = "INSERT INTO number_plates (scanned_text) VALUES (%s)"
    values = (scanned_text,)  # Ensure it's a tuple
    cursor.execute(sql, values)
    conn.commit()
    print(f"Inserted into database: {scanned_text}")


def extract_text_from_image(image_path):
    reader = easyocr.Reader(['en'])
    result = reader.readtext(image_path, detail=0)
    return " ".join(result)

# Real-time car plate detection
car_plate = "model/indian_license_plate.xml"
cap = cv2.VideoCapture(0)

cap.set(3, 640)  # Width
cap.set(4, 480)  # Height

min_area = 500
count = 0

while True:
    success, img = cap.read()
    if not success:
        print("Failed to capture image")
        continue

    # Load the cascade classifier
    plate = cv2.CascadeClassifier(car_plate)

    # Convert to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect plates
    plates = plate.detectMultiScale(img_gray, 1.1, 4)

    for (x, y, w, h) in plates:
        area = w * h
        if area > min_area:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img, "Number Plate", (x, y - 5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 255), 2)

            # Extract region of interest (ROI)
            img_roi = img[y:y+h, x:x+w]
            cv2.imshow("ROI", img_roi)

    # Display the result
    cv2.imshow("Result", img)

    # If 's' is pressed, save the scanned plate and extract text
    if cv2.waitKey(1) & 0xFF == ord('s'):
        # Save the image of the detected plate
        plate_image_path = f"plates/scanned_img_{count}.jpg"
        cv2.imwrite(plate_image_path, img_roi)

        # Extract text using EasyOCR
        scanned_text = extract_text_from_image(plate_image_path)
        print(f"Scanned Text: {scanned_text}")

        # Insert the scanned text into the database
        insert_number_plate(scanned_text)

        # Display confirmation on the screen
        cv2.rectangle(img, (0, 200), (640, 300), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, "Plate Saved", (150, 265), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255), 2)
        cv2.imshow("Results", img)
        cv2.waitKey(500)
        count += 1

    # Quit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
conn.close()
