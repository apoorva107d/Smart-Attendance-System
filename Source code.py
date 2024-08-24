import face_recognition
import cv2
import numpy as np
import csv
from datetime import datetime
import threading
# Setting Camera resolution
video_capture = cv2.VideoCapture(0)
video_capture.set(3, 1280)  # Width
video_capture.set(4, 720)   # Height
# Define face encodings and details
face_data = [
    {"name": "Apoorva Dhiman", "roll_number": "976", "encoding": None, "department": "comp. science", "year": "2"},
    {"name": "Bill Gates", "roll_number": "456", "encoding": None, "department": "Business", "year": "4"},
    {"name": "M S Dhoni", "roll_number": "789", "encoding": None, "department": "Sports", "year": "2"},
    {"name": "Elon Musk", "roll_number": "101", "encoding": None, "department": "Engineering", "year": "4"},
    {"name": "Steve Jobs", "roll_number": "202", "encoding": None, "department": "Business", "year": "4"},
    {"name": "Mark Zuckerberg", "roll_number": "303", "encoding": None, "department": "Computer Science", "year": "3"},
    {"name": "Narendra Modi", "roll_number": "404", "encoding": None, "department": "Politics", "year": "5"},
    {"name": "Ratan Tata", "roll_number": "505", "encoding": None, "department": "Business", "year": "4"},
    {"name": "Rohit Sharma", "roll_number": "606", "encoding": None, "department": "Sports", "year": "3"},
    {"name": "Virat Kohli", "roll_number": "707", "encoding": None, "department": "Sports", "year": "4"},
]

# Load face encodings
for data in face_data:
    image_path = f"photo/{data['name'].lower().replace(' ', '_')}.jpeg"
    data["encoding"] = face_recognition.face_encodings(face_recognition.load_image_file(image_path))[0]

# Extract information from face_data
known_face_encoding = [data["encoding"] for data in face_data]
known_faces_info = [(data["name"], data["roll_number"], data["department"], data["year"]) for data in face_data]

students = known_faces_info.copy()

def write_to_csv(name, roll_number, department, year, current_time):
    current_date = datetime.now().strftime("%Y-%m-%d")
    file_name = f'attendance_{current_date}.csv'

    with open(file_name, 'a', newline='') as f: 
        lnwriter = csv.writer(f)
        lnwriter.writerow([name, roll_number, department, year, current_time])

def process_frames():
    try:
        while True:
            _, frame = video_capture.read()

            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = small_frame[:, :, ::-1]

            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            now = datetime.now()

            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                matches = face_recognition.compare_faces(known_face_encoding, face_encoding)
                name = "Unknown"
                roll_number = ""
                department = ""
                year = ""

                face_distances = face_recognition.face_distance(known_face_encoding, face_encoding)
                best_match_index = np.argmin(face_distances)

                if matches[best_match_index]:
                    name, roll_number, department, year = known_faces_info[best_match_index]
                    color = (0, 255, 0)  # Green for recognized faces
                else:
                    color = (0, 0, 255)  # Red for unknown faces

                cv2.rectangle(frame, (left * 4, top * 4), (right * 4, bottom * 4), color, 3)

                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, f"{name} ({roll_number}, {department}, {year})", (left * 4 + 6, bottom * 4 + 30), font, 0.8, color, 2)

                if (name, roll_number, department, year) in known_faces_info and (name, roll_number, department, year) in students:
                    students.remove((name, roll_number, department, year))
                    current_time = now.strftime("%H-%M-%S")
                    write_to_csv(name, roll_number, department, year, current_time)

            cv2.imshow("Attendance system", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except Exception as e:
        print(f"Error: {e}")
    finally:
        video_capture.release()
        cv2.destroyAllWindows()
# Open the CSV file for writing the header
current_date = datetime.now().strftime("%Y-%m-%d")
file_name = f'attendance_{current_date}.csv'

with open(file_name, 'w', newline='') as f:
    lnwriter = csv.writer(f)
    lnwriter.writerow(["Name", "Roll Number", "Department", "Year", "Time"])

# Start processing frames and absentee checking in separate threads
threading.Thread(target=process_frames).start()
