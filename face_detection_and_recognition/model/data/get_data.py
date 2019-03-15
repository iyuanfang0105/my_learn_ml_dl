import face_recognition
import cv2
import pickle

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)

# Initialize some variables
faces = []
process_this_frame = 0
process_interval = 5

samples = 3000
while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    small_frame_rgb = small_frame[:, :, ::-1]

    # Only process every other frame of video to save time
    if process_this_frame % process_interval == 0:
        face_locations = face_recognition.face_locations(small_frame_rgb)
        face_embedding = face_recognition.face_encodings(small_frame_rgb)

    # extract the face
    for top, right, bottom, left in face_locations:
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 2
        right *= 2
        bottom *= 2
        left *= 2

        # Extract the region of the image that contains the face
        face_image = frame[top:bottom, left:right]

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

    if process_this_frame % process_interval == 0:
        faces.append({
            'face': face_image,
            'embedding': face_embedding
        })

    process_this_frame = process_this_frame + 1

    print("processing sample #%d / %d" % (process_this_frame, samples))

    if process_this_frame == samples:
        break

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()

with open('wind', 'wb') as fp:
    pickle.dump(faces, fp)

