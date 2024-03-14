import cv2
import numpy as np
import face_recognition
import time

# 훈련된 모델 불러오기
known_face_encodings = np.load('known_face_encodings.npy')
known_face_names = np.load('known_face_names.npy')

# 웹캠
video_capture = cv2.VideoCapture(0)

start_time = None  # 시작 시간을 저장할 변수 초기화

while True:
    ret, frame = video_capture.read()
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = np.ascontiguousarray(small_frame[:, :, ::-1])

    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    face_names = []
    for face_encoding in face_encodings:
        distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        min_value = min(distances)
        name = "Unknown"
        if min_value < 0.4:
            index = np.argmin(distances)
            name = known_face_names[index]
            if name != "Unknown" and start_time is None:
                start_time = time.time()

        face_names.append(name)

    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        if name == "Unknown":
            color = (0, 0, 255)
        else:
            color = (255, 0, 0)

        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    if all(name == face_names[0] for name in face_names) and start_time is not None and time.time() - start_time >= 5:
        cv2.putText(frame, "Success!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


    if (all(name == face_names[0] for name in face_names) and start_time is not None and time.time() - start_time >= 7):
        # success_time = time.time() # success 메시지가 표시된 시간을 저장
        # print("success time:",success_time)
        print("face:", name)
        break


    if all(name == "Unknown" for name in face_names) or not face_names:
        start_time = None

    # 화면 하단에 텍스트 추가
    cv2.putText(frame, "face recognition", (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255),
                1)

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
