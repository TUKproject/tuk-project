import face_recognition
import cv2
import numpy as np
import os
import time
from sklearn import svm

known_face_encodings = []
known_face_names = []

# 불러올 디렉토리 설정
train_dir = os.listdir('dataset/')  # 경로 넣어줌

# for문 각 사람별 반복
for person in train_dir:
    pix = os.listdir("dataset/" + person)

    # 각 사람의 이미지
    for person_img in pix:
        face = face_recognition.load_image_file("dataset/" + person + "/" + person_img)
        face_encodings = face_recognition.face_encodings(face)
        if len(face_encodings) > 0:  # 얼굴 인코딩 있으면
            face_enc = face_encodings[0]
            known_face_encodings.append(face_enc)
            known_face_names.append(person)
        else:
            continue

        # 값 초기화
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

# SVC 분류기 생성 , 훈련
clf = svm.SVC(gamma='scale')
clf.fit(known_face_encodings, known_face_names)

# 웹캠
video_capture = cv2.VideoCapture(0)

start_time = None  # 시작 시간을 저장할 변수 초기화
success_time = None  # success 메시지가 표시된 시간을 저장할 변수 초기화

while True:

    ret, frame = video_capture.read()
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = np.ascontiguousarray(small_frame[:, :, ::-1])

    if process_this_frame:

        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:

            # 이전과 달라진 부분
            # 현재 처리중인 얼굴과 이미 학습된 얼굴의 거리 비교
            distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            # 최소값 변수 설정
            min_value = min(distances)
            name = "Unknown"  # 기본적으로 unknown
            if min_value < 0.4:  # min_value가 0.4보다 작으면
                index = np.argmin(distances)  # 가장 작은거 반환
                name = known_face_names[index]
                # 얼굴이 인식되었을 때부터 시간을 시작
                if  name!= "Unknown" and start_time is None:
                    start_time = time.time()

            face_names.append(name)

    process_this_frame = not process_this_frame

    # 화면 보여줌
    for (top, right, bottom, left), name in zip(face_locations, face_names):

        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        if name == "Unknown":
            color = (0, 0, 255)
        else:
            color = (255, 0, 0)

        # 얼굴 박스
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

        # 아래는 이름뜨는 건데 보류
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # 성공 메시지를 표시할 시간이 되면 화면 상단에 표시
    if all(name != "Unknown" for name in face_names) and start_time is not None and time.time() - start_time >= 15:
        success_time = time.time()
        cv2.putText(frame, "Success!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        # success 메시지가 표시된 시간을 저장

    if all(name == "Unknown" for name in face_names):
        start_time = None

    # 화면 하단에 텍스트 추가
    cv2.putText(frame, "face recognition", (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255),
                1)

    if success_time is not None and time.time() - success_time >= 2:
        break

    # 화면에 보여줌
    cv2.imshow('Video', frame)

    # Success! 메시지가 표시된 후 2초가 경과하면 종료


    # q누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
