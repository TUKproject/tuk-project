import face_recognition  # 라이브러리로 face_recognition 
import cv2
import numpy as np
import os
from sklearn import svm   # svm 사용

# 사람 학습된 사람들의 얼굴은 구별함
# 서양인 얼굴은 없으면 unknown으로 잘뜨지만 동양인 얼굴은 unknown으로 잘안뜨고 다른 사람을 가리키는 문제 발생

known_face_encodings = []
known_face_names = []


train_dir = os.listdir('address/') # 경로 설정


for person in train_dir:     # for loop 돌면서
    pix = os.listdir("address/" + person)  

   
    for person_img in pix:
        face = face_recognition.load_image_file("address/" + person + "/" + person_img)  #불러옴
        face_encodings = face_recognition.face_encodings(face)
        if len(face_encodings) > 0:   # 얼굴인식되면 추가
            face_enc = face_encodings[0]
            known_face_encodings.append(face_enc)
            known_face_names.append(person)
        else:
            continue 

face_locations = []
face_encodings = []
face_names = []
process_this_frame = True


clf = svm.SVC(gamma='scale')
clf.fit(known_face_encodings, known_face_names)


video_capture = cv2.VideoCapture(0)

while True:
   
    ret, frame = video_capture.read()

    
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    
    rgb_small_frame =np.ascontiguousarray(small_frame[:, :, ::-1])


    if process_this_frame:
    
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"   # 기본적으로 unknown

            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame

# 이전 코드와 같음
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

    # 화면에 보여줌
    cv2.imshow('Video', frame)

    # q누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()