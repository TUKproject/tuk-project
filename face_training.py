import os
import face_recognition
import numpy as np

# 학습할 디렉토리 설정
train_dir = os.listdir('dataset/')

known_face_encodings = []
known_face_names = []

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


# 훈련된 모델 저장
np.save('known_face_encodings.npy', known_face_encodings)
np.save('known_face_names.npy', known_face_names)
