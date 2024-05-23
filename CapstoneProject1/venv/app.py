from flask import Flask, render_template, redirect, url_for, Response, jsonify, session, flash, request
import cv2
import numpy as np
import face_recognition
import time
from flask_socketio import SocketIO, send
import torch
import pandas as pd
import requests
import git
import threading
from pathlib import Path
import pathlib
import mysql.connector

# from yolov5.models.experimental import attempt_load
# import yolov5

app = Flask(__name__)
app.secret_key = 'your_secret_key'
# app.config['SERVER_NAME'] = 'localhost:5000'  # 예시 도메인 및 포트
# app.config['APPLICATION_ROOT'] = '/'
# app.config['PREFERRED_URL_SCHEME'] = 'http'

enable_switch = False
model = None
bmodel = None
# model1 = None
# model2 = None
ch_switch1 = False
ch_switch0 = False
b_switchCoat = False
b_switchShoes = False
detection_status = {
    'gloves': False,
    'goggles': False,
    'labcoat': False
}


temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

def load_yolov5_model(weights_path):
    return torch.hub.load('ultralytics/yolov5', 'custom', path=weights_path, force_reload=True)

# socketio = SocketIO(app)

@app.route('/')
def index():
    # enable_switch = False
    # ch_switch1 = False
    # ch_switch0 = False
    # b_switchCoat = False
    # b_switchShoes = False
    # detection_status = {
    #     'gloves': False,
    #     'goggles': False,
    #     'labcoat': False
    # }
    return render_template('main.html')

def db_connection():
    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="0000",
        database="mydb",
        port="3306"
    )
    return conn

@app.route('/login', methods=['GET', 'POST'])
def login():
    msg = ''
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        conn = db_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM admin WHERE admin_id = %s AND admin_pw = %s', (username, password,))
        account = cursor.fetchone()

        if account:
            session['username'] = account[1]
            return redirect(url_for('index'))
        else:
            msg = '등록된 관리자가 아닙니다! 다시입력해보세요'
    return render_template('login.html', error=msg)
@app.route('/logout')
def logout():
    session.pop('username', None)  # 세션에서 사용자 정보 삭제
    return redirect(url_for('index'))

@app.route('/admin')
def admin():
    if 'username' in session:
        return render_template('adminpage.html')
    else:
        flash('권한이 없습니다. 로그인하세요')
        return redirect(url_for('login'))

@app.route('/start')
def start():
    return redirect(url_for('start_page'))

@app.route('/start_page', methods=['GET','POST'])
def start_page():
    global enable_switch
    # enable_switch = False
    print("start_page", enable_switch)
    return  render_template('face_rec.html', enable_switch=enable_switch)

@app.route('/check_enable_switch')
def check_enable_switch():
    global enable_switch
    return jsonify({'enable_switch': enable_switch})

@app.route('/disable_enable_switch', methods=['POST'])
def disable_enable_switch():
    global enable_switch
    enable_switch = False
    return jsonify({'success': True, 'enable_switch': enable_switch})


@app.route('/add_user')
def add_user():
    if 'username' in session:
        return   render_template('add_user.html')
    else:
        flash('잘못된 접근입니다.')
        return redirect(url_for('login'))


@app.route('/user_info')
def user_info():
    if 'username' in session:
        return   render_template('user_info.html')
    else:
        flash('잘못된 접근입니다.')
        return redirect(url_for('login'))

@app.route('/access_rec')
def access_rec():
    if 'username' in session:
        return   render_template('access_rec.html')
    else:
        flash('잘못된 접근입니다.')
        return redirect(url_for('login'))

@app.route('/lab_opt')
def lab_opt():
    return  redirect(url_for('lab_page'))

@app.route('/lab_page')
def lab_page():
    return   render_template('lab_opt.html')

@app.route('/basic_opt')
def basic_opt():
    return redirect(url_for('basic_page'))

@app.route('/basic_page')
def basic_page():
    global bmodel
    global b_switchCoat, b_switchShoes
    # bmodel = load_yolov5_model('weights/0401m3.pt')
    bmodel = load_yolov5_model('weights/best.pt')
    return render_template('basic.html',b_switchCoat=b_switchCoat, b_switchShoes=b_switchShoes)

@app.route('/chemical_opt')
def chemical_opt():
    return  redirect(url_for('chemical_page'))

@app.route('/chemical_page')
def chemical_page():
    global model
    global ch_switch0, ch_switch1
    model= load_yolov5_model('weights/best.pt')
    return render_template('chemical.html', ch_switch0=ch_switch0, ch_switch1=ch_switch1)

#얼굴 인식
def gen_frames():
    # 훈련된 모델 불러오기
    known_face_encodings = np.load('known_face_encodings.npy')
    known_face_names = np.load('known_face_names.npy')

    # 웹캠
    video_capture = cv2.VideoCapture(0)

    start_time = None  # 시작 시간을 저장할 변수 초기화
    global enable_switch

    while True:
        success, frame = video_capture.read()
        if not success:
            break
        else:
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
                # cv2.putText(frame, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            if (all(name == face_names[0] for name in face_names) and start_time is not None and time.time() - start_time >= 7):
                print("face:", name)
                enable_switch = True
                print("gen_frame",enable_switch)
                break

            if all(name == "Unknown" for name in face_names) or not face_names:
                start_time = None

            # 화면 하단에 텍스트 추가
            cv2.putText(frame, "face recognition", (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255),
                        1)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    video_capture.release()


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

#화학 실험실 웹캠
def gen_chemical1():
    global ch_switch1, ch_switch0
    global model
    global detection_status
    model1= model
    # model = torch.hub.load('ultralytics/yolov5', 'custom',
    #                        path='weights/best.pt', force_reload=True)

    # 각 객체별 신뢰도 임계값 설정
    confidence_thresholds = {
        'gloves': 0.8,
        'goggles': 0.7,
        'labcoat': 0.8
    }

    # 객체별 바운딩 박스 색상 설정
    colors = {
        'gloves': (255, 0, 0),  # Red
        'goggles': (0, 255, 0),  # Green
        'labcoat': (0, 0, 255)  # Blue
    }

    # 웹캠 설정
    cap = cv2.VideoCapture(0)

    # 각 객체의 신뢰도 값과 탐지 시간을 저장하기 위한 딕셔너리
    confidence_values = {key: [] for key in confidence_thresholds.keys()}
    detection_start_time = {key: None for key in confidence_thresholds.keys()}
    continuous_detection = {key: False for key in confidence_thresholds.keys()}

    detection_completed = False

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        else:
            current_time = time.time()
            current_detections = []

            # YoloV5 모델에 이미지 전달 및 예측
            results = model1(frame)
            results_df = results.pandas().xyxy[0]

            for i, row in results_df.iterrows():
                obj_name = row['name']
                confidence = row['confidence']

                if obj_name in confidence_thresholds and confidence > confidence_thresholds[obj_name]:
                    if obj_name not in current_detections:
                        current_detections.append(obj_name)

                    if detection_start_time[obj_name] is None:
                        detection_start_time[obj_name] = current_time

                    confidence_values[obj_name].append(confidence)

                    # 바운딩 박스 및 신뢰도 그리기
                    x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
                    color = colors[obj_name]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"{obj_name} {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color,
                                2)

            # 3초 이상 연속 탐지된 객체의 평균 신뢰도 계산 및 출력
            for obj_name in confidence_thresholds.keys():
                if obj_name not in current_detections:
                    detection_start_time[obj_name] = None
                    confidence_values[obj_name] = []
                elif current_time - detection_start_time[obj_name] >= 3 and not continuous_detection[obj_name]:
                    continuous_detection[obj_name] = True
                    average_confidence = np.mean(confidence_values[obj_name])
                    print(f"{obj_name} 연속 3초 이상 탐지, 평균 신뢰도: {average_confidence:.2f}")
                    detection_status[obj_name] = True

            # 모든 지정된 객체들이 3초 이상 연속으로 탐지되었는지 확인
            if all(continuous_detection[obj_name] for obj_name in confidence_thresholds):
                detection_completed = True
                cv2.putText(frame, "Detection Success", (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                            cv2.LINE_AA)

            # '탐지 성공' 메시지가 표시된 후 3초 대기 후 종료
            if detection_completed:
                ch_switch1 = True
                cv2.waitKey(3000)  # 3초간 대기
                if ch_switch0 and ch_switch1:
                    print('문이 열렸습니다.')
                break

            ret2, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    # 자원 해제
    cap.release()
    cv2.destroyAllWindows()

@app.route('/chemical_feed1')
def chemical_feed1():
    return Response(gen_chemical1(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/check_detection_status')
def check_detection_status():
    global detection_status
    return jsonify(detection_status)

def gen_chemical0():
    global ch_switch0, ch_switch1
    global model
    model2 =model
    # model = torch.hub.load('ultralytics/yolov5', 'custom',
    #                        path='weights/best.pt', force_reload=True)

    # 각 객체별 신뢰도 임계값 설정
    confidence_thresholds = {
        'shoes': 0.75,
        'No-shoes': 0.85
    }

    # 객체별 바운딩 박스 색상 설정
    colors = {
        'shoes': (255, 255, 0),  # Yellow
        'No-shoes': (255, 105, 180)  # Pink
    }

    # 웹캠 설정
    cap = cv2.VideoCapture(1)

    # 각 객체의 신뢰도 값과 탐지 시간을 저장하기 위한 딕셔너리
    confidence_values = {key: [] for key in confidence_thresholds.keys()}
    detection_start_time = {key: None for key in confidence_thresholds.keys()}
    continuous_detection = {key: False for key in confidence_thresholds.keys()}

    detection_completed = False

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        else:
            current_time = time.time()
            current_detections = []

            # YoloV5 모델에 이미지 전달 및 예측
            results = model2(frame)
            results_df = results.pandas().xyxy[0]

            for i, row in results_df.iterrows():
                obj_name = row['name']
                confidence = row['confidence']

                if obj_name in confidence_thresholds and confidence > confidence_thresholds[obj_name]:
                    if obj_name not in current_detections:
                        current_detections.append(obj_name)

                    if detection_start_time[obj_name] is None:
                        detection_start_time[obj_name] = current_time

                    confidence_values[obj_name].append(confidence)

                    # 바운딩 박스 및 신뢰도 그리기
                    x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
                    color = colors[obj_name]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"{obj_name} {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color,
                                2)

            # 3초 이상 연속 탐지된 객체의 평균 신뢰도 계산 및 출력
            for obj_name in confidence_thresholds.keys():
                if obj_name not in current_detections:
                    detection_start_time[obj_name] = None
                    confidence_values[obj_name] = []
                elif current_time - detection_start_time[obj_name] >= 3 and not continuous_detection[obj_name]:
                    continuous_detection[obj_name] = True
                    average_confidence = np.mean(confidence_values[obj_name])
                    print(f"{obj_name} 연속 3초 이상 탐지, 평균 신뢰도: {average_confidence:.2f}")

            # 모든 지정된 객체들(No-shoes 제외)이 3초 이상 연속으로 탐지되었는지 확인
            if continuous_detection['shoes']:
                detection_completed = True
                cv2.putText(frame, "Detection Success", (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                            cv2.LINE_AA)

            # '탐지 성공' 메시지가 표시된 후 3초 대기 후 종료
            if detection_completed:
                ch_switch0 = True
                cv2.waitKey(3000)  # 3초간 대기
                if ch_switch1 and ch_switch0:
                    print('문이 열렸습니다.')
                break

            # 'q' 키를 누르면 루프 종료
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            ret2, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    # 자원 해제
    cap.release()
    cv2.destroyAllWindows()


@app.route('/chemical_feed0')
def chemical_feed0():
    return  Response(gen_chemical0(),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/check_chswitch')
def check_chswitch():
    global ch_switch1, ch_switch0
    print('check_chswitch:clothes', ch_switch1)
    print('check_chswitch:Shoes', ch_switch0)
    return jsonify({'ch_switch1': ch_switch1, 'ch_switch0':ch_switch0})

#기본 실험실 웹캠
# @app.route('/gen_basiccoat')
def gen_basicCoat():
    global b_switchCoat, b_switchShoes
    global bmodel

    bmodelCoat = bmodel

    # bmodelCoat.names[0] ='labcoat'

    # 각 객체별 신뢰도 임계값 설정
    confidence_thresholds = {
        'labcoat': 0.8
    }

    # 객체별 바운딩 박스 색상 설정
    # 색상이 다르게나와서 원래랑 같게 나오도록 수정
    colors = {
        'labcoat': (255, 0, 0)  # Blue
    }

    # 웹캠 설정
    cap = cv2.VideoCapture(0)

    # 각 객체의 신뢰도 값과 탐지 시간을 저장하기 위한 딕셔너리
    confidence_values = {key: [] for key in confidence_thresholds.keys()}
    detection_start_time = {key: None for key in confidence_thresholds.keys()}
    continuous_detection = {key: False for key in confidence_thresholds.keys()}

    detection_completed = False

    # with app.app_context():  # 애플리케이션 컨텍스트 설정
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        else:
            current_time = time.time()
            current_detections = []

            # YoloV5 모델에 이미지 전달 및 예측
            results = bmodelCoat(frame)
            results_df = results.pandas().xyxy[0]

            for i, row in results_df.iterrows():
                obj_name = row['name']
                confidence = row['confidence']

                if obj_name in confidence_thresholds and confidence > confidence_thresholds[obj_name]:
                    if obj_name not in current_detections:
                        current_detections.append(obj_name)

                    if detection_start_time[obj_name] is None:
                        detection_start_time[obj_name] = current_time

                    confidence_values[obj_name].append(confidence)

                    # 바운딩 박스 및 신뢰도 그리기
                    x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
                    color = colors[obj_name]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"{obj_name} {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color,
                                2)

            # 3초 이상 연속 탐지된 객체의 평균 신뢰도 계산 및 출력
            for obj_name in confidence_thresholds.keys():
                if obj_name not in current_detections:
                    detection_start_time[obj_name] = None
                    confidence_values[obj_name] = []
                elif current_time - detection_start_time[obj_name] >= 3 and not continuous_detection[obj_name]:
                    continuous_detection[obj_name] = True
                    average_confidence = np.mean(confidence_values[obj_name])
                    print(f"{obj_name} 연속 3초 이상 탐지, 평균 신뢰도: {average_confidence:.2f}")

            # 모든 지정된 객체들이 3초 이상 연속으로 탐지되었는지 확인
            if all(continuous_detection[obj_name] for obj_name in confidence_thresholds):
                detection_completed = True
                cv2.putText(frame, "Detection Success", (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                            cv2.LINE_AA)

            # '탐지 성공' 메시지가 표시된 후 3초 대기 후 종료
            if detection_completed:
                b_switchCoat = True
                cv2.waitKey(3000)  # 3초간 대기
                if b_switchCoat and b_switchShoes == True:
                    print("문이 열립니다.")
                break

            # 'q' 키를 누르면 루프 종료
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            ret2, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    # 자원 해제
    cap.release()
    cv2.destroyAllWindows()

@app.route('/basic_feed_coat')
def basic_feed_coat():
    return  Response(gen_basicCoat(), mimetype='multipart/x-mixed-replace; boundary=frame')

# 기본 실험실 신발
def gen_basicShoes():
    global b_switchShoes, b_switchCoat
    global bmodel
    bmodelShoes = bmodel

    # bmodelShoes.names[1] = 'shoes'
    # bmodelShoes.names[2] = 'No-shoes'

    # 각 객체별 신뢰도 임계값 설정
    # confidence_thresholds = {
    #     'shoes': 0.65,
    #     'No-shoes': 0.9
    # }
    #
    # # 객체별 바운딩 박스 색상 설정
    # # 색상이 다르게나와서 원래랑 같게 나오도록 수정
    # colors = {
    #     'shoes': (0, 255, 255),  # Yellow
    #     'No-shoes': (180, 105, 255)  # Pink
    # }

    confidence_thresholds = {
        'shoes': 0.7,
        'No-shoes': 0.8
    }

    # 객체별 바운딩 박스 색상 설정
    colors = {
        'shoes': (255, 255, 0),  # Yellow
        'No-shoes': (255, 105, 180)  # Pink
    }

    # 웹캠 설정
    cap = cv2.VideoCapture(1)

    # 각 객체의 신뢰도 값과 탐지 시간을 저장하기 위한 딕셔너리
    confidence_values = {key: [] for key in confidence_thresholds.keys()}
    detection_start_time = {key: None for key in confidence_thresholds.keys()}
    continuous_detection = {key: False for key in confidence_thresholds.keys()}

    detection_completed = False

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        else:
            current_time = time.time()
            current_detections = []

            # YoloV5 모델에 이미지 전달 및 예측
            results = bmodelShoes(frame)
            results_df = results.pandas().xyxy[0]

            for i, row in results_df.iterrows():
                obj_name = row['name']
                confidence = row['confidence']

                if obj_name in confidence_thresholds and confidence > confidence_thresholds[obj_name]:
                    if obj_name not in current_detections:
                        current_detections.append(obj_name)

                    if detection_start_time[obj_name] is None:
                        detection_start_time[obj_name] = current_time

                    confidence_values[obj_name].append(confidence)

                    # 바운딩 박스 및 신뢰도 그리기
                    x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
                    color = colors[obj_name]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"{obj_name} {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color,
                                2)

            # 3초 이상 연속 탐지된 객체의 평균 신뢰도 계산 및 출력
            for obj_name in confidence_thresholds.keys():
                if obj_name not in current_detections:
                    detection_start_time[obj_name] = None
                    confidence_values[obj_name] = []
                elif current_time - detection_start_time[obj_name] >= 3 and not continuous_detection[obj_name]:
                    continuous_detection[obj_name] = True
                    average_confidence = np.mean(confidence_values[obj_name])
                    print(f"{obj_name} 연속 3초 이상 탐지, 평균 신뢰도: {average_confidence:.2f}")

            # 모든 지정된 객체들(No-shoes 제외)이 3초 이상 연속으로 탐지되었는지 확인
            if continuous_detection['shoes']:
                detection_completed = True
                cv2.putText(frame, "Detection Success", (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                            cv2.LINE_AA)

            # '탐지 성공' 메시지가 표시된 후 3초 대기 후 종료
            if detection_completed:
                b_switchShoes = True
                cv2.waitKey(3000)  # 3초간 대기
                if b_switchCoat and b_switchShoes == True:
                    print("문이 열립니다.")
                break

            ret2, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    # 자원 해제
    cap.release()
    cv2.destroyAllWindows()

@app.route('/basic_feed_shoes')
def basic_feed_shoes():
    return  Response(gen_basicShoes(),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/check_bswitch')
def check_bswitch():
    global b_switchShoes, b_switchCoat
    print('check_bswitch:Coat', b_switchCoat)
    print('check_bswitch:Shoes', b_switchShoes)
    return jsonify({'b_switchCoat': b_switchCoat,'b_switchShoes': b_switchShoes})

@app.route('/hardware')
def hardware():
    if enable_switch and ((b_switchCoat and b_switchShoes) or (ch_switch1 and ch_switch0)):
        return redirect("http://192.168.170.117:5000")
        # return render_template('hardware.html')
    else:
        flash('잘못된 접근입니다.')
        return redirect(url_for('index'))
    # return redirect("http://192.168.170.117:5000")
    # return render_template('hardware.html')


if __name__ == '__main__':
    app.run(debug=True)
