```python
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from jamo import h2j, j2hcj, j2h
from gtts import gTTS
import subprocess
import os

# MediaPipe 손 인식 모델 초기화
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

# 한글 자모음 정의
gesture_dict = {
    0: 'ㄱ', 1: 'ㄴ', 2: 'ㄷ', 3: 'ㄹ', 4: 'ㅁ', 5: 'ㅂ', 6: 'ㅅ', 7: 'ㅇ', 8: 'ㅈ', 9: 'ㅎ',
    10: 'ㅏ', 11: 'ㅓ', 12: 'ㅕ', 13: 'ㅗ', 14: 'ㅜ', 15: 'ㅡ', 16: 'ㅣ', 17: 'ㅐ', 18: 'ㅔ', 19: 'space', 20: 'clear', 21: 'next'
}

# Gesture recognition data
data_file = "/home/wogud/gesture_train.csv"  # 실제 경로로 변경
file = pd.read_csv(data_file)
anglefile = file.iloc[:, :-1].values
labelfile = file.iloc[:, -1].values
angle = anglefile.astype(np.float32)
label = labelfile.astype(np.float32)
knn = cv2.ml.KNearest_create()
knn.train(angle, cv2.ml.ROW_SAMPLE, label)

# 인식된 제스처를 저장하기 위한 변수 초기화
last_recognized_gesture = None
gesture_counter = 0
gesture_threshold = 10  # 동일한 제스처가 연속적으로 나타날 최소 프레임 수

# 실시간으로 쌓일 문자를 저장하는 변수
full_text = ""
# 현재 입력된 자모음을 저장하는 변수
recognized_characters = []

# 'ㄱ'을 세 번 인식하는 것을 추적하기 위한 변수
g_count = 0

def recognize_hand_sign(image, gesture_dict):
    global last_recognized_gesture, gesture_counter

    img = cv2.flip(image, 1)
    h, w, c = img.shape
    results = hands.process(img)

    if results.multi_hand_landmarks:
        for hand_landmark in results.multi_hand_landmarks:
            joint = np.zeros((21, 3))
            for j, lm in enumerate(hand_landmark.landmark):
                joint[j] = [lm.x, lm.y, lm.z]

            # 관절 사이 벡터 계산
            v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19], :]
            v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], :]
            v = v2 - v1  # [20,3]
            # 정규화
            v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

            # 각도 계산
            angle = np.arccos(np.einsum('nt,nt->n',
                                        v[[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18], :],
                                        v[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], :]))  # [15,]

            angle = np.degrees(angle)  # Convert radian to degree

            # Inference gesture
            data = np.array([angle], dtype=np.float32)
            ret, results, neighbours, dist = knn.findNearest(data, 3)
            idx = int(results[0][0])

            if idx in gesture_dict:
                if last_recognized_gesture == gesture_dict[idx]:
                    gesture_counter += 1
                else:
                    gesture_counter = 0

                last_recognized_gesture = gesture_dict[idx]

                if gesture_counter >= gesture_threshold:
                    gesture_counter = 0
                    return gesture_dict[idx], img

            mp_draw.draw_landmarks(img, hand_landmark,
                                   mp_hands.HAND_CONNECTIONS,
                                   mp_draw.DrawingSpec((0, 0, 255), 6, 3),
                                   mp_draw.DrawingSpec((0, 255, 0), 4, 2)
                                   )
    return None, img

# 자모음을 결합하여 한글 음절로 변환하는 함수
def combine_jamos(jamo_list):
    if len(jamo_list) == 2:
        return j2h(jamo_list[0], jamo_list[1])
    elif len(jamo_list) == 3:
        return j2h(jamo_list[0], jamo_list[1], jamo_list[2])
    else:
        return ''.join(jamo_list)

# TTS 기능을 수행하는 함수
def speak_text(text):
    tts = gTTS(text=text, lang='ko')
    tts.save("output.mp3")
    subprocess.run(["mpg123", "output.mp3"])  # mpg123를 사용해 재생
    os.remove("output.mp3")  # 재생 후 파일 삭제

# 실시간 웹캠 처리
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 손 제스처 인식
    recognized_gesture, processed_img = recognize_hand_sign(frame, gesture_dict)

    # 인식된 제스처 출력 및 한글 자모음 처리
    if recognized_gesture:
        print(f"Recognized Gesture: {recognized_gesture}")
        if recognized_gesture == 'ㄱ':
            g_count += 1
            if g_count == 3:
                g_count = 0  # count 초기화
                print("TTS: ", full_text)
                speak_text(full_text)  # TTS로 텍스트 읽기
        else:
            g_count = 0  # 다른 제스처가 나오면 count 초기화

        if recognized_gesture == 'space':
            recognized_characters.append(' ')
        elif recognized_gesture == 'clear':
            if full_text:
                full_text = full_text[:-1]  # 마지막 글자 삭제
        elif recognized_gesture == 'next':
            if recognized_characters:
                # 자모음을 결합하여 한글로 변환
                combined = combine_jamos(recognized_characters)
                full_text += combined  # 최종 텍스트에 추가
                print(f"Combined: {full_text}")
                recognized_characters = []  # 현재 문자 초기화
            else:
                print(f"Combined: {full_text}")
        else:
            recognized_characters.append(recognized_gesture)

        # 전체 텍스트를 화면에 출력
        cv2.putText(processed_img, full_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

    # 결과 영상 출력
    cv2.imshow('Hand Sign Detection', processed_img)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# 인식된 문자열 출력
print(f"Recognized Characters: {full_text}")
```
