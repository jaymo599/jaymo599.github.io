{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyN5VkDD+eZTPVKQ+BCU3sAZ",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/jaymo599/jaymo599.github.io/blob/master/Untitled11.md\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "-brdajcW2DC1",
        "outputId": "e54d3cc2-ef0c-4bc7-a9fc-97a7cbecd128"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Mounted at /content/drive\n"
          ]
        }
      ],
      "source": [
        "from google.colab import drive\n",
        "drive.mount('/content/drive')"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "pip install mediapipe"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "m-wH92t-2fxn",
        "outputId": "03ca9aaf-f4b4-46ff-9fe2-52c0e6467412"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Collecting mediapipe\n",
            "  Downloading mediapipe-0.10.14-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (35.7 MB)\n",
            "\u001b[2K     \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m35.7/35.7 MB\u001b[0m \u001b[31m24.9 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hRequirement already satisfied: absl-py in /usr/local/lib/python3.10/dist-packages (from mediapipe) (1.4.0)\n",
            "Requirement already satisfied: attrs>=19.1.0 in /usr/local/lib/python3.10/dist-packages (from mediapipe) (23.2.0)\n",
            "Requirement already satisfied: flatbuffers>=2.0 in /usr/local/lib/python3.10/dist-packages (from mediapipe) (24.3.25)\n",
            "Requirement already satisfied: jax in /usr/local/lib/python3.10/dist-packages (from mediapipe) (0.4.26)\n",
            "Requirement already satisfied: jaxlib in /usr/local/lib/python3.10/dist-packages (from mediapipe) (0.4.26+cuda12.cudnn89)\n",
            "Requirement already satisfied: matplotlib in /usr/local/lib/python3.10/dist-packages (from mediapipe) (3.7.1)\n",
            "Requirement already satisfied: numpy in /usr/local/lib/python3.10/dist-packages (from mediapipe) (1.25.2)\n",
            "Requirement already satisfied: opencv-contrib-python in /usr/local/lib/python3.10/dist-packages (from mediapipe) (4.8.0.76)\n",
            "Collecting protobuf<5,>=4.25.3 (from mediapipe)\n",
            "  Downloading protobuf-4.25.3-cp37-abi3-manylinux2014_x86_64.whl (294 kB)\n",
            "\u001b[2K     \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m294.6/294.6 kB\u001b[0m \u001b[31m26.9 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hCollecting sounddevice>=0.4.4 (from mediapipe)\n",
            "  Downloading sounddevice-0.4.7-py3-none-any.whl (32 kB)\n",
            "Requirement already satisfied: CFFI>=1.0 in /usr/local/lib/python3.10/dist-packages (from sounddevice>=0.4.4->mediapipe) (1.16.0)\n",
            "Requirement already satisfied: ml-dtypes>=0.2.0 in /usr/local/lib/python3.10/dist-packages (from jax->mediapipe) (0.2.0)\n",
            "Requirement already satisfied: opt-einsum in /usr/local/lib/python3.10/dist-packages (from jax->mediapipe) (3.3.0)\n",
            "Requirement already satisfied: scipy>=1.9 in /usr/local/lib/python3.10/dist-packages (from jax->mediapipe) (1.11.4)\n",
            "Requirement already satisfied: contourpy>=1.0.1 in /usr/local/lib/python3.10/dist-packages (from matplotlib->mediapipe) (1.2.1)\n",
            "Requirement already satisfied: cycler>=0.10 in /usr/local/lib/python3.10/dist-packages (from matplotlib->mediapipe) (0.12.1)\n",
            "Requirement already satisfied: fonttools>=4.22.0 in /usr/local/lib/python3.10/dist-packages (from matplotlib->mediapipe) (4.53.1)\n",
            "Requirement already satisfied: kiwisolver>=1.0.1 in /usr/local/lib/python3.10/dist-packages (from matplotlib->mediapipe) (1.4.5)\n",
            "Requirement already satisfied: packaging>=20.0 in /usr/local/lib/python3.10/dist-packages (from matplotlib->mediapipe) (24.1)\n",
            "Requirement already satisfied: pillow>=6.2.0 in /usr/local/lib/python3.10/dist-packages (from matplotlib->mediapipe) (9.4.0)\n",
            "Requirement already satisfied: pyparsing>=2.3.1 in /usr/local/lib/python3.10/dist-packages (from matplotlib->mediapipe) (3.1.2)\n",
            "Requirement already satisfied: python-dateutil>=2.7 in /usr/local/lib/python3.10/dist-packages (from matplotlib->mediapipe) (2.8.2)\n",
            "Requirement already satisfied: pycparser in /usr/local/lib/python3.10/dist-packages (from CFFI>=1.0->sounddevice>=0.4.4->mediapipe) (2.22)\n",
            "Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.10/dist-packages (from python-dateutil>=2.7->matplotlib->mediapipe) (1.16.0)\n",
            "Installing collected packages: protobuf, sounddevice, mediapipe\n",
            "  Attempting uninstall: protobuf\n",
            "    Found existing installation: protobuf 3.20.3\n",
            "    Uninstalling protobuf-3.20.3:\n",
            "      Successfully uninstalled protobuf-3.20.3\n",
            "\u001b[31mERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.\n",
            "tensorflow-metadata 1.15.0 requires protobuf<4.21,>=3.20.3; python_version < \"3.11\", but you have protobuf 4.25.3 which is incompatible.\u001b[0m\u001b[31m\n",
            "\u001b[0mSuccessfully installed mediapipe-0.10.14 protobuf-4.25.3 sounddevice-0.4.7\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "pip install jamo"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "nC1nqX_a5ALY",
        "outputId": "811955a2-1590-40a3-a568-e7f0006b9037"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Collecting jamo\n",
            "  Downloading jamo-0.4.1-py3-none-any.whl (9.5 kB)\n",
            "Installing collected packages: jamo\n",
            "Successfully installed jamo-0.4.1\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import cv2\n",
        "import mediapipe as mp\n",
        "import numpy as np\n",
        "import pandas as pd\n",
        "from jamo import h2j, j2hcj\n",
        "\n",
        "# MediaPipe 손 인식 모델 초기화\n",
        "mp_hands = mp.solutions.hands\n",
        "hands = mp_hands.Hands()\n",
        "mp_draw = mp.solutions.drawing_utils\n",
        "\n",
        "# 한글 자모음 정의\n",
        "gesture_dict = {\n",
        "    0: 'ㄱ', 1: 'ㄴ', 2: 'ㄷ', 3: 'ㄹ', 4: 'ㅁ', 5: 'ㅂ', 6: 'ㅅ', 7: 'ㅇ', 8: 'ㅈ', 9: 'ㅎ',\n",
        "    10: 'ㅏ', 11: 'ㅓ', 12: 'ㅕ', 13: 'ㅗ', 14: 'ㅜ', 15: 'ㅡ', 16: 'ㅣ', 17: 'ㅐ', 18: 'ㅔ', 19: 'space', 20: 'clear', 21: 'next'\n",
        "}\n",
        "\n",
        "# Gesture recognition data\n",
        "data_file = r\"path\\to\\your\\gesture_train.csv\"  # Use raw string for path\n",
        "file = pd.read_csv(data_file)\n",
        "anglefile = file.iloc[:, :-1].values\n",
        "labelfile = file.iloc[:, -1].values\n",
        "angle = anglefile.astype(np.float32)\n",
        "label = labelfile.astype(np.float32)\n",
        "knn = cv2.ml.KNearest_create()\n",
        "knn.train(angle, cv2.ml.ROW_SAMPLE, label)\n",
        "\n",
        "def recognize_hand_sign(image, gesture_dict):\n",
        "    img = cv2.flip(image, 1)\n",
        "    h, w, c = img.shape\n",
        "    results = hands.process(img)\n",
        "\n",
        "    if results.multi_hand_landmarks:\n",
        "        for hand_landmark in results.multi_hand_landmarks:\n",
        "            joint = np.zeros((21, 3))\n",
        "            for j, lm in enumerate(hand_landmark.landmark):\n",
        "                joint[j] = [lm.x, lm.y, lm.z]\n",
        "\n",
        "            # 관절 사이 벡터 계산\n",
        "            v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19], :]\n",
        "            v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], :]\n",
        "            v = v2 - v1  # [20,3]\n",
        "            # 정규화\n",
        "            v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]\n",
        "\n",
        "            # 각도 계산\n",
        "            angle = np.arccos(np.einsum('nt,nt->n',\n",
        "                                        v[[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18], :],\n",
        "                                        v[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], :]))  # [15,]\n",
        "\n",
        "            angle = np.degrees(angle)  # Convert radian to degree\n",
        "\n",
        "            # Inference gesture\n",
        "            data = np.array([angle], dtype=np.float32)\n",
        "            ret, results, neighbours, dist = knn.findNearest(data, 3)\n",
        "            idx = int(results[0][0])\n",
        "\n",
        "            if idx in gesture_dict:\n",
        "                return gesture_dict[idx], img\n",
        "\n",
        "            mp_draw.draw_landmarks(img, hand_landmark,\n",
        "                                   mp_hands.HAND_CONNECTIONS,\n",
        "                                   mp_draw.DrawingSpec((0, 0, 255), 6, 3),\n",
        "                                   mp_draw.DrawingSpec((0, 255, 0), 4, 2)\n",
        "                                   )\n",
        "    return None, img\n",
        "\n",
        "# 실시간 웹캠 처리\n",
        "cap = cv2.VideoCapture(0)\n",
        "\n",
        "recognized_characters = []\n",
        "\n",
        "while cap.isOpened():\n",
        "    ret, frame = cap.read()\n",
        "    if not ret:\n",
        "        break\n",
        "\n",
        "    # 손 제스처 인식\n",
        "    recognized_gesture, processed_img = recognize_hand_sign(frame, gesture_dict)\n",
        "\n",
        "    # 인식된 제스처 출력 및 한글 자모음 처리\n",
        "    if recognized_gesture:\n",
        "        print(f'Recognized Gesture: {recognized_gesture}')\n",
        "        if recognized_gesture == 'space':\n",
        "            recognized_characters.append(' ')\n",
        "        elif recognized_gesture == 'clear':\n",
        "            recognized_characters = []\n",
        "        elif recognized_gesture == 'next':\n",
        "            if recognized_characters:\n",
        "                combined = ''.join(recognized_characters)\n",
        "                print(f'Combined: {combined}')\n",
        "                recognized_characters = []\n",
        "        else:\n",
        "            recognized_characters.append(recognized_gesture)\n",
        "\n",
        "        cv2.putText(processed_img, recognized_gesture, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)\n",
        "\n",
        "    # 결과 영상 출력\n",
        "    cv2.imshow('Hand Sign Detection', processed_img)\n",
        "\n",
        "    # 'q' 키를 누르면 종료\n",
        "    if cv2.waitKey(1) & 0xFF == ord('q'):\n",
        "        break\n",
        "\n",
        "cap.release()\n",
        "cv2.destroyAllWindows()\n",
        "\n",
        "# 인식된 문자열 출력\n",
        "print(f'Recognized Characters: {\"\".join(recognized_characters)}')\n"
      ],
      "metadata": {
        "id": "sJdJd0RTNbV0"
      },
      "execution_count": null,
      "outputs": []
    }
  ]
}