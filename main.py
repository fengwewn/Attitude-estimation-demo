import cv2
import mediapipe as mp
import numpy as np
import time
from ffpyplayer.player import MediaPlayer
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
a_time =0
# 加载模型
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
canvas_width, canvas_height = 1280, 720
# 打开摄像头
cap = cv2.VideoCapture(0)

# 加载参考图片
#reference_image = cv2.imread('reference_image.jpg')
ref_cap = cv2.VideoCapture('reference_video.mp4')
player = MediaPlayer('reference_video.mp4') # create a MediaPlayer object to play the audio of the video file
fps = ref_cap.get(cv2.CAP_PROP_FPS) # get the frame rate of the video
sleep_ms = int(np.round((1/fps)*1000)) # calculate the millisecond value for cv2.waitKey
total_frames = int(ref_cap.get(cv2.CAP_PROP_FRAME_COUNT)) # 获取视频文件的总帧数

while True:


    while True:
        # 加载参考图像
        # reference_image = cv2.imread("reference_image.jpg", cv2.IMREAD_UNCHANGED)
        ret, ref_image = ref_cap.read()
        audio_frame, val = player.get_frame()


        def get_video_timestamp(frame_number):
            # 一个函数用于获取视频帧的时间戳（秒），根据总帧数和帧率计算出每一帧对应的秒数
            return frame_number / total_frames * (total_frames / fps)
        if not ret:
            ref_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        if val != 'eof' and audio_frame is not None:  # if audio frame is valid, store the data and timestamp
            img, a_time = audio_frame
        # 处理参考图像
        reference_results = pose.process(ref_image)
        #reference_results = pose.process(cv2.cvtColor(ref_image, cv2.COLOR_BGR2RGB))
        # 提取参考图像的关键点
        if reference_results.pose_landmarks:
            mp_drawing.draw_landmarks(ref_image, reference_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            reference_landmarks = np.array([(lm.x, lm.y) for lm in reference_results.pose_landmarks.landmark],
                                           dtype=np.float32)
        else:
            reference_landmarks = None
        # 读取帧
        success, image = cap.read()
        image = cv2.flip(image, 1)

        if not success:
            print("无法读取视频流.")
            break

        # 将图像从BGR转换为RGB
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 处理图像
        results = pose.process(image)

        # 绘制姿势估计结果
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # 如果有参考图像，则计算参考图像和当前帧的差异
            if reference_landmarks is not None:
                current_landmarks = np.array([(lm.x, lm.y) for lm in results.pose_landmarks.landmark], dtype=np.float32)
                diff = np.abs(reference_landmarks - current_landmarks)
                diff = np.mean(diff, axis=0)

                # 根据差异判断姿势是否正确
                if diff[0] < 0.05 and diff[1] < 0.05:
                    cv2.putText(image, "Correct posture", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                else:
                    cv2.putText(image, "Incorrect posture", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # 显示结果
        v_time = get_video_timestamp(ref_cap.get(cv2.CAP_PROP_POS_FRAMES))  # 获取画面对应的时间戳（秒）
        time_diff = abs(v_time - a_time)  # 计算画面和声音之间时间戳（秒）的绝对差值
        #if time_diff > 0.05:  # 如果差值大于设定的阈值（0.05秒），暂停一段时间来同步它们
            #time.sleep(time_diff)
        if reference_landmarks is not None:
            #将参考图像和相机图像拼接在一起

            dst1 = cv2.resize(ref_image, None, fx=0.3, fy=0.3, interpolation=cv2.INTER_LINEAR)
            cv2.imshow('reference_image', dst1)

            dst = cv2.resize(image, None, fx=0.3, fy=0.3, interpolation=cv2.INTER_LINEAR)

            cv2.imshow('image', dst)




        if cv2.waitKey(1) == ord('q'):
            break

# 释放资源
cap.release()
cv2.destroyAllWindows()
video.release()