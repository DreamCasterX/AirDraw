import cv2
import numpy as np
import mediapipe as mp


# 攝影機設定
cam_width, cam_height = 640, 480
cap = cv2.VideoCapture(0)  # 填入0或1試試看
cap.set(30, cam_width)  # 調整影像寬度
cap.set(40, cam_height)  # 調整影像長度

# 畫框設定
rec_width_1 = int(cam_width * 0.1)
rec_width_2 = int(cam_width * 0.9)
rec_height_1 = int(cam_height * 0.1)
rec_height_2 = int(cam_height * 0.9)

# 使用medidapipe裡的手部辨識功能
mphands = mp.solutions.hands  

# 設定手部辨識模型
hands = mphands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) 

# 繪畫工具 
mpDraw = mp.solutions.drawing_utils  

# 手部骨架樣式設定
handLms_style = mpDraw.DrawingSpec(color=(73, 93, 70), thickness=6)  # 手座標樣式
handCon_style = mpDraw.DrawingSpec(color=(255, 250, 240), thickness=3)  # 手連接樣式

# 記錄軌跡
trail_image = np.zeros((cam_height, cam_width, 3), np.uint8)  # 創建一個黑色背景

# 軌跡顏色 (BGR)
trail_color_1 = (255, 255, 0)  # Cyan
trail_color_2 = (0, 0, 255)    # Red
trail_color_3 = (0, 255, 255)  # Yellow
trail_color_4 = (102, 255, 0)  # Green
trail_color_5 = (255, 0, 255)  # Pink
trail_color_6 = (255, 0, 0)    # Blue
trail_color_7 = (0, 57, 77)    # Brown

# 預設軌跡顏色
selected_color = trail_color_1

# 迴圈區域
drawing = False  # 預設繪圖模式關閉
show_skeleton = True  # 預設顯示骨架
while True:
    key = cv2.waitKey(1)  # 等待1毫秒
    # 按空白鍵開啟/關閉繪圖
    if key == 32:  
        drawing = not drawing
        show_skeleton = not show_skeleton
        if drawing:
            print("Drawing mode is ON")
        else:
            print("Drawing mode is OFF")
    # 按C鍵清除畫布 (重設畫布為黑色)
    if key == ord("c"):  
        trail_image = np.zeros((cam_height, cam_width, 3), np.uint8)
        print("Canvas cleared")
    # 按S鍵儲存畫布
    if key == ord("s"):  
        cv2.imwrite("trail_image.png", trail_image)
        print("繪圖已存檔")
    # 按Q鍵離開程式
    if key == ord("q"):
        break
    # 選顏色
    if key == ord("1"):
        print("Cyan")
        selected_color = trail_color_1
    elif key == ord("2"):
        print("Red")
        selected_color = trail_color_2
    elif key == ord("3"):
        print("Yellow")
        selected_color = trail_color_3
    elif key == ord("4"):
        print("Green")
        selected_color = trail_color_4
    elif key == ord("5"):
        print("Pink")
        selected_color = trail_color_5
    elif key == ord("6"):
        print("Blue")
        selected_color = trail_color_6
    elif key == ord("7"):
        print("Brown")
        selected_color = trail_color_7
    ret, img = cap.read()

    # hand_detection
    if ret:
        img = cv2.flip(img, 1)  # 翻轉圖像
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR轉RGB
        result = hands.process(img_rgb)

        img_height = img.shape[0]
        img_width = img.shape[1]

        if result.multi_hand_landmarks:
            for handLms in result.multi_hand_landmarks:
                if show_skeleton:
                    # 顯示每個手的座標及連線、設定樣式
                    mpDraw.draw_landmarks(img, handLms, mphands.HAND_CONNECTIONS, handLms_style, handCon_style)
                for i, lm in enumerate(handLms.landmark):  # 輸出每個手的座標點
                    xPos = int(lm.x * img_width)  # 比例乘上寬度
                    yPos = int(lm.y * img_height)  # 比例乘上高度

                    # 記錄拇指與食指座標
                    if i == 8:
                        x8, y8 = xPos, yPos
                        if (
                            "drawing" in locals() and drawing
                        ):  # 如果啟動繪畫模式
                            cv2.circle(
                                trail_image, (xPos, yPos), 9, selected_color, cv2.FILLED
                            )
        img = cv2.addWeighted(img, 1, trail_image, 1, 0)

    cv2.imshow("img", img)
