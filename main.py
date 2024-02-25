import cv2  # opencv 影像處理的套件
import numpy as np  # 矩陣運算
import mediapipe as mp  # 偵測模型


##攝影機設定##
cam_width, cam_height = 640, 480
cap = cv2.VideoCapture(0)  # 填入0或1試試看
cap.set(30, cam_width)  # 調整影像寬度
cap.set(40, cam_height)  # 調整影像長度

## 畫框設定
rec_width_1 = int(cam_width * 0.1)
rec_width_2 = int(cam_width * 0.9)

rec_height_1 = int(cam_height * 0.1)
rec_height_2 = int(cam_height * 0.9)

##手部模型偵測參數設定、功能引入##
mphands = mp.solutions.hands  # 使用medidapipe裡的手部辨識功能
hands = mphands.Hands(
    min_detection_confidence=0.5, min_tracking_confidence=0.5
)  # 設定手部辨識模型
mpDraw = mp.solutions.drawing_utils  # 繪畫工具

##手部骨架樣式設定##
handLms_style = mpDraw.DrawingSpec(color=(73, 93, 70), thickness=6)  # 手座標樣式
handCon_style = mpDraw.DrawingSpec(color=(255, 250, 240), thickness=3)  # 手連接樣式

# 記錄軌跡
trail_image = np.zeros((cam_height, cam_width, 3), np.uint8)  # 創建一個黑色背景

# 軌跡顏色
trail_color_1 = (105, 165, 218)
trail_color_2 = (255, 250, 240)
trail_color_3 = (73, 93, 70)
trail_color_4 = (0, 69, 255)


##迴圈區域##
drawing = False  # 預設繪圖模式關閉
while True:
    key = cv2.waitKey(1)  # 等待1毫秒
    if key == ord("s"):  # 按 's' 鍵儲存畫布
        cv2.imwrite("trail_image.png", trail_image)  # 將畫布儲存為 trail_image.png
        print("繪圖已存檔")

    if key == 32:
        drawing = not drawing
        if drawing:
            print("開啟繪圖")
        else:
            print("關閉繪圖")
    if key == ord("c"):  # 按 'c' 鍵清除畫布
        trail_image = np.zeros((cam_height, cam_width, 3), np.uint8)  # 重設畫布為黑色
        print("軌跡已清除")

    if key == ord("q"):
        break

    # 選顏色
    trail_color = trail_color_1
    if key == ord("1"):
        trail_color = trail_color_1
    if key == ord("2"):
        trail_color = trail_color_2
    if key == ord("3"):
        trail_color = trail_color_3
    if key == ord("4"):
        trail_color = trail_color_4

    ret, img = cap.read()

    # hand_detection
    if ret:
        img = cv2.flip(img, 1)  # 翻轉圖像

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # bgr轉rgb
        result = hands.process(img_rgb)

        img_height = img.shape[0]
        img_width = img.shape[1]

        if result.multi_hand_landmarks:
            for (
                handLms
            ) in result.multi_hand_landmarks:  # 顯示每個手的座標及連線、設定樣式
                mpDraw.draw_landmarks(
                    img, handLms, mphands.HAND_CONNECTIONS, handLms_style, handCon_style
                )

                for i, lm in enumerate(handLms.landmark):  # 輸出每個手的座標點
                    xPos = int(lm.x * img_width)  # 比例乘上寬度
                    yPos = int(lm.y * img_height)  # 比例乘上高度

                    # 記錄拇指與食指座標
                    if i == 8:
                        x8, y8 = xPos, yPos

                        # 偵測食指尖，是否在畫布範圍內
                        if (
                            rec_width_1 + 9 < x8 < rec_width_2 - 9
                            and rec_height_1 + 9 < y8 < rec_height_2 - 9
                        ):
                            in_canvas = True
                        else:
                            in_canvas = False

                        if (
                            "drawing" in locals() and drawing and in_canvas
                        ):  # 如果啟動繪畫模式
                            cv2.circle(
                                trail_image, (xPos, yPos), 9, trail_color, cv2.FILLED
                            )

        img = cv2.addWeighted(img, 1, trail_image, 1, 0)

    # 顯示可繪圖區域外框
    cv2.rectangle(
        img,
        (rec_width_1, rec_height_1),
        (rec_width_2, rec_height_2),
        (105, 165, 218),
        3,
    )

    cv2.imshow("img", img)
