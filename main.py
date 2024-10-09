import cv2
import numpy as np
import mediapipe as mp


# Set camera image resolution
cam_width, cam_height = 1024, 768


def initialize_camera(cam_width, cam_height):
    cap = cv2.VideoCapture(0)
    cap.set(3, cam_width)
    cap.set(4, cam_height)
    return cap

def air_draw(cap):
    mphands = mp.solutions.hands  # 使用medidapipe裡的手部辨識功能
    hands = mphands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)  # 設定手部辨識模型
    mpDraw = mp.solutions.drawing_utils  # 繪畫工具
    # 手部骨架樣式設定
    handLms_style = mpDraw.DrawingSpec(color=(73, 93, 70), thickness=6)  # 手座標樣式
    handCon_style = mpDraw.DrawingSpec(color=(255, 250, 240), thickness=3)  # 手連接樣式
    trail_image = np.zeros((cam_height, cam_width, 3), np.uint8)  # 記錄軌跡並創建一個黑色背景
    # 軌跡顏色 (BGR)  https://www.rapidtables.com/web/color/RGB_Color.html
    trail_color_1 = (255, 255, 0)   # Cyan
    trail_color_2 = (0, 0, 255)     # Red
    trail_color_3 = (0, 255, 255)   # Yellow
    trail_color_4 = (0, 255, 0)     # Green
    trail_color_5 = (255, 0, 255)   # Pink
    trail_color_6 = (255, 0, 0)     # Blue
    trail_color_7 = (0, 128, 255)   # Orange
    trail_color_8 = (255, 0, 127)   # Purple
    trail_color_9 = (128, 128, 128) # Grey
    trail_color_0 = (255, 255, 255) # White
    # 預設值
    selected_color = trail_color_0  # 軌跡顏色
    drawing = False                 # 關閉繪圖模式
    show_skeleton = True            # 顯示骨架
    blank_mode = False              # 顯示攝影機畫面
    while True:
        key = cv2.waitKey(1)  # 等待1毫秒
        # 按空白鍵開啟/關閉繪畫模式
        if key == 32:  
            drawing = not drawing
            show_skeleton = not show_skeleton
            print("Drawing mode is", "ON" if drawing else "OFF")
        # 按B鍵關閉攝影機畫面
        if key == ord("b"):
            blank_mode = not blank_mode
            print("Camera is", "OFF" if blank_mode else "ON")             
        # 按C鍵清除畫布
        if key == ord("c"):  
            trail_image = np.zeros((cam_height, cam_width, 3), np.uint8)
            print("Canvas cleared")
        # 按S鍵儲存畫布
        if key == ord("s"):  
            cv2.imwrite("trail_image.png", trail_image)
            print("Canvas saved")
        # 按Q鍵離開程式
        if key == ord("q"):
            break
        # 按數字鍵選顏色
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
            print("Orange")
            selected_color = trail_color_7
        elif key == ord("8"):
            print("Purple")
            selected_color = trail_color_8
        elif key == ord("9"):
            print("Grey")
            selected_color = trail_color_9
        elif key == ord("0"):
            print("White (default)")
            selected_color = trail_color_0
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
                        xPos = int(lm.x * img_width)   # 比例乘上寬度
                        yPos = int(lm.y * img_height)  # 比例乘上高度
                        # 記錄拇指與食指座標
                        if i == 8:
                            x8, y8 = xPos, yPos
                            if ("drawing" in locals() and drawing):
                                cv2.circle(trail_image, (xPos, yPos), 9, selected_color, cv2.FILLED)                             
            # 確保trail_image與img大小一致
            trail_image = cv2.resize(trail_image, (img.shape[1], img.shape[0]))
            # 疊加影像
            img = cv2.addWeighted(img, 1, trail_image, 1, 0)
        if not blank_mode:
            cv2.imshow("AirDraw", img)
        else:
            cv2.imshow("AirDraw", trail_image)   
    cap.release()
    cv2.destroyAllWindows()  
    
def face_detect(cap, file="haarcascade_frontalface_default.xml"):
    detector = cv2.CascadeClassifier(file)
    while True:
        ret, img = cap.read()
        img = cv2.flip(img, 1)  # 翻轉圖像         
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)   # 將鏡頭影像轉換成灰階
        try:
            faces = detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=4)  # 偵測人臉
        except cv2.error:
            print("找不到XML模型文件，請下載\nhttps://raw.githubusercontent.com/opencv/opencv/4.x/data/haarcascades/haarcascade_frontalface_default.xml\n")
            break
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)   # 使用綠框標記人臉
        cv2.imshow('Face detection', img)
        if cv2.waitKey(1) == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows() 

while True:            
    option = input("[1] 空氣繪圖   [2] 臉部偵測   [Q] 離開\n")    
    if option == "1":
        cap = initialize_camera(cam_width, cam_height)
        air_draw(cap)
    elif option == "2":
        cap = initialize_camera(cam_width, cam_height)
        face_detect(cap)
    elif option.upper() == "Q":
        break
    else:
        continue

