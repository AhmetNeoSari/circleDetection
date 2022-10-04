import cv2
import numpy as np

cap = cv2.VideoCapture(0)
'''
    Docstring = webcam'den aldığı görüntüde kırmızıyı filtreleyen program
    Input: webcam görüntü
    Output: kırmızıyı algılayan görüntü
'''
while 1:
    ret,frame = cap.read(0) # frame'lere baktık
    frame = cv2.flip(frame,1) # y ye göre yansıma aldık
    '''
    opencv ile renk tespitinde renk üzerine mask uygulanır.
    mask'ı uygulamak için önce renk aralığı seçilir
    ve frame'de bu renk aralığı içinde kalanlar korunur
    gerisi atılır bgr renk aralığı kullanılmaz hsv kullanılır
    '''
    # hsv => h: renk , saturation : solukluğu,yoğunluğu, value: parlaklık

    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_red = np.array([170,100,20])
    upper_red = np.array([180,255,255])
    red_mask = cv2.inRange(hsv_frame, lower_red, upper_red)
    # kırmızı şeyler kırmızı görünsün gerisi siyah görünsün:
    # mask değeri ile frame' leri karşılaştırır
    red = cv2.bitwise_and(frame, frame, mask = red_mask) # 2 kere aynı frame i gönerip maskemizle karşılaştırıp and liyoruz

    # dikdörtgen  içine alma
    hsv_frame = cv2.cvtColor(hsv_frame, cv2.COLOR_HSV2BGR)
    contours, hierarchy = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # print(contours)
    # print(hierarchy)
    if len(contours) != 0 :
        for contour in contours:
            if cv2.contourArea(contour) >500:
                x,y,w,h = cv2.boundingRect(contour)
                cv2.rectangle(hsv_frame, (x,y), (x+w, y+h,),(0,0,255),1)

    cv2.imshow("Red Mask",hsv_frame)
    # cv2.imshow("webcam",frame) #deneme
    # cv2.imshow("redMask2",red_mask) # deneme
    # cv2.imshow("Red",red) # deneme
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()




