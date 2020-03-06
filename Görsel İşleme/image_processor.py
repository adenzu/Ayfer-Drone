import cv2
import numpy as np
import pyautogui

cap = cv2.VideoCapture(0) #cv2 ile kamera erişimini sağladım.

lower_blue = np.array([94, 80, 75]) #Mavi renginin HSV alt limiti
upper_blue = np.array([126, 255, 255]) #Mavi renginin HSV üst limiti

while True:
	_, frame = cap.read() #Kameradan frame çekme
	
	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) #frame'in HSV hali
	mask = cv2.inRange(hsv, lower_blue, upper_blue) #Sadece limit arası rengindeki kısımları al
	blurred_mask = cv2.medianBlur(mask, 15) #mask'i bulanıklaştır

	#mask'te beyaz kontürleri bulma
	ret,thresh = cv2.threshold(blurred_mask,127,255,0) 
	contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	
	res = cv2.bitwise_and(frame, frame, mask=blurred_mask) #frame'in sadece mask kısmı

	for c in contours:

		if cv2.contourArea(c) >= 150: #Eğer kontürün alanı 150'den fazlaysa
			M = cv2.moments(c)

			if M["m00"] != 0:
				cX = int(M["m10"] / M["m00"]) #Kontürün merkezinin x değeri
				cY = int(M["m01"] / M["m00"]) #Kontürün merkezinin y değeri
			else:
				cX, cY = 0, 0

			cv2.drawContours(frame, [c], 0, (0,255,0), 3) #Kontürü çizme
			cv2.circle(frame, (cX, cY), 5, (0, 0, 255), -1) #Kontür merkezine nokta çizme
	 

	cv2.imshow("frame", frame)
	cv2.imshow("res", res)
	
	k = cv2.waitKey(5) & 0xFF
	if k == 27:
		break

cv2.destroyAllWindows()
