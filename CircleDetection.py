
import cv2 as cv
import numpy as np


videoCapture = cv.VideoCapture(0)

prevCircle = None 
dist = lambda x1,y1,x2,y2: (x1-x2)**2+(y1-y2)**2

while True:
	ret,frame = videoCapture.read()
	frame = cv.flip(frame,1) #görüntüyü y eksenine göre yansımasını aldık

	if not ret: break

	# gürültüleri atmak için 2 adımda grileştirip bulanıklaştıralım
	grayFrame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
	blurFrame = cv.GaussianBlur(grayFrame,(17,17),0)

	# çember bulma fonksiyonumuz
	circles = cv.HoughCircles(blurFrame, cv.HOUGH_GRADIENT,1.2,400,param1=150,param2=30,minRadius=1,maxRadius=400)
	"""
	parametrelerin anlamları: 
	 1. parametre: kaynak görüntü
	 2. parametre: cv.HOUGH_GRADİENT(değişim)
	 3. parametre: dp => eğer büyük olursa yakın bulduğumuz çemberler birleşecek, birleşme olasılıkları daha yüksek olacak
	 bu yüzden çember konumunun pek doğru olmamasını ama daha hesaplı kararlar vermesini sağlayacak
	 genel olarak 1-2 arasında 1,2 1,4 gibi değerler alır
	 4. parametre: mindistance: minimum mesafe bulunan iki olası daire arasındaki minimum mesafeyi ifade eder eğer;
	 kendimizde sadece tek bir daire bulmak istiyorsak, o zaman 100 gibi büyük değerler vermeliyiz
	 5.parametre param1: hassasiyeti ifade eder yüksek tutarsak fazla daire bulamaz düşük tutarsak çok fazla daire bulur
	 6.parametre param2: daire algılamanın doğruluğudur. Çalışma şekli bir daire olduğunu bildirmek için gereken kenar 
	 noktalarnın sayısının belirlenmesidir. Eğer yüksek sayı verirsek yeterince daire bulamaz
	 7.parametre minRadius: minimum yarıçap, algılanabilen dairenin minimum boyutudur
	 8. parametre maxRadius: 
	 not: bu fonksiyon daki parametreler bulduğu çemberleri bize bir liste olarak geri döndürür ve daire olarak depolar
	 bu listeden geçmemiz lazım bu temelde numpyd dizilerine dönüştürmemiz gerek
	"""

	if circles is not None:
		circles = np.uint16(np.around(circles)) # bunlar bizim sahip olduğumuz daireler ve biz en iyi olasılıklı daireleri istiyoruz
		chosen = None
		for i in circles[0, :]:

			if chosen is None : chosen = i
			if prevCircle is not None:
				if dist(chosen[0],chosen[1],prevCircle[0],prevCircle[1]) <= dist(i[0],i[1],prevCircle[0],prevCircle[1]):
					# daha yakın bir daire bulursak, merkez noktasına daha yakın bir daire bulursak, bunu yeni merkezimiz olarak ayarlayacağız
					chosen = i
		#şimdi etrafına çember çizelim:
		cv.circle(frame,(chosen[0],chosen[1]),1,(0,100,100),3)
		cv.circle(frame,(chosen[0],chosen[1]),chosen[2],(0,100,100),3)
		"""
		 parametreler:
		 1.parametre: kaynak görüntü blurdan etkilenmeyen
		 2.parametre: merkez noktası
		 3.parametre: yarıçap
		 4.parametre: renk
		 5.parametre: çemberin kalınlığı
		"""
		# önceki daireyi mevcut daireye eşitleyelim:
		prevCircle = chosen
	

	cv.imshow("circles",frame)
	if cv.waitKey(1) & 0xFF == ord("q"): break

videoCapture.release()
cv.destroyAllWindows()
