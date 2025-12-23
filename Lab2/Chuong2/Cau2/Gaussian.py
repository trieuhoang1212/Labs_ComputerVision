import cv2


img = cv2.imread(r"C:\Users\PC\Downloads\images.jpg")

if img is not None:
 
    blurred = cv2.GaussianBlur(img, (21, 21), 0) 

    cv2.imshow('original ', img)
    cv2.imshow('Gaussian Blur', blurred)
    cv2.waitKey(0)
    cv2.destroyAllWindows()