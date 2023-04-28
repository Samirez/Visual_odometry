import cv2

frame1 = cv2.imread("output/images/DJI_0199_1200.jpg")
sift1 = cv2.SIFT_create()
keypoints1, descriptors1 = sift1.detectAndCompute(frame1, None)
img1 = cv2.drawKeypoints(frame1, keypoints1, frame1, cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

frame2 = cv2.imread("output/images/DJI_0199_1250.jpg")
sift2 = cv2.SIFT_create()
keypoints2, descriptors2 = sift2.detectAndCompute(frame2, None)
img2 = cv2.drawKeypoints(frame2, keypoints2, frame2, cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imshow("frame 1 ", img1)
cv2.waitKey()
cv2.imshow("frame2", img2)
cv2.waitKey()
