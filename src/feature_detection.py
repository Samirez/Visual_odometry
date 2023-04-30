import cv2
import numpy as np

frame1 = cv2.imread("../output/images/DJI_0199_1200.jpg")
sift1 = cv2.SIFT_create()
keypoints1, descriptors1 = sift1.detectAndCompute(frame1, None)
img1 = cv2.drawKeypoints(frame1, keypoints1, frame1, cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

frame2 = cv2.imread("../output/images/DJI_0199_1250.jpg")
sift2 = cv2.SIFT_create()
keypoints2, descriptors2 = sift2.detectAndCompute(frame2, None)
img2 = cv2.drawKeypoints(frame2, keypoints2, frame2, cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

'''cv2.imshow("frame 1 ", img1)
#cv2.waitKey()
#cv2.imshow("frame2", img2)
cv2.waitKey()'''

matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_FLANNBASED)
knbmatches = matcher.knnMatch(descriptors1, descriptors2, 2)
ratio_thresh = 0.3
good_matches = []
for m, n in knbmatches:
    if m.distance < ratio_thresh * n.distance:
        good_matches.append(m)

outimg = cv2.drawMatches(frame1, keypoints1, frame2, keypoints2, good_matches, None,
                         flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
cv2.imwrite("../output/images/matches.jpg", outimg)

CM = np.array([[2676.1051390718389, -35.243952918157035, -279.58562078697361],
               [0.0097935857180804498, -0.021794052829051412, 0.017776502734846815],
               [0.0046443590741258711, -0.0045664024579022498, 1.]])
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

matches = bf.match(descriptors1, descriptors2)

points1_temp = []
points2_temp = []
match_indices_temp = []
for idx, m in enumerate(matches):
    points1_temp.append(keypoints1[m.queryIdx].pt)
    points2_temp.append(keypoints2[m.trainIdx].pt)
    match_indices_temp.append(idx)

points1 = np.float32(points1_temp)
points2 = np.float32(points2_temp)
match_indices = np.int32(match_indices_temp)
ransacReprojecThreshold = 1
confidence = 0.99

essentialMatrix, mask = cv2.findEssentialMat(
    points1,
    points2,
    CM,
    cv2.FM_RANSAC,
    confidence,
    ransacReprojecThreshold)

img3 = cv2.drawMatches(frame1, keypoints1, frame2, keypoints2, matches, None,
                       flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
cv2.imwrite("../output/images/essentialmatrix_matching.jpg", img3)

match_indices = match_indices[mask.ravel() == 1]
filtered_matches = []
for idx in match_indices:
    m = matches[idx]
    filtered_matches.append(matches[idx])

img3 = cv2.drawMatches(frame1, keypoints1, frame2, keypoints2, filtered_matches, None,
                       flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
cv2.imwrite("../output/images/essentialmatrix_matching_filtered.jpg", img3)

frame1 = cv2.imread("../output/images/DJI_0199_1200.jpg")
sift1 = cv2.SIFT_create()
keypoints1, descriptors1 = sift1.detectAndCompute(frame1, None)

frame2 = cv2.imread("../output/images/DJI_0199_1250.jpg")
keypoints2, descriptors2 = sift1.detectAndCompute(frame2, None)

matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_FLANNBASED)
knbmatches = matcher.knnMatch(descriptors1, descriptors2, 2)

ratio_thresh = 0.3
good_matches = []
for m, n in knbmatches:
    if m.distance < ratio_thresh * n.distance:
        good_matches.append(m)

CM = np.array([[2676.1051390718389, -35.243952918157035, -279.58562078697361],
               [0.0097935857180804498, -0.021794052829051412, 0.017776502734846815],
               [0.0046443590741258711, -0.0045664024579022498, 1.]])

ransacReprojecThreshold = 1
confidence = 0.99
essentialMatrix, mask = cv2.findEssentialMat(
    keypoints1,
    keypoints2,
    CM,
    cv2.FM_RANSAC,
    confidence,
    ransacReprojecThreshold)

inlier_matches = [good_matches[i] for i in match_indices]
match_indices = np.where(mask.ravel() == 1)[0]

epipolar_lines1 = cv2.computeCorrespondEpilines(
    [k.pt for k in keypoints1],
    1,
    essentialMatrix)
epipolar_lines2 = cv2.computeCorrespondEpilines(
    [k.pt for k in keypoints2],
    2,
    essentialMatrix)

distance = []
for match in inlier_matches:
    points1 = keypoints1[match.queryIdx].pt
    points2 = keypoints2[match.trainIdx].pt
    line1 = epipolar_lines1[match.queryIdx][0]
    line2 = epipolar_lines2[match.trainIdx][0]
    dist1 = abs(line1.dot(np.array([points2[0], points2[1], 1])))
    dist2 = abs(line2.dot(np.array([points1[0], points1[1], 1])))
    distance.append(dist1 + dist2)

mean_distance = np.mean(distance)
std_distance = np.std(distance)
print("mean distance: ", mean_distance)
print("std distance: ", std_distance)
