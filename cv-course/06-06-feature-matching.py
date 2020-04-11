import numpy as np
import cv2
import matplotlib.pyplot as plt
import helpers

reeses = helpers.imread('data/reeses_puffs.png', 0)

cereals = cv2.imread('data/many_cereals.jpg', 0)

# helpers.implot(reeses)
# helpers.implot(cereals)

# brute force with orb
# orb = cv2.ORB_create()
# kp1, des1 = orb.detectAndCompute(reeses, None)
# kp2, des2 = orb.detectAndCompute(cereals, None)

# bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
# matches = bf.match(des1, des1)
# matches = sorted(matches, key=lambda x:x.distance)
# rs_matches = cv2.drawMatches(reeses, kp1, cereals, kp2, matches[:25], None, flags=2)
# helpers.implot(rs_matches)

# SIFT for different scales
# sift = cv2.xfeatures2d.SIFT_create()
# kp1, des1 = sift.detectAndCompute(reeses, None)
# kp2, des2 = sift.detectAndCompute(cereals, None)

# bf = cv2.BFMatcher()
# matches = bf.knnMatch(des1, des2, k=2)

# good = []
# for match1,match2 in matches:
#   if match1.distance < 0.75*match2.distance: good.append([match1])

# sf_matches = cv2.drawMatchesKnn(reeses, kp1, cereals, kp2, good, None, flags=2)

# helpers.implot(sf_matches)


# FLANN based, not best but good looking matches
sift = cv2.xfeatures2d.SIFT_create()
kp1, des1 = sift.detectAndCompute(reeses, None)
kp2, des2 = sift.detectAndCompute(cereals, None)

FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, 2)
mask = [[0,0] for i in range(len(matches))]

for i, (match1,match2) in enumerate(matches):
  if match1.distance < 0.7*match2.distance: mask[i] = [1,0]

draw_params = dict(matchColor=(0,255,0),
                   singlePointColor=(255,0,0),
                   matchesMask=mask,
                   flags=0)

fl_matches = cv2.drawMatchesKnn(reeses, kp1, cereals, kp2, matches, None, **draw_params)
helpers.implot(fl_matches)
