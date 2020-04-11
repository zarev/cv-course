import numpy as np
import cv2
import matplotlib.pyplot as plt
import helpers as hlp

chess = hlp.imread('resources/flat_chessboard.png')

found, corners = cv2.findChessboardCorners(chess, (7,7))

cv2.drawChessboardCorners(chess, (7,7), corners, found)

hlp.implot(chess)