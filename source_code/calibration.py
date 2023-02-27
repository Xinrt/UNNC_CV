# Statement at the beginning of the file
# -*- coding:utf-8 -*-

# This file is used for camera calibration
# the result of camera calibration will be stored in:
# distortion factor.txt, external matrix.txt, intrinsic matrix

import cv2
import numpy as np
import glob

if __name__ == '__main__':
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    length = 1.8  # length of the small square in the checkerboard grid
    num_vertical = 6
    num_horizontal = 13

    point_realWorld = np.zeros((num_vertical * num_horizontal, 3),
                               np.float32)  # coordinates in the real World Coordinate System
    point_realWorld[:, :2] = np.mgrid[0:num_vertical * length:length, 0:num_horizontal * length:length].T.reshape(-1, 2)

    points_realWorld = []  # points in the real world
    points_image = []  # points in the image

    j = 1
    images = glob.glob('./images/cali/*.jpg')  # read all .jpg images
    for file_name in images:
        img = cv2.imread(file_name)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find corner points
        ret, corners = cv2.findChessboardCorners(gray, (num_vertical, num_horizontal), None)

        if ret is True:
            cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            # add to the real world coordinates system
            points_realWorld.append(point_realWorld)
            # add to the image coordinates system
            points_image.append(corners)
            # show and store the result images
            cv2.drawChessboardCorners(img, (num_vertical, num_horizontal), corners, ret)
            cv2.imshow('findCorners', img)

            cv2.imwrite('./images/cali_result/cali result %d.jpg' % j, img)
            cv2.waitKey(1)
            print(str(j) + "success\n")
            j += 1
        else:
            print('error')
            j += 1
    cv2.destroyAllWindows()

    # find the intrinsic and extrinsic parameters of the camera
    ret, intrinsic_matrix, distortion_factor, rotation_vector, translation_vector = cv2.calibrateCamera(
        points_realWorld,
        points_image,
        gray.shape[::-1],
        None, None)

    f = open('./files/intrinsic matrix.txt', 'w+')
    f.write('intrinsic_matrix:\n' + str(intrinsic_matrix) + '\n')
    f.close()
    f = open('./files/distortion factor.txt', 'w+')
    f.write('distortion factor:\n' + str(distortion_factor) + '\n')
    f.close()

    f = open('./files/external matrix.txt', 'w+')
    # obtain the translation vectors and rotation vectors
    for t in range(0, j - 1):
        new_rotation_vector = cv2.Rodrigues(rotation_vector[t].ravel())
        new_translation_vector = cv2.Rodrigues(translation_vector[t].ravel())
        f.write('external parameter matrix of the ' + str(t + 1) + ' image:\n' + '(1)rotation matrix: \n' + str(
            new_rotation_vector[0]) + '\n' + '(2)Translation matrix: \n' + str(new_translation_vector[0]) + '\n')
        t += 1
    f.close()
