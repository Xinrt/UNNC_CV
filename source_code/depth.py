# This file is used for generating depth map and calculate the distance in real world
# including the implemented BM algorithm

import cv2
import time
import numpy
import math
import camera_config


# preprocess to reduce the light effect
def preprocess(img1, img2):
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # histogram equilibrium
    img1 = cv2.equalizeHist(img1)
    img2 = cv2.equalizeHist(img2)

    return img1, img2


# implement the sobel filter
def sobel_filter(image):
    height, width = image.shape
    out_image = numpy.zeros((height, width))

    filter_x = numpy.array(([-1, -2, -1], [0, 0, 0], [1, 2, 1]))
    filter_y = numpy.array(([-1, 0, 1], [-2, 0, 2], [-1, 0, 1]))

    for y in range(2, width - 2):
        for x in range(2, height - 2):
            cx, cy = 0, 0
            for offset_y in range(0, 3):
                for offset_x in range(0, 3):
                    pix = image[x + offset_x -
                                1, y + offset_y - 1]
                    if offset_x != 1:
                        cx += pix * filter_x[offset_x, offset_y]
                    if offset_y != 1:
                        cy += pix * filter_y[offset_x, offset_y]
            out_pix = math.sqrt(cx ** 2 + cy ** 2)
            out_image[x, y] = out_pix if out_pix > 0 else 0
    numpy.putmask(out_image, out_image > 255, 255)
    return out_image


# Calculate left disparity
def calc_left_disparity(gray_left, gray_right, num_disparity, block_size):
    height, width = gray_right.shape
    disparity_matrix = numpy.zeros((height, width), dtype=numpy.float32)
    half_block = block_size // 2

    for i in range(half_block, height - half_block):
        print("%d%% " % (i * 100 // height), end=' ', flush=True)

        for j in range(half_block, width - half_block):
            left_block = gray_left[i - half_block:i +
                                                  half_block, j - half_block:j + half_block]
            diff_sum = 32767
            disp = 0

            for d in range(0, min(j - half_block - 1, num_disparity)):
                right_block = gray_right[i - half_block:i +
                                                        half_block, j - half_block - d:j + half_block - d]
                sad_val = sum(sum(abs(right_block - left_block)))

                if sad_val < diff_sum:
                    diff_sum = sad_val
                    disp = d

            disparity_matrix[i - half_block, j - half_block] = disp
    print('100%')
    return disparity_matrix


# Calculate right disparity
def calc_right_disparity(gray_left, gray_right, num_disparity, block_size):
    height, width = gray_right.shape
    disparity_matrix = numpy.zeros((height, width), dtype=numpy.float32)
    half_block = block_size // 2

    for i in range(half_block, height - half_block):
        print("%d%% " % (i * 100 // height), end=' ', flush=True)

        for j in range(half_block, width - half_block):
            right_block = gray_right[i - half_block:i +
                                                    half_block, j - half_block:j + half_block]
            diff_sum = 32767
            disp = 0

            for d in range(0, min(width - j - half_block, num_disparity)):

                left_block = gray_left[i - half_block:i +
                                                      half_block, j - half_block + d:j + half_block + d]
                sad_val = sum(sum(abs(right_block - left_block)))

                if sad_val < diff_sum:
                    diff_sum = sad_val
                    disp = d

            disparity_matrix[i - half_block, j - half_block] = disp
    print('100%')
    return disparity_matrix


# left and right verification
def left_right_check(disparity_left, disparity_right):
    height, width = disparity_left.shape
    out_image = disparity_left

    for h in range(1, height - 1):
        for w in range(1, width - 1):
            left = int(disparity_left[h, w])
            if w - left > 0:
                right = int(disparity_right[h, w - left])
                dispDiff = left - right
                if dispDiff < 0:
                    dispDiff = -dispDiff
                elif dispDiff > 1:
                    out_image[h, w] = 0
    return out_image


# implement the Block Matching algorithm
def implement_BM(left_image, right_image, num_disparity, block_size):
    start_time = time.time()

    # get filtered images
    sobel_left = sobel_filter(left_image)
    sobel_right = sobel_filter(right_image)

    # Calculate left disparity
    disparity_left = calc_left_disparity(
        sobel_left, sobel_right, num_disparity, block_size)
    disparity_left_color = cv2.applyColorMap(cv2.convertScaleAbs(
        disparity_left, alpha=256 / num_disparity), cv2.COLORMAP_JET)

    # Calculate right disparity
    disparity_right = calc_right_disparity(
        sobel_left, sobel_right, num_disparity, block_size)
    disparity_right_color = cv2.applyColorMap(cv2.convertScaleAbs(
        disparity_right, alpha=256 / num_disparity), cv2.COLORMAP_JET)

    # Post-processing
    disparity = left_right_check(disparity_left, disparity_right)
    print('Duration: %s seconds\n' % (time.time() - start_time))

    # Generate color image and save to file
    disparity_color = cv2.applyColorMap(cv2.convertScaleAbs(
        disparity, alpha=256 / num_disparity), cv2.COLORMAP_JET)

    disparity_gray = cv2.cvtColor(disparity_color, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('./images/depth_result/depth.jpg', disparity_gray)

    return disparity_gray, disparity_color


# rectify the input image by using the parameters obtained from the camera calibration
def rectify_image(image):
    config = camera_config.Camera()
    num_horizontal, num_vertical = image.shape[:2]

    # Internal parameter matrix
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(config.intrinsic_matrix, config.distortion,
                                                           (num_vertical, num_horizontal), 0,
                                                           (num_vertical, num_horizontal))
    # correction transformation calculation
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(config.intrinsic_matrix, config.distortion,
                                                      config.intrinsic_matrix, config.distortion,
                                                      (num_vertical, num_horizontal), config.R, config.T, alpha=0)

    # corrected image
    result = cv2.undistort(image, config.intrinsic_matrix, config.distortion, None, new_camera_matrix)

    return result, Q


# the main function: obtain the depth map and shown in a new window
def depth_calculation(flag):
    if int(flag) == 1:
        iml = cv2.imread('./images/output_correction/left1.jpg')  # left image
        imr = cv2.imread('./images/output_correction/right1.jpg')  # right image
        num_disparity = 50
        block_size = 29
    elif int(flag) == 2:
        iml = cv2.imread('./images/output_correction/left2.jpg')  # left image
        imr = cv2.imread('./images/output_correction/right2.jpg')  # right image
        num_disparity = 64
        block_size = 27
    elif int(flag) == 3:
        iml = cv2.imread('./images/output_correction/left3.jpg')  # left image
        imr = cv2.imread('./images/output_correction/right3.jpg')  # right image
        num_disparity = 100
        block_size = 27
    else:
        print("Invalid input, please try again.")

    iml, Q = rectify_image(iml)
    imr, _ = rectify_image(imr)

    # reduce the effects of the uneven light
    iml, imr = preprocess(iml, imr)

    # calculate the depth map by BM algorithm
    dis_gray, dis_color = implement_BM(iml, imr, num_disparity, block_size)

    # calculate the 3D position of the points
    points_3d = cv2.reprojectImageTo3D(dis_gray, Q)

    # calculate and print the distance in real world
    def mouse_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print('point (%d, %d) coordinate: (%f, %f, %f)' % (
                x, y, points_3d[y, x, 0], points_3d[y, x, 1], points_3d[y, x, 2]))
            dis = ((points_3d[y, x, 0] ** 2 + points_3d[y, x, 1] ** 2 + points_3d[y, x, 2] ** 2) ** 0.5) / 100
            print('distance from the point (%d, %d) to the left camera: %0.3f m' % (x, y, dis))

    cv2.namedWindow("Stereo Vision", 0)
    cv2.imshow("Stereo Vision", dis_gray)
    cv2.setMouseCallback("Stereo Vision", mouse_click, 0)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
