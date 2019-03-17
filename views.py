from django.shortcuts import render
from django.http import HttpResponse, HttpResponseRedirect
from .forms import UploadFileForm
from cv.models import GameMap
import sys
from skimage.filters import threshold_adaptive
from skimage.transform import pyramid_reduce
import numpy as np
#import argparse
import cv2
import json
import operator

def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype = "float32")

    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # return the ordered coordinates
    return rect

def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # return the warped image
    return warped

def translate(image, x, y):
    # Define the translation matrix and perform the translation
    M = np.float32([[1, 0, x], [0, 1, y]])
    shifted = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

    # Return the translated image
    return shifted

def rotate(image, angle, center = None, scale = 1.0):
    # Grab the dimensions of the image
    (h, w) = image.shape[:2]

    # If the center is None, initialize it as the center of
    # the image
    if center is None:
        center = (w / 2, h / 2)

    # Perform the rotation
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))

    # Return the rotated image
    return rotated

def resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized

def get_image(jimage):

    image = cv2.imread(jimage)
    ratio = image.shape[0] / 500.0
    orig_img = image.copy()
    #TODO: Check image orientation and rotate if neccessary with game_map = np.rot90(game_map, k=3)
    resized_img = resize(image, height=500)
    return (orig_img, resized_img, ratio)


def get_edges(image):
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_img = cv2.GaussianBlur(image, (5, 5), 0)
    edged_img = cv2.Canny(gray_img, 75, 200)
    return edged_img

def get_largest_contour(edged_img):
    # find the contours in the edged image, keeping only the largest ones, and initialize the screen contour
    (_, contours, _) = cv2.findContours(edged_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    # loop over the contours
    for countour in contours:
        # approximate the contour
        peri = cv2.arcLength(countour, True)
        approx = cv2.approxPolyDP(countour, 0.02 * peri, True)

        # if our approximated contour has four points, then we
        # can assume that we have found our screen
        if len(approx) == 4:
            screen_contour = approx
            break
    return screen_contour

def isolate_paper(orig_img, screen_contour, ratio):
    paper_img = four_point_transform(orig_img, screen_contour.reshape(4, 2) * ratio)
    return paper_img

def filter_colors(paper_img):
    resized_img = resize(paper_img, height=650)
    hsv_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2HSV)
    return (resized_img, hsv_img)

def get_blue_img(hsv):
    # define range of blue color in HSV
    lower_blue = np.array([100, 50, 50])
    upper_blue = np.array([140, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    blue_img = cv2.bitwise_and(hsv, hsv, mask=mask)
    return blue_img

def get_red_img(hsv):
    red = np.uint8([[[0, 0, 255]]])
    hsv_red = cv2.cvtColor(red, cv2.COLOR_BGR2HSV)  # (0,255,255)
    lower_red = np.array([0, 50, 50])  # also parses as red
    upper_red = np.array([25, 255, 255])  # also parses as red
    mask = cv2.inRange(hsv, lower_red, upper_red)
    red_img = cv2.bitwise_and(hsv, hsv, mask=mask)
    return red_img

def get_green_img(hsv):
    green = np.uint8([[[0, 255, 0]]])
    hsv_green = cv2.cvtColor(green, cv2.COLOR_BGR2HSV)
    hsv_green  # (60, 255, 255)
    lower_green = np.array([40, 30, 30])
    upper_green = np.array([85, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    green_img = cv2.bitwise_and(hsv, hsv, mask=mask)
    return green_img

def get_pink_img(hsv):
    lower_pink = np.array([140, 30, 30])
    upper_pink = np.array([175, 255, 255])
    mask = cv2.inRange(hsv, lower_pink, upper_pink)
    pink_img = cv2.bitwise_and(hsv, hsv, mask=mask)
    return pink_img

def get_black_img(hsv):
    # Black Image
    black_img = hsv.copy()
    nrows = hsv.shape[0]
    ncols = hsv.shape[1]
    for i in range(nrows):
        for j in range(ncols):
            if (hsv[i][j][2] < 125):
                black_img[i][j] = [0, 0, 0]  # set black
            else:
                black_img[i][j] = [0, 0, 255]  # set white
    return black_img

def img_to_game_map(paper_img, resized_image, green_img, black_img, red_img, blue_img, pink_img):
    # convert the paper_img image to grayscale, then threshold it
    paper_img = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    paper_img = threshold_adaptive(paper_img, 251, offset=10)
    paper_img = paper_img.astype("uint8") * 255
    nrows = paper_img.shape[0]
    ncols = paper_img.shape[1]
    game_map = np.zeros((nrows, ncols))
    for i in range(nrows):
        for j in range(ncols):
            if (paper_img[i][j] == 255):
                game_map[i][j] = 1 # background block
            else:
                if (green_img[i][j][2] > 3):
                    game_map[i][j] = 12 # points block
                elif (black_img[i][j][2] == 0):
                    game_map[i][j] = 2 # wall block
                elif (red_img[i][j][2] > 3):
                    game_map[i][j] = 9 # lava block
                elif (blue_img[i][j][2] > 3):
                    game_map[i][j] = 8 # endpoint block
                elif (pink_img[i][j][2] > 5):
                    game_map[i][j] = 5 # bouncy block
                else:
                    game_map[i][j] = 2 # if something there but none of the above colors default to wall block (black)
    return game_map


def get_rescaled_game_map(img, output_width, output_height): # (img, width=44, height=36) # 650 / 44 = 14.77, 498 / 36 = 13.8333
    result = np.zeros((output_height, output_width))
    nrows = img.shape[0] # 498
    ncols = img.shape[1] # 650
    row_size = nrows / output_height #498 / 36 = 13.8333
    col_size = ncols / output_width # 650 / 44 = 14.77
    for outter_i in range(0,output_height):
        for outter_j in range(0, output_width):
            votes = {1: 0, 12: 0, 9: 0, 8: 0, 5: 0, 2:0}
            for inner_i in range(int(outter_i* row_size), int(outter_i*row_size + row_size)):
                for inner_j in range(int(outter_j*col_size), int(outter_j*col_size + col_size)):
                    if (inner_i < nrows and inner_j < ncols):
                        color = img[inner_i][inner_j]
                        votes[color] += 1
                    best_color = max(iter(votes.items()), key=operator.itemgetter(1))[0]
                    if (best_color == 1 and (votes[1] < (0.7 *row_size * col_size))):
                        votes[1] = 0
                        best_color = max(iter(votes.items()), key=operator.itemgetter(1))[0]
            result[outter_i][outter_j] = best_color
    return np.rint(result).astype(int) #casts float into int

""" removes multiple instances of coins next to each other, finishing points near each other """
def eliminate_dups(img):
    nrows = img.shape[0]
    ncols = img.shape[1]

    def suppress_neighbhor(img, i, j, value):
        if i - 1 > 0:
            if (img[i - 1][j] == value):
                img[i - 1][j] = 1
        if i + 1 < nrows :
            if (img[i + 1][j] == value):
                img[i + 1][j] = 1
        if j - 1 > 0:
            if (img[i][j-1] == value):
                img[i][j-1] = 1
        if j + 1 < ncols:
            if (img[i][j + 1] == value):
                img[i][j + 1] = 1
        if (i - 1 > 0 and j - 1 > 0):
                if (img[i-1][j-1] == value):
                    img[i-1][j-1] = 1
        if (i - 1 > 0 and j + 1 < ncols):
                        if (img[i-1][j+1] == value):
                                img[i-1][j-1] = 1
        if (i+1 < nrows and j - 1 > 0):
                        if (img[i+1][j-1] == value):
                                img[i+1][j-1] = 1
        if (i + 1 < nrows) and (j + 1 < ncols):
                img[i+1][j+1] = 1
        
    for i in range(nrows):
        for j in range(ncols):
            if (img[i][j] == 12):
                suppress_neighbhor(img, i, j, 12)
                suppress_neighbhor(img, i, j, 2)
            elif (img[i][j] == 8):
                suppress_neighbhor(img, i, j, 8)
    
def add_border(img):
    nrows = img.shape[0]
    ncols = img.shape[1]
    for j in range(ncols):
        img[nrows-1][j] = 2
        img[0][j] = 2
    for i in range(nrows):
        img[i][0] = 2
        img[i][ncols-1] = 2

def game_map_to_string(img):
    game_map_text = json.dumps(img.tolist())
    return game_map_text

def test(game_map_text):
    assert game_map_text == '[[2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2], [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 5, 5, 5, 9, 9, 9, 9, 9, 9, 9, 9, 5, 5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2], [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 5, 5, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2], [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 1, 1, 1, 1, 1, 1, 1, 1, 2], [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 5, 9, 9, 9, 9, 9, 9, 9, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 9, 9, 9, 9, 9, 1, 1, 1, 1, 1, 1, 1, 2], [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 5, 9, 9, 9, 9, 9, 9, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 9, 9, 9, 1, 1, 1, 1, 1, 1, 2], [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 9, 9, 9, 9, 9, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 9, 9, 1, 1, 1, 1, 1, 1, 2], [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 9, 9, 9, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 5, 5, 5, 5, 1, 1, 1, 1, 9, 9, 1, 1, 1, 1, 1, 1, 2], [2, 1, 1, 1, 1, 1, 1, 1, 1, 9, 9, 5, 1, 1, 1, 5, 5, 5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 5, 5, 5, 5, 1, 1, 1, 1, 1, 9, 9, 1, 1, 1, 1, 1, 2], [2, 1, 1, 1, 1, 1, 1, 1, 9, 9, 5, 1, 1, 1, 5, 5, 5, 5, 5, 1, 1, 1, 1, 1, 1, 1, 5, 5, 12, 1, 5, 5, 1, 1, 1, 1, 9, 9, 1, 1, 1, 1, 1, 2], [2, 1, 1, 1, 1, 1, 1, 1, 9, 9, 5, 1, 1, 5, 5, 12, 1, 1, 5, 1, 1, 1, 1, 1, 1, 1, 5, 5, 1, 1, 5, 5, 1, 1, 1, 1, 9, 9, 1, 1, 1, 1, 1, 2], [2, 1, 1, 1, 1, 1, 1, 9, 9, 5, 1, 1, 1, 5, 5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 5, 9, 9, 1, 1, 1, 1, 2], [2, 1, 1, 1, 1, 1, 1, 9, 9, 5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 5, 9, 9, 1, 1, 1, 1, 2], [2, 1, 1, 1, 1, 1, 1, 9, 5, 5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 5, 9, 9, 1, 1, 1, 1, 2], [2, 1, 1, 1, 1, 1, 9, 9, 5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 5, 9, 9, 1, 1, 1, 1, 2], [2, 1, 1, 1, 1, 1, 9, 9, 5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 5, 9, 9, 1, 1, 1, 1, 2], [2, 1, 1, 1, 1, 1, 9, 9, 5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 5, 9, 9, 1, 1, 1, 1, 2], [2, 1, 1, 1, 1, 1, 9, 9, 5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 5, 1, 1, 1, 1, 1, 1, 2], [2, 1, 1, 1, 1, 1, 9, 9, 5, 5, 1, 1, 1, 1, 5, 5, 1, 1, 1, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 5, 5, 5, 1, 1, 5, 5, 1, 1, 1, 1, 1, 1, 2], [2, 1, 1, 1, 1, 1, 1, 1, 5, 5, 1, 1, 1, 1, 5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 5, 5, 1, 8, 1, 5, 5, 1, 1, 1, 1, 1, 1, 2], [2, 1, 1, 1, 1, 1, 1, 1, 5, 5, 1, 1, 1, 1, 5, 5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 5, 5, 1, 1, 1, 5, 5, 1, 1, 1, 1, 1, 1, 2], [2, 1, 1, 1, 1, 1, 1, 1, 5, 5, 1, 1, 1, 1, 5, 5, 5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 5, 5, 1, 1, 1, 1, 5, 5, 1, 1, 1, 1, 1, 1, 2], [2, 1, 1, 1, 1, 1, 1, 1, 1, 5, 5, 1, 1, 1, 1, 5, 5, 5, 5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 5, 5, 5, 1, 1, 1, 5, 5, 1, 1, 1, 1, 1, 1, 1, 2], [2, 1, 1, 1, 1, 1, 1, 1, 1, 5, 5, 1, 1, 1, 1, 1, 5, 5, 5, 5, 5, 1, 1, 1, 1, 1, 5, 5, 5, 5, 1, 1, 1, 1, 5, 5, 1, 1, 1, 1, 1, 1, 1, 2], [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 5, 5, 1, 1, 1, 1, 1, 1, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 1, 1, 1, 1, 1, 5, 5, 1, 1, 1, 1, 1, 1, 1, 1, 2], [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 5, 5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 5, 5, 5, 5, 1, 1, 1, 1, 1, 1, 1, 5, 5, 1, 1, 1, 1, 1, 1, 1, 1, 2], [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 5, 5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 5, 5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2], [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 5, 5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2], [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 9, 9, 9, 9, 9, 9, 1, 1, 1, 1, 1, 1, 1, 1, 1, 9, 9, 5, 5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2], [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2], [2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2], [2, 1, 1, 1, 1, 2, 2, 1, 1, 2, 1, 1, 1, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2], [2, 1, 1, 1, 1, 2, 2, 2, 2, 2, 1, 1, 1, 1, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2], [2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2], [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]]'

def scan_driver(pathtoimage):
    orig_img, image, ratio = get_image(pathtoimage) # from command line gets path of image to process and loads it into cv image object
    edged_img = get_edges(image) # greyscales image, then finds edges with Canny edge detector
    screen_countour = get_largest_contour(edged_img) # finds the largest contour in the image (aka finds largest shape which should be the piece of paper)
    paper_img = isolate_paper(orig_img, screen_countour, ratio) # using locations of largest contours, applies a 4-point-transform to remove background and return the image object
    resized_image, hsv = filter_colors(paper_img) # convert image to HSV scale for color-filtering
    
    # filters image down to 5 colors (white, black, green, blue, red)
    blue_img = get_blue_img(hsv)
    red_img = get_red_img(hsv)
    green_img = get_green_img(hsv)
    pink_img = get_pink_img(hsv)
    black_img = get_black_img(hsv)

    game_map = img_to_game_map(paper_img, resized_image, green_img, black_img, red_img, blue_img, pink_img) # makes 2d array representing map used by JS game engine
    rescaled_game_map = get_rescaled_game_map(game_map, output_width=44, output_height=36) # resizes it to desired end game map size
    eliminate_dups(rescaled_game_map) # removes multiple coins/exit points clustered nearby each other
    add_border(rescaled_game_map) # adds wall border around map
    game_map_text = game_map_to_string(rescaled_game_map) # gets string representation of the game map
    return game_map_text

#if __name__ == '__main__':
def output(file):
    game_map_text = scan_driver(file)
    return(game_map_text)
"""
def handle_uploaded_file(f):
    with open('input.jpg', 'wb+') as destination:
        for chunk in f.chunks():
            destination.write(chunk)
    import os
    os.system("python ../scan.py --image input.jpg > test_output.txt")
    os.system("chmod 777 test_output.txt")
"""
def upload_file(request):
    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            input_img = request.FILES['file']
            filename = input_img.name
            with open('input.jpg', 'wb+') as destination:
                for chunk in input_img.chunks():
                    destination.write(chunk)
            #handle_uploaded_file(input_img)
            map_title = request.POST['title']
            #with open('test_output.txt', 'r') as f:
            file_contents = output("input.jpg")
            new_game_map = GameMap.objects.create(title=map_title,map=file_contents,input_img=input_img)
            new_game_map.save()
            return (HttpResponseRedirect("/play/" + map_title))
    else:
        form = UploadFileForm()
    return render(request, 'index.html', {'form': form})

def game(request, map_name):
    game_obj = GameMap.objects.filter(title=map_name)
    if game_obj:
        game_obj = game_obj[0]
        high_score_string = game_obj.high_score
        if (game_obj.high_score == 9999): # defualt value
            high_score_string = "Level never beaten!"
        return render(request, 'play.html', {'map': game_obj, 'high_score': high_score_string})
    else:
        return HttpResponse("<h1> Game Page Does Not Exist With Name: </h1>" + map_name)

def update_score(request, map_name):
    if request.method == 'POST':
        high_score = int(request.POST.get("score"))
        username = request.POST.get("username")
        game_obj = GameMap.objects.filter(title=map_name)[0]
        print >> sys.stderr, (high_score, username)
        if (game_obj.high_score > high_score):
            game_obj.high_score = high_score
            if username == "":
                username = "Anonymous"
            game_obj.high_score_name = username
            game_obj.save()
        return HttpResponse("Succesfully Updated High Score")
    else:
        return HttpResponse("Requires POST request to update scores")

def vote(request):
    for key, value in request.POST.items():
        print >>sys.stderr, (key, value)

    if request.method == 'POST':
        map_id = request.POST.get("id")
        is_upvote = request.POST.get("up")
        is_downvote = request.POST.get("down")
        prev = request.POST.get("prev") #previously clicked button {none,downvoted,upvoted}

        game_obj = GameMap.objects.filter(id=map_id)[0]
        if prev == "none":
            if is_upvote == "true":
                game_obj.votes = game_obj.votes + 1
            else:
                game_obj.votes = game_obj.votes - 1
        elif prev == "downvoted":
            if is_upvote == "true":
                game_obj.votes = game_obj.votes + 2
            else:
                game_obj.votes = game_obj.votes + 1
        elif prev == "upvoted":
            if is_downvote == "true":
                game_obj.votes = game_obj.votes - 2
            else:
                game_obj.votes = game_obj.votes - 1
        else:
            print >> sys.stderr, "Logical Error, prev not matched to any of the cases"

        game_obj.save()
        return HttpResponse("Updated Votes")

    else:
        return HttpResponse("Requires POST request to update votes")

def discover_home(request):
    return discover(request, 1)

def discover(request, page):
    page = int(page)
    start_index = page*10-10
    end_index = page*10
    next_page = page + 1
    recent_maps = GameMap.objects.order_by('-votes')[start_index:end_index]
    return render(request, 'discover.html', {'recent_maps': recent_maps, 'next_page': next_page})

