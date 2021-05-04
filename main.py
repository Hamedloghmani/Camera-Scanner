import cv2
import numpy as np
import matplotlib.pyplot as plt


def wrapped_up_canny(image: np.ndarray) -> np.ndarray:
    """Wrapped up Canny with Gaussian blur function
    used builtin method in order to get more accurate answers

    Args:
        image(np.ndarray): input image

    Returns:
        np.ndarray : image after Canny edge detection
    """
    return cv2.Canny(cv2.GaussianBlur(image, (3, 3), 0), 90, 100, apertureSize=3)


def line_equation(p1: tuple, p2: tuple) -> tuple:
    """Calculation of line equation based on two given points on the line
    Args:
        p1: coordination number 1 as a tuple
        p2: coordination number 2 as a tuple

    Returns:
        tuple : includes gradient of a line and interception
    """
    a, b = 0, 0
    if p1[0] - p2[0] == 0:
        a = (p1[1] - p2[1]) / 0.00000001
    elif a * p1[0] == p1[1]:
        a = (p1[1] - p2[1]) / (p1[0] - p2[0])
        b = 0
    else:
        a = (p1[1] - p2[1]) / (p1[0] - p2[0])
        b = p1[1] - (p1[0] * a)

    return a, b


def line_intersection(line1: tuple, line2: tuple) -> tuple:
    """Finding the intersection of two given lines based on the line equations
    Args:
        line1(tuple): equation of first line
        line2(tuple): equation of second line

    Returns:
        tuple: coordination of the intersection
    """
    try:
        x = int((line2[1] - line1[1]) / (line1[0] - line2[0]))

    except ZeroDivisionError:
        x = int((line2[1] - line1[1]))

    y = line1[0] * x + line1[1]
    return tuple(map(int, (x, y)))


def hough_lines_normalize(canny_image: np.ndarray, hough_threshold: int = 150) -> list:
    """finding and normalizing Houghlines , method and a block of code is extracted from opencv docs
    Args:
        canny_image(np.ndarray): image after Canny transform
        hough_threshold: threshold for Houghlines function (different for each picture)

    Returns:

    """
    lines = cv2.HoughLines(canny_image, 1, np.pi / 180, hough_threshold)
    inter = list()

    for x in lines:
        for rho, theta in x:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * a)
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * a)
            inter.append(line_equation((x1, y1), (x2, y2)))
    return inter


def hough_intersections(image: np.ndarray, hough_threshold: int = 150) -> list:
    """finding the intersection of hough lines , using line_intersection() function
    Args:
        image(np.ndarray): input image
        hough_threshold(int): a threshold for hoghLines_normalize() function

    Returns:
        list: list of intersection points
    """
    intersection = hough_lines_normalize(wrapped_up_canny(image), hough_threshold)
    points = list()
    for i in range(0, len(intersection)):
        for j in range(0, len(intersection)):
            if i != j:
                points.append(line_intersection(intersection[i], intersection[j]))
    for i in points:
        if i[0] < 0 or i[1] < 0:
            points.remove(i)

    return points


def harris_corner_detection_wrap_up(gray_image: np.ndarray) -> np.ndarray:
    """ Applying cornerHarris on a given image and return corner's coordination after reshape
    Args:
        gray_image(np.ndarray): input grayscale image

    Returns:
        np.ndarray: array of tuples that contain coordination of corners
    """
    dst = cv2.cornerHarris(gray_image, 4, 3, 0.1)
    dst = cv2.dilate(dst, None)
    harris_edge_coordinates = np.where(dst > 0.001 * dst.max())

    return np.array(list(zip(harris_edge_coordinates[1], harris_edge_coordinates[0])), dtype=np.float32)


def point_to_point_distance(point1: tuple, point2: tuple) -> float:
    """Point to point distance calculation
    Args:
        point1(tuple): coordination of first point
        point2(tuple): coordination of second point

    Returns:
        float: distance between two given points
    """
    return (((point1[0] - point2[0]) ** 2) + ((point1[1] - point2[1]) ** 2)) ** 0.5


def final_shape_detector(harris_points: np.ndarray, hough_intersection_points: list, threshold_distance: int,
                         second_threshold) -> set:
    """finding 4 points around final shape
    Args:
        harris_points(np.ndarray): harris points from harris_corner_detection_wrap_up()
        hough_intersection_points(list): hough intersection points from hough_intersection()
        threshold_distance: threshold for distance between two points (hough and harris)
        second_threshold: threshold to remove the points which are almost identical

    Returns:
        set: set of suitable points
    """
    shape = set()
    for x in hough_intersection_points:
        for y in harris_points:
            if point_to_point_distance(x, y) < threshold_distance:
                if (x[0], x[1] + 1) not in shape and (x[0], x[1] - 1) not in shape:
                    shape.add(x)

    # Dummy way to remove  points closer than second threshold
    remove_set = set()
    for x in shape:
        for y in shape:
            if x[0] == y[0] and x[1] != y[1]:
                if abs(x[1] - y[1]) < second_threshold:
                    remove_set.add(x)
            elif x[1] == y[1] and x[0] != y[0]:
                if abs(x[0] - y[0]) < second_threshold:
                    remove_set.add(x)

    return shape - remove_set


def perspective_wrap_up(image: np.ndarray, first_points: list, second_points: list) -> np.ndarray:
    """ right way to  use perspective transformation some parts are extracted from OpenCV documentations

    Args:
        image(np.ndarray): input image
        first_points(list): our shape's corners found with final_shape_detector()
        second_points(list): target coordination

    Returns:
        np.ndarray: image after transformation
    """
    first_coordination, second_coordination = np.float32(first_points), np.float32(second_points)
    matrix = cv2.getPerspectiveTransform(first_coordination, second_coordination)
    max_x, max_y = 0, 0

    for i in second_points:
        if i[0] > max_x:
            max_x = i[0]
        if i[1] > max_y:
            max_y = i[1]

    after_perspective = cv2.warpPerspective(image, matrix, (max_x, max_y))
    return after_perspective


def histogram(image: np.ndarray) -> list:
    """Calculation of histogram of an image

    Args:
        image(np.ndarray): input image

    Returns:
        list: histogram of our image
    """
    # Creating a list of zeros
    h = list(np.zeros(256))

    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            i = image[x, y]
            h[i] = h[i] + 1
    return h


def histogram_eq(image: np.ndarray) -> np.ndarray:
    """Histogram equalization transform

    Args:
        image(np.ndarray): input image

    Returns:
        np.ndarray: image with equalized histogram
    """
    height, width = image.shape
    equalized = image
    tmp = 1.0 / (height * width)
    b = np.zeros((256,), dtype=np.float16)
    a = histogram(image)
    for i in range(256):
        for j in range(0, i + 1):
            b[i] += a[j] * tmp
        b[i] = round(b[i] * 255)

    # now b contains the equalized histogram
    b = b.astype(np.uint8)
    # Re-map values from equalized histogram into the image pixels
    for i in range(width):
        for j in range(height):
            g = image[j, i]
            equalized[j, i] = b[g]

    return equalized


def document_scanner(image: np.ndarray, max_x: int = 600, max_y: int = 500, shape_detector_threshold: int = 10,
                     hough_threshold: int = 150) -> tuple:
    """Final function to wrap up previous functions which were necessary for document scanning process
    Args:
        image(np.ndarray):input document
        max_x(int): maximum X of our target scale
        max_y(int): maximum Y of our target scale
        shape_detector_threshold(int): first threshold of final_shape_detector()
        hough_threshold: threshold for hough line detector

    Returns:
        tuple: images after document scanning process
    """
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    final_document = final_shape_detector(harris_corner_detection_wrap_up(gray_image),
                                          hough_intersections(gray_image, hough_threshold), shape_detector_threshold, 4)
    document_corners = sorted(list(final_document))
    page_corners = [[0, 0], [max_x, 0], [0, max_y], [max_x, max_y]]

    final_document = perspective_wrap_up(image, document_corners, page_corners)
    final_document_histogram_eq = histogram_eq(cv2.cvtColor(final_document, cv2.COLOR_BGR2GRAY))

    return final_document, final_document_histogram_eq


def plot(image: np.ndarray, name):
    plt.title(name)
    plt.imshow(image, cmap='gray')
    plt.show()


if __name__ == '__main__':
    document = cv2.imread('paper.png')

    scanned = document_scanner(document)

    plot(scanned[0], 'scanned document')
    plot(scanned[1], 'histogram equalized output')
