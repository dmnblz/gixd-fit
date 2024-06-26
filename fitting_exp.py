import numpy as np
from typing import List, Tuple
import pickle
from Dataset.dataset.dataset_adv import H5GIWAXSDataset
from Dataset.dataset.dataset import calc_polar_image, calc_quazipolar_image
from Dataset.dataset.dataset_h import H5GIWAXSDataset_h
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import os
from skimage.morphology import disk, opening
from skimage import data, img_as_float
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit
from multiprocessing import Pool
import lmfit
import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
from scipy.ndimage import label
import pywt
from matplotlib.patches import Ellipse
import emcee
import corner
from lm_functions import LMFunctions
from fit_functions_lmfit import N1DGaussian
import pandas as pd
import time
from numba import jit
from math import sqrt
from scipy.spatial import distance
import matplotlib.gridspec as gridspec
from fit_functions_lmfit import FitBoxes

from gixd_detectron.metrics.match_criteria import (
    Matcher,
    IoUMatcher,
    QMatcher,
)

from gixd_detectron.ml.postprocessing import (
    Postprocessing,
    StandardPostprocessing,
    MergeBoxesPostprocessing,
    SmallQFilter,
    TargetBasedSmallQFilter,
)

from gixd_detectron.metrics.calc_metrics import get_exp_metrics
from gixd_detectron.simulations import FastSimulation, SimulationConfig

from gixd_detectron.ml.postprocessing import (
    Postprocessing,
    PostprocessingPipeline,
    StandardPostprocessing,
    MergeBoxesPostprocessing,
    SmallQFilter,
    TargetBasedSmallQFilter,
)


class SimpleBinaryClassifier(nn.Module):
    def __init__(self):
        super(SimpleBinaryClassifier, self).__init__()
        self.fc1 = nn.Linear(14, 32)  # First hidden layer
        self.relu = nn.ReLU()  # ReLU activation function
        self.fc2 = nn.Linear(32, 16)  # Second hidden layer
        self.output = nn.Linear(16, 1)  # Output layer
        self.sigmoid = nn.Sigmoid()  # Sigmoid activation function

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.output(x))
        return x


class SimpleBinaryClassifier_12(nn.Module):
    def __init__(self):
        super(SimpleBinaryClassifier_12, self).__init__()
        self.fc1 = nn.Linear(12, 32)  # First hidden layer
        self.relu = nn.ReLU()  # ReLU activation function
        self.fc2 = nn.Linear(32, 16)  # Second hidden layer
        self.output = nn.Linear(16, 1)  # Output layer
        self.sigmoid = nn.Sigmoid()  # Sigmoid activation function

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.output(x))
        return x



def image_show(image):
    plt.imshow(image, cmap="gray")
    # plt.axis("off")
    # plt.show()


def d3_plot(image):
    # plot 3d terrain
    x = np.linspace(0, image.shape[1] - 1, image.shape[1])
    y = np.linspace(0, image.shape[0] - 1, image.shape[0])
    X, Y = np.meshgrid(x, y)

    # create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the terrain surface
    ax.plot_surface(X, Y, image, cmap='viridis')

    # Add labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.invert_xaxis()

    # Show the plot
    plt.show()


def flatten_x_axis(image):
    flat_image = []
    for i in range(image.shape[0]):
        image_row = image[i, :]
        non_zero = (image_row != 0)
        non_zero_number = np.sum(non_zero)
        row_sum = np.sum(image_row)
        relative_sum = row_sum / non_zero_number
        flat_image.append(relative_sum)
    return flat_image


def flatten_y_axis(image):
    """
    Calculates the mean of each row in the y direction excluding the 0's (the part where there is no detection)
    """
    flat_image = []
    for i in range(image.shape[1]):
        image_row = image[:, i]
        non_zero = (image_row != 0)
        non_zero_number = np.sum(non_zero)
        row_sum = np.sum(image_row)
        relative_sum = row_sum / non_zero_number
        flat_image.append(relative_sum)
    return flat_image


def flatten_y_axis(image):
    """
    Calculates the mean of each row in the y direction excluding the 0's (the part where there is no detection)
    """
    non_zero_counts = np.count_nonzero(image, axis=0)
    sum_values = np.sum(image, axis=0)
    # Avoid division by zero for columns that are all zeros
    non_zero_counts[non_zero_counts == 0] = 1
    flat_image = sum_values / non_zero_counts

    return flat_image


def rolling_ball_background_subtraction(image, radius):
    # Create a disk-shaped structuring element with the desired radius
    selem = disk(radius)

    # Use morphological opening to estimate the background
    background = opening(image, selem)

    # Subtract the background
    subtracted = image - background

    # To avoid negative values after subtraction, you can clip the result
    subtracted = np.clip(subtracted, 0, None)

    return subtracted, background


def execute_rolling_ball(image, r):
    # Test with a sample image
    image = img_as_float(image)  # Convert to floating point
    radius = r
    subtracted, background = rolling_ball_background_subtraction(image, radius)

    # Display the results
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))

    ax[0].imshow(image, cmap='gray')
    ax[0].set_title('Original Image')
    ax[0].axis('off')

    ax[1].imshow(background, cmap='gray')
    ax[1].set_title('Estimated Background')
    ax[1].axis('off')

    ax[2].imshow(subtracted, cmap='gray')
    ax[2].set_title('Background Subtracted Image')
    ax[2].axis('off')

    plt.tight_layout()
    plt.show()

    return subtracted


def draw_boxes(boxes, ratios, limit):
    """
    draws boxes
    red = line
    green = peak
    """
    for box, ratio in zip(boxes, ratios):
        x = box[0]
        y = box[3]
        x_width = box[2] - box[0]
        y_width = box[3] - box[1]
        if ratio >= limit:
            rect = patches.Rectangle((x, y), x_width, -y_width, linewidth=1, edgecolor='r', facecolor='none')
        else:
            rect = patches.Rectangle((x, y), x_width, -y_width, linewidth=1, edgecolor='g', facecolor='none')
        plt.gca().add_patch(rect)


def draw_boxes_basic(boxes, color='r', linewidth=2):
    """
    draws boxes
    """
    for box in boxes:
        x = box[0]
        y = box[3]
        x_width = box[2] - box[0]
        y_width = box[3] - box[1]
        rect = patches.Rectangle((x, y), x_width, -y_width, linewidth=linewidth, edgecolor=color, facecolor='none')
        plt.gca().add_patch(rect)



def draw_groupboxes(boxes):
    """
    draws pink boxes
    """
    for box in boxes:
        x = box[0]
        y = box[3]
        x_width = box[2] - box[0]
        y_width = box[3] - box[1]
        rect = patches.Rectangle((x, y), x_width, -y_width, linewidth=1, edgecolor='pink', facecolor='none')
        plt.gca().add_patch(rect)


def draw_single_box(box):
    """
    draws single box
    """
    x = box[0]
    y = box[3]
    x_width = box[2] - box[0]
    y_width = box[3] - box[1]
    rect = patches.Rectangle((x, y), x_width, -y_width, linewidth=1, edgecolor='r', facecolor='none')
    plt.gca().add_patch(rect)


def draw_boxes_ratios(boxes, ratios, limit):
    """
    draws text next to the boxes about the box/"actual image" ratio
    red = line
    green = peak
    """
    for box, ratio in zip(boxes, ratios):
        x = box[0]
        y = box[3]
        x_width = box[2] - box[0]
        y_width = box[3] - box[1]
        text = str(round(ratio, 2))
        if ratio >= limit:
            plt.gca().text(x + x_width, y, text, fontsize=10, color="red")
        else:
            plt.gca().text(x + x_width, y, text, fontsize=10, color="green")


def draw_enumerate_boxes(boxes, ratios, limit):
    """
    enumerates the peaks for better understanding
    red = line
    green = peak
    """
    for box, ratio in zip(enumerate(boxes), ratios):
        x = box[1][0]
        y = box[1][3]
        x_width = box[1][2] - box[1][0]
        y_width = box[1][3] - box[1][1]
        text = str(box[0])
        if ratio >= limit:
            plt.gca().text(x + x_width, y - y_width, text, fontsize=10, color="red")
        else:
            plt.gca().text(x + x_width, y - y_width, text, fontsize=10, color="green")


def draw_enumerate_boxes_simple(boxes):
    """
    enumerates the peaks for better understanding
    red = line
    green = peak
    """
    for box in enumerate(boxes):
        x = box[1][0]
        y = box[1][3]
        x_width = box[1][2] - box[1][0]
        y_width = box[1][3] - box[1][1]
        text = str(box[0])

        plt.gca().text(x + x_width, y - y_width, text, fontsize=10, color="red")



def draw_enumerate_boxes_simple_id(boxes):
    """
    enumerates the peaks for better understanding
    red = line
    green = peak
    """
    for box in boxes:
        x = box[0]
        y = box[3]
        x_width = box[2] - box[0]
        y_width = box[3] - box[1]
        text = str(box[4])

        plt.gca().text(x + x_width, y - y_width, text, fontsize=10, color="b")



def draw_enumerate_boxes_simple_scores(boxes, scores):
    """
    enumerates the peaks for better understanding
    red = line
    green = peak
    """
    for box, score in zip(boxes, scores):
        x = box[0]
        y = box[3]
        x_width = box[2] - box[0]
        y_width = box[3] - box[1]
        text = str(round(score, 2))

        plt.gca().text(x + x_width, y - y_width, text, fontsize=10, color="b")


def get_box_mid(boxes):
    boxes_mids = []
    for box in boxes:
        x_mid = box[0] + (box[2] - box[0]) / 2
        y_mid = box[1] + (box[3] - box[1]) / 2
        boxes_mids.append([x_mid, y_mid])
    return np.array(boxes_mids)


def calculate_box_to_height_ratios(image, boxes):
    """
    Calculates the box / height ratio
    height is defined as part of the image that is not zero (aka the part where the recording is)
    """

    boxes_heights = []
    boxes_x_mid = []
    for box in boxes:
        boxes_heights.append(int(np.round(box[3] - box[1])))
        boxes_x_mid.append(int(np.round(box[0] + (box[2] - box[0]) / 2)))
    image_heights = calculate_available_image_height(image[:, boxes_x_mid])
    ratios = []
    for i, b in zip(image_heights, boxes_heights):
        ratios.append(b / i)
    return ratios


def draw_mids_y(mids):
    for mid in mids[:, 0]:
        plt.axvline(x=mid, color='red', linestyle='--', label='Vertical Line')


def drop_boxes_based_on_confidence(labels, conf):
    '''drops boxes of a cirtain confidence'''
    return np.delete(labels.boxes, np.where(np.isin(labels.confidences, conf)), axis=0)


def single_peak(image, box, x_peak_extend, y_peak_extend):
    """
    filter out a single peak
    peak extend is percent of the box x width
    """
    box = np.ceil(box).astype(int)
    # peak_extend extend the cut image of the peak by this amount of pixels
    xpe = int(np.ceil((box[2] - box[0]) * x_peak_extend) + 1)  # the 1 is there in case the box is of size 0
    ype = int(np.ceil((box[3] - box[1]) * y_peak_extend) + 1)  # the 1 is there in case the box is of size 0
    if any(bx == 0 for bx in box):
        peak = image[(box[1]):(box[3]),  # y direction
               (box[0]):(box[2])]  # x direction
    else:
        peak = image[(box[1] - ype):(box[3] + ype),  # y direction
               (box[0] - xpe):(box[2] + xpe)]  # x direction
    return peak


def median_filter(image):
    """Apply median filtering on the image."""
    kernel_size = 3
    return cv2.medianBlur(image, kernel_size)


def gaussian_blur(image, size):
    """Apply gaussian on the image."""
    kernel_size = (size, size)  # Adjust the size as needed
    sigma = 0
    return cv2.GaussianBlur(image, kernel_size, sigma)


def gaussian_blur_y_only(image):
    """Apply gaussian on the image in only y direction"""
    kernel_size = (1, 21)  # Adjust the size as needed
    sigma = 0
    return cv2.GaussianBlur(image, kernel_size, sigma)


def compute_noise_in_direction(image):
    """ calculates the noise in each direction separately"""
    # Calculate the difference in the x direction (horizontal)
    diff_x = np.diff(image, axis=1)

    # Calculate the difference in the y direction (vertical)
    diff_y = np.diff(image, axis=0)

    # Compute the standard deviation of differences as a measure of noise
    noise_std_x = np.std(diff_x)
    noise_std_y = np.std(diff_y)

    return noise_std_x, noise_std_y


def keep_only_upper_percent_intensity(image, per):
    """ removes the lower part of image intenisty completely"""
    max_int = np.max(image)
    min_int = np.min(image)
    dif_int = max_int - min_int
    top_int = dif_int * per
    s_peak_top_int = image.copy()
    s_peak_top_int[s_peak_top_int < top_int] = 0

    return s_peak_top_int


def detect_line(image):
    theta = np.nan

    # smooth the image for better detection
    image_smooth = gaussian_blur(image)
    image_smooth = np.round(image_smooth).astype(np.uint8)

    # image_show(image_smooth)
    # plt.show()

    # Apply edge detection to highlight edges (e.g., using Canny)
    edges = cv2.Canny(image_smooth, threshold1=50, threshold2=150, apertureSize=3)

    # Use Hough Line Transform to detect lines
    lines = cv2.HoughLines(edges, rho=1, theta=np.pi / 180, threshold=100)

    # Draw detected lines on the original image
    if lines is not None:
        for line in lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255 / 2), 2)  # Draw the line in red

    return theta


def rotate_image(image, angle):
    # Define the angle of rotation (in degrees)
    angle = angle * 180 / np.pi  # Adjust the angle as needed

    # Get the image dimensions
    height, width = image.shape[:2]

    # Calculate the rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)

    # Perform the rotation
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))

    return rotated_image


def find_zero_ranges(lst):
    zero_ranges = []
    start = None

    for i, value in enumerate(lst):
        if value == 0:
            if start is None:
                start = i
        else:
            if start is not None:
                zero_ranges.append((start, i - 1))
                start = None

    if start is not None:
        zero_ranges.append((start, len(lst) - 1))

    return zero_ranges


def find_zero_ranges_np(lst):
    lst = np.array(lst)
    diff = np.diff(np.concatenate(([0], lst != 0, [0])))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0] - 1
    return list(zip(starts, ends))


def calculate_available_image_height(image):
    """Calculates the height in pixels of the actual measurement"""
    h = []
    for p in np.transpose(image):
        zeros = find_zero_ranges_np(p)
        if len(zeros) != 0:
            len_to_remove = []
            for z in zeros:
                len_to_remove.append(z[1] + 1 - z[0])
            h.append(len(p) - sum(len_to_remove))
        else:
            h.append(len(p))
    return h


def calculate_usable_pixels(image):
    """
    Calculates amount of pixels with information in them -> doesnt count 0´s
    """
    return np.count_nonzero(image)


def gauss_slope(x, a, mu, c, m, b):
    gauss = a * np.exp(-((x - mu) ** 2) / (2 * c ** 2))
    linear = m * x + b
    return gauss + linear


def initial_guesses(flat_image, cutoff):
    x = (len(flat_image) + cutoff) / 2 - cutoff  # calculate the new relative mid
    dx = len(flat_image)  # width of the x box
    a = flat_image[round(x)]  # amplitude at the middle of the box
    mu = x  # middle of the box
    c = dx / 1.5  # standard deviation of the predicted model
    m = (flat_image[-1] - flat_image[0]) / dx
    b = flat_image[0]
    return [a, mu, c, m, b]


def fit_gaussian(image, weight_function, peak_nr):
    # flattens the image
    y = flatten_y_axis(image)
    # crops the data in such way that if the y value becomes larger than the given 1.05 x amplitude it cuts it off
    y, start_cutoff = find_and_cut_list(y, 1.05)
    # calculates initial guess from the data
    ig = initial_guesses(y, start_cutoff)
    # set boundary conditions: a, mu, c, m, b (amplitude, mean, std, slope, intercept)
    lower_bounds = [0, 0, 0, -np.inf, 0]
    upper_bounds = [np.inf, len(y), np.inf, np.inf, np.inf]
    x = np.arange(0, len(y), 1)

    if weight_function == True:
        # Create weights: Higher weights for data near the center
        weights = 1 / (0.3 + (x / 0.5) ** 2)
        params, covariance = curve_fit(gauss_slope, x, y, p0=ig, maxfev=100000, sigma=weights, absolute_sigma=True,
                                       bounds=(lower_bounds, upper_bounds))
    else:
        params, covariance = curve_fit(gauss_slope, x, y, p0=ig, maxfev=100000, bounds=(lower_bounds, upper_bounds))

    y_fit = gauss_slope(x, params[0], params[1], params[2], params[3], params[4])

    # goodnes of fit
    mee = mse(np.array(y), y_fit)
    var = np.var(y)
    pixel_count = calculate_usable_pixels(image)
    mse_var_ratio = mee / var
    mse_var_pixel_ratio = mee / (var * pixel_count)

    # plot
    plt.figure(1)
    plt.subplot(1, 2, 2)
    plt.plot(x, y)
    plt.plot(x, y_fit)
    plt.gca().text(min(x), max(y), "mse/variance: " + str(round(mse_var_ratio, 4)), fontsize=10, color="red")
    plt.gca().text(min(x), min(y), "pixels: " + str(pixel_count), fontsize=8, color="red")
    plt.subplot(1, 2, 1)
    image_show(image)
    plt.gca().text(0, -5, str(peak_nr), fontsize=10, color="red")
    plt.scatter(params[1] + start_cutoff, len(image) / 2, color='red', marker='x', label='Point')
    plt.show()

    return params, covariance


def fit_1d_gaussian(image, box):
    def gaussian_with_slope(x, A, sigma, mu, m, b):
        """
        Gaussian distribution with a linear background.
        A: Amplitude of the Gaussian
        sigma: Standard deviation of the Gaussian
        mu: Mean of the Gaussian
        m: Slope of the linear background
        b: Intercept of the linear background
        x: Independent variable
        """
        gaussian = A * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))
        linear = m * x + b
        return gaussian + linear

    def gaussian_with_slope_deriv(x, A, sigma, mu, m, b):
        expon = np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))
        d_A = expon
        d_sigma = A * ((x - mu) ** 2 / sigma ** 3) * expon
        d_mu = A * ((x - mu) / sigma ** 2) * expon
        d_m = x
        d_b = np.ones_like(x)

        return np.array([d_A, d_sigma, d_mu, d_m, d_b]).T

    ex = int(round((box[2] - box[0]) / 2))

    y0 = np.array(image[int(np.floor(box[0] - ex)):int(np.ceil(box[2] + ex))])
    y = y0[~np.isnan(y0)]  # remove nan if extension goes out of the image

    # count nans at the start
    num_nans_at_start = np.where(~np.isnan(y0))[0][0]

    x = np.arange(len(y))

    # Create a model
    model = lmfit.Model(gaussian_with_slope, fjac=gaussian_with_slope_deriv)

    # Create parameters with initial guesses
    params = lmfit.Parameters()
    params.add('A', value=max(y) - min(y), min=(max(y) - min(y)) / 2, max=(max(y) - min(y)) * 1.3)
    params.add('sigma', value=(box[2] - box[0]) / 2, min=0, max=box[2] - box[0])
    params.add('mu', value=x[np.argmax(y)], min=ex, max=ex + (box[2] - box[0]))
    params.add('m', value=min(y), min=0, max=max(y))
    params.add('b', value=y[0])

    # Fit the model to the data
    result = model.fit(y, params, x=x, method='leastsq')

    # Optionally, plot the results
    mid = result.params["mu"].value
    plt.plot(x + np.floor(box[0]) - ex + num_nans_at_start, result.best_fit, 'r--')  # Fitted curve

    plt.vlines(mid + np.floor(box[0]) - ex + num_nans_at_start, min(y), max(y), colors="g")


def mask_peak_boxes(image, boxes_peaks):
    """
    Masks (= 0) all areas where peaks are
    """
    boxes_peaks_int = []
    # make boxes int
    for br in boxes_peaks:
        boxes_peaks_int.append([np.floor(br[0]), np.floor(br[1]),
                                np.ceil(br[2]), np.ceil(br[3])])
    boxes_peak_int = np.array(boxes_peaks_int).astype(int)

    for box in boxes_peak_int:
        image[box[1]:box[3], box[0]:box[2]] = 0

    return image


def fit_gauss_line():
    # Define the 2D function
    def gaussian_line_2d(xy, mean_x, amplitude, sigma_x, slope, intercept):
        x, y = xy
        gaussian = amplitude * np.exp(-((x - mean_x) ** 2) / (2 * sigma_x ** 2))
        linear = slope * y + intercept
        return gaussian * linear

    # Example data (replace with your actual data)
    x = np.linspace(-10, 10, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    Z = gaussian_line_2d((X, Y), mean_x=0, amplitude=1, sigma_x=2, slope=0.1, intercept=0)

    # Add some noise
    Z += np.random.normal(0, 0.1, Z.shape)

    # Flatten the X, Y, Z arrays for fitting
    x_data = X.ravel()
    y_data = Y.ravel()
    z_data = Z.ravel()

    # Create a model
    model = lmfit.Model(gaussian_line_2d, independent_vars=['xy'])

    # Initial parameters
    params = model.make_params(mean_x=0, amplitude=1, sigma_x=1, slope=0, intercept=0)

    # Fit the model
    result = model.fit(z_data, xy=(x_data, y_data), params=params)

    # Print the fitting results
    print(result.fit_report())

    # Optionally, plot the results
    import matplotlib.pyplot as plt
    plt.imshow(Z, extent=[x.min(), x.max(), y.min(), y.max()], origin='lower')
    plt.title('Original Data')
    plt.show()

    plt.imshow(result.best_fit.reshape(X.shape), extent=[x.min(), x.max(), y.min(), y.max()], origin='lower')
    plt.title('Fitted Data')
    plt.show()


def mse(y, y_fit):
    return np.mean((y - y_fit) ** 2)


def find_and_cut_list(lst, f):
    """
    Takes the value in the middle, muliplies it with a factor f and cuts the list in both directions
    if the entry of the list exceeds this value
    lst is the list (y values) and f is the factor that multiplies the value in the middle

    returns the new list and where the old middle now is (for initial guess)

    TODO: MAKE THE FACTOR RELATIVE TO THE DIMENSIONS OF Y????
    """

    """
    Cuts the flat image off, when y-value becomes bigger than the assumed Amplitude
    y values, f = factor for allowed y value
    """

    # Find the index of the middle value aka amplitude
    middle_index = len(lst) // 2
    x = lst[middle_index] * f

    # Initialize the indices to start cutting from
    cut_index_start = 0
    cut_index_end = -1

    # Starting from the middle, find the index where the value becomes larger than x in both directions
    if max(lst[middle_index:-1]) > x:
        for i in range(middle_index, len(lst)):
            if lst[i] > x:
                cut_index_end = i
                break
    if max(lst[0:middle_index]) > x:
        for i in range(middle_index, -1, -1):
            if lst[i] > x:
                cut_index_start = i
                break

    # Cut the list in both directions from the identified indices
    if (cut_index_start == 0) and (cut_index_end == -1):
        pass
    else:
        if cut_index_end == -1:
            lst = lst[cut_index_start + 1:]
        else:
            lst = lst[cut_index_start + 1:cut_index_end]

    return lst, cut_index_start


####### Peak finding functions ########


def gaussian(x, y, amplitude, xo, yo, sigma_x, sigma_y, theta):
    xo = float(xo)
    yo = float(yo)
    a = (np.cos(theta) ** 2) / (2 * sigma_x ** 2) + (np.sin(theta) ** 2) / (2 * sigma_y ** 2)
    b = -(np.sin(2 * theta)) / (4 * sigma_x ** 2) + (np.sin(2 * theta)) / (4 * sigma_y ** 2)
    c = (np.sin(theta) ** 2) / (2 * sigma_x ** 2) + (np.cos(theta) ** 2) / (2 * sigma_y ** 2)
    g = amplitude * np.exp(- (a * ((x - xo) ** 2) + 2 * b * (x - xo) * (y - yo) + c * ((y - yo) ** 2)))
    return g


def draw_marks_on_peaks(locs):
    for (x0, y0) in locs:
        plt.scatter(x0, y0, color='red', marker='x', label='Point')


def peak_extend(box, x_peak_extend, y_peak_extend):
    """
    Extends the boundaries of the box by the factors given in x and y direction
    useful for acquiring more data for the fit by including the areas around the box
    returns extended box boundaries
    """
    # convert to int
    box = np.array([np.floor(box[0]), np.floor(box[1]), np.ceil(box[2]), np.ceil(box[3])]).astype(int)
    # peak_extend extend the cut image of the peak by this amount of pixels
    xpe = int(np.ceil((box[2] - box[0]) * x_peak_extend) + 1)  # the 1 is there in case the box is of size 0
    ype = int(np.ceil((box[3] - box[1]) * y_peak_extend) + 1)  # the 1 is there in case the box is of size 0
    return [box[0] - xpe, box[1] - ype, box[2] + xpe, box[3] + ype]


def cluster_peak_boxes(image, boxes, r, groupbox_extend):
    """
    r = range, the range of the gaussian blur that clusters peaks together
    groupbox_extend = after determining the clusters, extend the box even further
    TODO: MAYBE since the bigger peaks have a bigger range of extend, rewrite the groupbox_extend in such a way,
     that it depends on the size of the box --- Think about this more
    """

    def separate_peaks_lines(image, boxes, ratio):
        """
        Takes the image and the boxes and separates them into two lists of boxes, either peak or line
        Returns list of boxes of peaks and list of boxes of lines
        """
        box_ratios = calculate_box_to_height_ratios(image, boxes)
        boxes_peaks = []
        boxes_lines = []
        for bo, r in zip(boxes, box_ratios):
            # filter out peaks that are not lines
            if r <= ratio:
                boxes_peaks.append(bo)
            else:
                boxes_lines.append(bo)
        return np.array(boxes_peaks), np.array(boxes_lines)

    def correct_boxes_boundaries(boxes):
        """
        Makes sure the box boundaies are inside the image.
        Being outside the image causes problems with parts of the code
        """
        for box in boxes:
            if box[0] < 0:
                box[0] = 0
            if box[1] < 0:
                box[1] = 0
            if box[2] > 1024:
                box[2] = 1024
            if box[3] > 512:
                box[3] = 512
        return boxes

    def fill_clusters_with_ones(arr):
        # Identify and label each cluster
        labeled_array, num_features = label(arr)
        boxes_groups = []
        # For each cluster, compute its bounding box and fill it
        for i in range(1, num_features + 1):
            positions = np.where(labeled_array == i)
            ymin, ymax = np.min(positions[0]), np.max(positions[0])
            xmin, xmax = np.min(positions[1]), np.max(positions[1])

            arr[ymin:ymax + 1, xmin:xmax + 1] = 1
            boxes_groups.append([xmin - groupbox_extend, ymin - groupbox_extend,
                                 xmax + groupbox_extend, ymax + groupbox_extend])

        return arr, np.array(boxes_groups)

    def is_box_inside(initial_box, cluster_box):
        """
        Checks if the initial_box is inside the cluster_box.

        Each box is represented as [xmin, ymin, xmax, ymax].
        """

        i_xmin, i_ymin, i_xmax, i_ymax = initial_box
        c_xmin, c_ymin, c_xmax, c_ymax = cluster_box

        return (i_xmin >= c_xmin) and (i_xmax <= c_xmax) and (i_ymin >= c_ymin) and (i_ymax <= c_ymax)

    def get_boxes_inside_clusters(initial_boxes, cluster_boxes):
        """
        Returns a list where each entry consists of a list of the initial_boxes inside each cluster_box.
        """
        boxes_inside_clusters = []

        for cluster_box in cluster_boxes:
            boxes_inside = [init_box for init_box in initial_boxes if is_box_inside(init_box, cluster_box)]
            boxes_inside_clusters.append(boxes_inside)

        return boxes_inside_clusters

    boxes = correct_boxes_boundaries(boxes)

    # separate peaks and lines
    boxes_peak, boxes_line = separate_peaks_lines(image, boxes, ratio=0.8)

    # turn boxes into int
    boxes_peak_int = []
    for br in boxes_peak:
        boxes_peak_int.append([np.floor(br[0]), np.floor(br[1]),
                               np.ceil(br[2]), np.ceil(br[3])])
    boxes_peak_int = np.array(boxes_peak_int).astype(int)

    mask = np.zeros_like(image)
    for box in boxes_peak_int:
        mask[box[1]:box[3], box[0]:box[2]] = 1

    mask_blur = gaussian_blur(mask, r)
    mask2 = np.copy(mask_blur)
    mask2[mask2 != 0] = 1

    mask3, cluster_boxes = fill_clusters_with_ones(mask2)

    boxes_inside_clusters = get_boxes_inside_clusters(boxes_peak, cluster_boxes)

    return cluster_boxes, boxes_inside_clusters, boxes_peak, boxes_line


def cluster_peak_boxes2(image, boxes, r, groupbox_extend):
    """
    r = range, the range of the gaussian blur that clusters peaks together
    groupbox_extend = after determining the clusters, extend the box even further
    TODO: MAYBE since the bigger peaks have a bigger range of extend, rewrite the groupbox_extend in such a way,
     that it depends on the size of the box --- Think about this more
    """

    def separate_peaks_lines(image, boxes, ratio):
        """
        Takes the image and the boxes and separates them into two lists of boxes, either peak or line
        Returns list of boxes of peaks and list of boxes of lines
        """
        box_ratios = calculate_box_to_height_ratios(image, boxes)
        boxes_peaks = []
        boxes_lines = []
        for bo, r in zip(boxes, box_ratios):
            # filter out peaks that are not lines
            if r <= ratio:
                boxes_peaks.append(bo)
            else:
                boxes_lines.append(bo)
        return np.array(boxes_peaks), np.array(boxes_lines)

    def correct_boxes_boundaries(boxes):
        """
        Makes sure the box boundaies are inside the image.
        Being outside the image causes problems with parts of the code
        """
        for box in boxes:
            if box[0] < 0:
                box[0] = 0
            if box[1] < 0:
                box[1] = 0
            if box[2] > 1024:
                box[2] = 1024
            if box[3] > 512:
                box[3] = 512
        return boxes

    def fill_clusters_with_ones(arr):
        # Identify and label each cluster
        labeled_array, num_features = label(arr)
        boxes_groups = []
        # For each cluster, compute its bounding box and fill it
        for i in range(1, num_features + 1):
            positions = np.where(labeled_array == i)
            ymin, ymax = np.min(positions[0]), np.max(positions[0])
            xmin, xmax = np.min(positions[1]), np.max(positions[1])

            arr[ymin:ymax + 1, xmin:xmax + 1] = 1
            boxes_groups.append([xmin - groupbox_extend, ymin - groupbox_extend,
                                 xmax + groupbox_extend, ymax + groupbox_extend])

        return arr, np.array(boxes_groups)

    def is_box_inside(initial_box, cluster_box):
        """
        Checks if the initial_box is inside the cluster_box.

        Each box is represented as [xmin, ymin, xmax, ymax].
        """

        i_xmin, i_ymin, i_xmax, i_ymax = initial_box
        c_xmin, c_ymin, c_xmax, c_ymax = cluster_box

        return (i_xmin >= c_xmin) and (i_xmax <= c_xmax) and (i_ymin >= c_ymin) and (i_ymax <= c_ymax)

    def get_boxes_inside_clusters(initial_boxes, cluster_boxes):
        """
        Returns a list where each entry consists of a list of the initial_boxes inside each cluster_box.
        """
        boxes_inside_clusters = []

        for cluster_box in cluster_boxes:
            boxes_inside = [init_box for init_box in initial_boxes if is_box_inside(init_box, cluster_box)]
            boxes_inside_clusters.append(boxes_inside)

        return boxes_inside_clusters

    # make sure boxes are in bounds of the image
    boxes = correct_boxes_boundaries(boxes)

    # separate peaks and lines
    boxes_peak, boxes_line = separate_peaks_lines(image, boxes, ratio=0.8)

    # turn boxes into int
    if np.any(boxes_peak):
        boxes_peak_int = np.concatenate([np.floor(boxes_peak[:, :2]), np.ceil(boxes_peak[:, 2:])], axis=1)
        boxes_peak_int = boxes_peak_int.astype(int)
    else:
        boxes_peak_int = boxes_peak

    # turn boxes into int
    if np.any(boxes_line):
        boxes_line_int = np.concatenate([np.floor(boxes_line[:, :2]), np.ceil(boxes_line[:, 2:])], axis=1)
        boxes_line_int = boxes_line_int.astype(int)
    else:
        boxes_line_int = boxes_line

    """PEAK MASK"""
    mask_peak = np.zeros_like(image)
    for box in boxes_peak_int:
        mask_peak[box[1]:box[3], box[0]:box[2]] = 1

    # mask everything that is not a peak
    mask_peak_blur = gaussian_blur(mask_peak, r)
    mask_peak2 = np.copy(mask_peak_blur)
    mask_peak2[mask_peak2 != 0] = 1

    # cluster the peaks
    mask_peak3, p_cluster_boxes = fill_clusters_with_ones(np.copy(mask_peak2))
    p_boxes_inside_clusters = get_boxes_inside_clusters(boxes_peak, p_cluster_boxes)

    """LINE MASK"""
    mask_line = np.zeros_like(image)
    for box in boxes_line_int:
        mask_line[box[1]:box[3], box[0]:box[2]] = 1

    # mask everything that is not a line
    mask_line_blur = gaussian_blur(mask_line, r)
    mask_line2 = np.copy(mask_line_blur)
    mask_line2[mask_line2 != 0] = 1

    # isolate the lines (removes background and peaks in the way)
    image_line_data = np.copy(image)
    image_line_data[mask_peak2 == 1] = 0
    image_line_data[mask_line2 == 0] = 0
    #
    # image_line_data0 = np.copy(giwaxs_img)
    # image_line_data0[mask_peak2 == 1] = 0
    # image_line_data0[mask_line2 == 0] = 0
    #
    # plt.imshow(image_line_data0, cmap='gray')
    # plt.show()

    # cluster the lines
    mask_line3, l_cluster_boxes = fill_clusters_with_ones(np.copy(mask_line2))
    l_boxes_inside_clusters = get_boxes_inside_clusters(boxes_line, l_cluster_boxes)
    # draw_groupboxes(l_cluster_boxes)

    # image_show(image_line_data)
    # plt.show()

    image_line_data_y_flat = flatten_y_axis(image_line_data)
    # plt.plot(image_line_data_y_flat)

    return p_cluster_boxes, p_boxes_inside_clusters, l_cluster_boxes, l_boxes_inside_clusters, boxes_peak, boxes_line, image_line_data_y_flat


def blur_background_without_boxes(image, boxes):
    boxes_int = []
    for br in boxes:
        boxes_int.append([np.floor(br[0]), np.floor(br[1]),
                          np.ceil(br[2]), np.ceil(br[3])])
    boxes_int = np.array(boxes_int).astype(int)

    image_blur = gaussian_blur(image, 3)
    image_blur = gaussian_blur(image_blur, 5)
    image_blur = gaussian_blur(image_blur, 7)
    image_blur = gaussian_blur(image_blur, 15)
    for box in boxes_int:
        image_blur[box[0]:box[2], box[1]:box[3]] = 0  # image[box[0]:box[2], box[1]:box[3]]
    image_show(image_blur)
    plt.show()


def wavelet_transform(image):
    # Perform 2D Discrete Wavelet Transform
    coeffs2 = pywt.dwt2(image, 'haar')

    # coeffs2 is a tuple containing the coefficients
    cA, (cH, cV,
         cD) = coeffs2  # cA = Approximation coefficients, cH = Horizontal details, cV = Vertical details, cD = Diagonal details

    # To visualize
    plt.imshow(cA, cmap='gray')
    plt.title("wavelet")
    plt.show()


def draw_simga_ellipses(fits, axis):
    for fit_i in fits:
        # amplitude, xo, yo, sigma_x, sigma_y, theta, offset
        #     0      1   2     3        4       5        6
        ellipse = Ellipse(xy=(fit_i[1], fit_i[2]), width=2 * fit_i[3], height=2 * fit_i[4],
                          angle=np.degrees(-fit_i[5]),
                          edgecolor=(1, 0, 0), fc='None', lw=2, ls='--')
        axis.add_patch(ellipse)
        ellipse = Ellipse(xy=(fit_i[1], fit_i[2]), width=4 * fit_i[3], height=4 * fit_i[4],
                          angle=np.degrees(-fit_i[5]),
                          edgecolor=(0.5, 0, 0), fc='None', lw=2, ls='--')
        axis.add_patch(ellipse)


def plot_fit_image(image, results):
    fit_image = np.zeros_like(image)
    # prepare the image for fitting
    y_len, x_len = fit_image.shape
    x, y = np.meshgrid(np.arange(x_len), np.arange(y_len))
    X, Y = x.ravel(), y.ravel()
    for res in results:
        # amplitude, xo, yo, sigma_x, sigma_y, theta, offset
        #     0      1   2     3        4       5        6
        fit = gaussian(X, Y, res[0], res[1], res[2], res[3], res[4], res[5]).reshape((y_len, x_len))
        fit_image = fit_image + fit
    image_show(fit_image)
    plt.show()


def mcmc():
    # Assuming you have your data in 'x_data' and 'y_data' variables
    # Let's create some synthetic data for demonstration purposes
    np.random.seed(42)
    x_data = np.linspace(0, 10, 100)
    true_slope = 0.5
    true_offset = 4.0
    true_amplitude = 1.0
    true_mean = 5.0
    true_std_dev = 0.5
    y_data = true_offset + true_slope * x_data + true_amplitude * np.exp(
        -0.5 * (x_data - true_mean) ** 2 / true_std_dev ** 2) + np.random.normal(0, 0.1, size=len(x_data))

    # Define the model function
    def gaussian_with_slope_and_offset(params, x):
        offset, slope, amplitude, mean, std_dev = params
        return offset + slope * x + amplitude * np.exp(-0.5 * (x - mean) ** 2 / std_dev ** 2)

    # Define the log likelihood function
    def log_likelihood(params, x, y):
        offset, slope, amplitude, mean, std_dev = params
        model = gaussian_with_slope_and_offset(params, x)
        sigma = 0.1  # assumed known error
        return -0.5 * np.sum((y - model) ** 2 / sigma ** 2 + np.log(2 * np.pi * sigma ** 2))

    # Define the log prior function
    def log_prior(params):
        offset, slope, amplitude, mean, std_dev = params
        if 0 < amplitude < 2 and 0 < mean < 10 and 0 < std_dev < 2 and 0 < offset < 10 and -2 < slope < 2:
            return 0.0  # log(1)
        return -np.inf  # log(0)

    # Define the log probability function
    def log_probability(params, x, y):
        lp = log_prior(params)
        if not np.isfinite(lp):
            return -np.inf
        return lp + log_likelihood(params, x, y)

    # Set up the properties of the problem
    ndim = 5  # number of parameters in the model: offset, slope, amplitude, mean, std_dev
    nwalkers = 200  # number of MCMC walkers
    nburn = 500  # "burn-in" period to let chains stabilize
    nsteps = 4000  # number of MCMC steps to take

    # Set up the initial positions of the walkers in the parameter space
    np.random.seed(42)
    starting_guesses = np.random.rand(nwalkers, ndim)

    # Initialize the sampler
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(x_data, y_data))

    # Run the MCMC simulation
    sampler.run_mcmc(starting_guesses, nsteps, progress=True)

    # Discard the burn-in samples and flatten the rest
    samples = sampler.get_chain(discard=nburn, flat=True)

    """CORNER PLOT"""
    # Generate a corner plot
    corner_fig = corner.corner(samples, labels=["offset", "slope", "amplitude", "mean", "std_dev"],
                               truths=[true_offset, true_slope, true_amplitude, true_mean, true_std_dev])
    corner_fig.suptitle('MCMC Sampling with emcee')
    plt.show()

    """FIT PARAMETERS"""
    # Compute the quantiles for each parameter
    quantiles = np.percentile(samples, [16, 50, 84], axis=0)
    medians = quantiles[1]  # The median of the posterior
    lower_bounds = quantiles[0]
    upper_bounds = quantiles[2]

    # Print out the results
    print("Parameter estimates:")
    for i, param in enumerate(["offset", "slope", "amplitude", "mean", "std_dev"]):
        median = medians[i]
        lower_bound = median - lower_bounds[i]
        upper_bound = upper_bounds[i] - median
        print(f"{param}: {median:.3f} (+{upper_bound:.3f}, -{lower_bound:.3f})")

    # To find the maximum a posteriori estimate, you would find the sample with the highest posterior probability
    map_idx = np.argmax(
        sampler.flatlnprobability)  # This gives the index of the flattened chain with the highest log probability
    map_estimate = samples[map_idx]
    print("\nMaximum A Posteriori Estimates:")
    for i, param in enumerate(["offset", "slope", "amplitude", "mean", "std_dev"]):
        print(f"{param}: {map_estimate[i]:.3f}")

    # Using the medians as the best estimate
    best_fit_params = medians
    best_fit_curve = gaussian_with_slope_and_offset(best_fit_params, x_data)

    # Plot the data and the best fit curve
    plt.figure(figsize=(10, 5))
    plt.plot(x_data, y_data, 'o', label='Data')
    plt.plot(x_data, best_fit_curve, label='Best Fit')
    plt.legend()
    plt.xlabel('X data')
    plt.ylabel('Y data')
    plt.title('Data and Best Fit Curve from MCMC')
    plt.show()


def mcmc_2d(image, cont=False):
    x_coords = np.linspace(0, 10, image.shape[1])  # Replace with your actual x-coordinates
    y_coords = np.linspace(0, 10, image.shape[0])  # Replace with your actual y-coordinates
    x, y = np.meshgrid(x_coords, y_coords)

    # Flatten the x, y, and z data for use in emcee
    x_data = x.ravel()
    y_data = y.ravel()
    z_data = image.ravel()

    # Define the model function
    def gaussian_2d(params, x, y):
        x_mean, y_mean, x_sigma, y_sigma, amplitude, offset = params
        return offset + amplitude * np.exp(-((x - x_mean) ** 2 / (2 * x_sigma ** 2) +
                                             (y - y_mean) ** 2 / (2 * y_sigma ** 2)))

    # Define the log likelihood function
    def log_likelihood(params, x, y, z):
        model = gaussian_2d(params, x, y)
        sigma = 0.1  # assumed known error
        return -0.5 * np.sum((z - model) ** 2 / sigma ** 2 + np.log(2 * np.pi * sigma ** 2))

    # Define the log prior function
    def log_prior(params):
        x_mean, y_mean, x_sigma, y_sigma, amplitude, offset = params
        if 0 < x_sigma < 10 and 0 < y_sigma < 10 and 0 < amplitude < 1000 and 0 < offset < 1000:
            return 0.0  # log(1)
        return -np.inf  # log(0)

    # Define the log probability function
    def log_probability(params, x, y, z):
        lp = log_prior(params)
        if not np.isfinite(lp):
            return -np.inf
        return lp + log_likelihood(params, x, y, z)

    # Set up the properties of the problem
    ndim = 6  # number of parameters in the model
    nwalkers = 100  # number of MCMC walkers
    nburn = 500  # "burn-in" period to let chains stabilize
    nsteps = 5000  # number of MCMC steps to take

    starting_guesses = np.random.rand(nwalkers, ndim)
    # Set up the initial positions of the walkers in the parameter space
    # Assuming you have some reasonable ranges for your parameters
    # x_mean_start = np.random.uniform(low=4, high=6, size=nwalkers)
    # y_mean_start = np.random.uniform(low=4, high=6, size=nwalkers)
    # x_sigma_start = np.random.uniform(low=0.1, high=10, size=nwalkers)  # Avoid starting at zero
    # y_sigma_start = np.random.uniform(low=0.1, high=10, size=nwalkers)  # Avoid starting at zero
    # amplitude_start = np.random.uniform(low=0, high=np.max(image), size=nwalkers)
    # offset_start = np.random.uniform(low=0, high=np.max(image), size=nwalkers)
    # sigma_start = np.random.uniform(low=0.1, high=0.004, size=nwalkers)  # Avoid starting at zero
    #
    # # Combine these into the initial guesses
    # starting_guesses = np.array([x_mean_start, y_mean_start, x_sigma_start, y_sigma_start,
    #                              amplitude_start, offset_start, sigma_start]).T

    # Initialize the sampler
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(x_data, y_data, z_data))

    # Run the MCMC simulation
    sampler.run_mcmc(starting_guesses, nsteps, progress=True)

    # Discard the burn-in samples and flatten the rest
    samples = sampler.get_chain(discard=nburn, flat=True)

    # Generate a corner plot
    corner_fig = corner.corner(
        samples,
        color="blue",  # Sets the color of the 1D histograms
        hist_kwargs=dict(histtype='step', linewidth=1.5, alpha=1.0),
        labels=["x_mean", "y_mean", "x_sigma", "y_sigma", "amplitude", "offset", "sigma"],
        show_titles=True,
        title_kwargs={"fontsize": 12}
    )
    corner_fig.suptitle('MCMC Sampling with emcee for 2D Gaussian')
    plt.show()

    # Calculate the best-fitting parameters (using the median of the posterior here)
    best_fit_params = np.median(samples, axis=0)

    # Generate the best-fit 2D Gaussian
    z_fit = gaussian_2d(best_fit_params, x, y)

    # Plot the original data and the best-fit model side by side
    fig, axes = plt.subplots(1, 2, figsize=(16 / 2, 6 / 2), sharey=True)
    if cont:
        # Original data
        cnt_orig = axes[0].contourf(x, y, image.reshape(50, 50), cmap='viridis')
        fig.colorbar(cnt_orig, ax=axes[0], label='Intensity')
        axes[0].set_title('Original Data')
        axes[0].set_xlabel('X')
        axes[0].set_ylabel('Y')

        # Best-fit data
        cnt_fit = axes[1].contourf(x, y, z_fit.reshape(50, 50), cmap='viridis')
        fig.colorbar(cnt_fit, ax=axes[1], label='Intensity')
        axes[1].set_title('Best-fit 2D Gaussian')
        axes[1].set_xlabel('X')

        plt.tight_layout()
        plt.show()

    else:
        # Original data
        img_orig = axes[0].imshow(image, cmap='viridis', origin='lower', extent=(0, 10, 0, 10))
        fig.colorbar(img_orig, ax=axes[0], label='Intensity')
        axes[0].set_title('Original Data')
        axes[0].set_xlabel('X')
        axes[0].set_ylabel('Y')

        # Best-fit data
        img_fit = axes[1].imshow(z_fit, cmap='viridis', origin='lower', extent=(0, 10, 0, 10))
        fig.colorbar(img_fit, ax=axes[1], label='Intensity')
        axes[1].set_title('Best-fit 2D Gaussian')
        axes[1].set_xlabel('X')

        plt.tight_layout()
        plt.show()

    # Print the best fit parameters
    param_names = ["x_mean", "y_mean", "x_sigma", "y_sigma", "amplitude", "offset"]
    print("Best fit parameters:")
    for name, value in zip(param_names, best_fit_params):
        print(f"{name}: {value}")

    # If you want to report uncertainties you can use the 16th and 84th percentiles as a 1-sigma interval
    uncertainties = np.percentile(samples, [16, 84], axis=0)

    # Print the uncertainties
    print("\nParameter uncertainties:")
    for name, lower, upper in zip(param_names, uncertainties[0], uncertainties[1]):
        print(f"{name}: -{best_fit_params - lower} +{upper - best_fit_params}")

    print("a")


def draw_line(param):
    line_l = 400
    # amplitude, xo, yo, sigma_x, sigma_y, theta, offset
    #     0      1   2     3        4       5        6
    start_points = [param[1] - line_l * np.sin(param[5]), param[2] - line_l * np.cos(param[5])]
    end_points = [param[1] + line_l * np.sin(param[5]), param[2] + line_l * np.cos(param[5])]

    plt.plot([start_points[0], end_points[0]], [start_points[1], end_points[1]])


def draw_fit_image(image, fit_params_line, fit_params_peak):
    def rotated_gaussian_constant(x, y, amplitude, xo, sigma, theta):
        xo = float(xo)
        x_rot = x * np.cos(theta) - y * np.sin(theta)
        y_rot = x * np.sin(theta) + y * np.cos(theta)  # not used since constant in y
        g = amplitude * np.exp(-((x_rot - xo) ** 2) / (2 * sigma ** 2))
        return g

    def gaussian(x, y, amplitude, xo, yo, sigma_x, sigma_y, theta):
        xo = float(xo)
        yo = float(yo)
        a = (np.cos(theta) ** 2) / (2 * sigma_x ** 2) + (np.sin(theta) ** 2) / (2 * sigma_y ** 2)
        b = -(np.sin(2 * theta)) / (4 * sigma_x ** 2) + (np.sin(2 * theta)) / (4 * sigma_y ** 2)
        c = (np.sin(theta) ** 2) / (2 * sigma_x ** 2) + (np.cos(theta) ** 2) / (2 * sigma_y ** 2)
        g = amplitude * np.exp(- (a * ((x - xo) ** 2) + 2 * b * (x - xo) * (y - yo) + c * ((y - yo) ** 2)))
        return g

    x = np.linspace(1, 1024, 1024)
    y = np.linspace(1, 512, 512)
    x, y = np.meshgrid(x, y)

    fit_image = np.zeros_like(image)
    # display line fits
    for fit in fit_params_line:
        gg = rotated_gaussian_constant(x, y, fit[0], fit[1], fit[2], 0)
        fit_image = fit_image + gg

    # amplitude, xo, yo, sigma_x, sigma_y, theta, offset
    #     0      1   2     3        4       5        6
    # display peak fits
    for fit in fit_params_peak:
        gg = gaussian(x, y, fit[0], fit[1], fit[2], fit[3], fit[4], fit[5])
        fit_image = fit_image + gg

    # remove where beamstop is
    fit_image[image == 0] = 0

    fig1 = plt.figure(1)
    plt.imshow(fit_image, origin='lower', cmap='hot', vmax=500)
    plt.ylim(512, 0)
    plt.colorbar()

    fig2 = plt.figure(2)
    image_show(giwaxs_img)
    plt.colorbar()

    fig3 = plt.figure(3)
    residual_image = image - fit_image
    image_show(residual_image)
    plt.colorbar()
    plt.show()


# def save_fit_image(image, fit_params_line, fit_params_peak, image_cont):
#     def rotated_gaussian_constant(x, y, amplitude, xo, sigma, theta):
#         xo = float(xo)
#         x_rot = x * np.cos(theta) - y * np.sin(theta)
#         y_rot = x * np.sin(theta) + y * np.cos(theta)  # not used since constant in y
#         g = amplitude * np.exp(-((x_rot - xo) ** 2) / (2 * sigma ** 2))
#         return g
#
#     def gaussian(x, y, amplitude, xo, yo, sigma_x, sigma_y, theta):
#         xo = float(xo)
#         yo = float(yo)
#         a = (np.cos(theta) ** 2) / (2 * sigma_x ** 2) + (np.sin(theta) ** 2) / (2 * sigma_y ** 2)
#         b = -(np.sin(2 * theta)) / (4 * sigma_x ** 2) + (np.sin(2 * theta)) / (4 * sigma_y ** 2)
#         c = (np.sin(theta) ** 2) / (2 * sigma_x ** 2) + (np.cos(theta) ** 2) / (2 * sigma_y ** 2)
#         g = amplitude * np.exp(- (a * ((x - xo) ** 2) + 2 * b * (x - xo) * (y - yo) + c * ((y - yo) ** 2)))
#         return g
#
#     x = np.linspace(1, 1024, 1024)
#     y = np.linspace(1, 512, 512)
#     x, y = np.meshgrid(x, y)
#
#     fit_image = np.zeros_like(image)
#     # display line fits
#     for fit in fit_params_line:
#         gg = rotated_gaussian_constant(x, y, fit[0], fit[1], fit[2], 0)
#         fit_image = fit_image + gg
#
#     # amplitude, xo, yo, sigma_x, sigma_y, theta, offset
#     #     0      1   2     3        4       5        6
#     # display peak fits
#     for fit in fit_params_peak:
#         gg = gaussian(x, y, fit[0], fit[1], fit[2], fit[3], fit[4], fit[5])
#         fit_image = fit_image + gg
#
#     # remove where beamstop is
#     fit_image[image == 0] = 0
#
#     # Create subplots
#     fig, axs = plt.subplots(2, 2, figsize=(20, 12))  # Adjust the figsize as needed
#
#     # Plot for fit_image
#     im0 = axs[0, 0].imshow(fit_image, origin='lower', cmap='hot', vmax=500)
#     axs[0, 0].set_ylim(512, 0)
#     fig.colorbar(im0, ax=axs[0, 0])
#
#     # Plot for giwaxs_img (assuming giwaxs_img is defined)
#     im1 = axs[0, 1].imshow(image, cmap="gray", vmin=0, vmax=2000)
#     axs[0, 1].set_ylim(512, 0)
#     fig.colorbar(im1, ax=axs[0, 1])
#
#     # Plot for residual_image
#     residual_image = image - fit_image
#     im2 = axs[1, 0].imshow(residual_image, cmap='gray', vmin=0, vmax=2000)
#     axs[1, 0].set_ylim(512, 0)
#     fig.colorbar(im2, ax=axs[1, 0])
#
#     # Plot for images with boxes
#     im2 = axs[1, 1].imshow(image_cont, cmap='gray')
#     axs[1, 1].set_ylim(512, 0)
#     draw_boxes(boxes=boxes, ratios=boxes_ratios, limit=0.8)
#     fig.colorbar(im2, ax=axs[1, 1])
#
#     # save the image
#     plt.savefig(f'labeled_fit_data/frame_{i:03d}.png')
#     plt.close()


def a_test_set_hagen():
    data2 = H5GIWAXSDataset_h("../hagen/batch_1_labeled.h5", buffer_size=5, unskewed_polar=False)
    for i, giwaxs_img_container in enumerate(data2.iter_images()):
        giwaxs_img = giwaxs_img_container.converted_polar_image
        raw_giwaxs_img = giwaxs_img_container.raw_polar_image
        labels = giwaxs_img_container.reciprocal_labels
        boxes = labels.boxes
        confidences = labels.confidences
        boxes = np.delete(boxes, np.where(confidences < 0.9), axis=0)

        plt.figure(figsize=(10.24, 5.12))
        image_show(giwaxs_img)
        boxes_ratios = calculate_box_to_height_ratios(image=raw_giwaxs_img, boxes=boxes)
        draw_boxes(boxes=boxes, ratios=boxes_ratios, limit=0.8)
        plt.savefig(f'video_data_label_high_conf/frame_{i:03d}.png')
        plt.close()
        # plt.show()
        num_frames = i
        print(i)
    # create video
    # Parameters

    fps = 10  # Frames per second
    size = (1024, 512)  # Size of the video
    output_video = 'output_label_high_conf.mp4'  # Output video file

    # Create a video writer object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_video, fourcc, fps, size)
    # Add images to video
    for i in range(num_frames):
        img_path = f'video_data_label_high_conf/frame_{i:03d}.png'
        frame = cv2.imread(img_path)
        video.write(frame)
    # Release the video writer
    video.release()

    print("a")


def save_plot_to_mp4():
    fps = 10  # Frames per second
    size = (1000, 1000)  # Size of the video
    output_video = 'output_label_high_conf_fit.mp4'  # Output video file

    # Create a video writer object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_video, fourcc, fps, size)
    # Add images to video
    for i in range(717):
        img_path = f'video_data_label_high_conf_fit/frame_{i:03d}.png'
        frame = cv2.imread(img_path)
        video.write(frame)
    # Release the video writer
    video.release()


def histogram_e(image):
    image[image == 0] = np.nan
    image = np.transpose(image)
    num_rows = image.shape[0]
    num_bins = 1

    # Initialize an array to store histograms
    histograms = np.zeros((num_rows, num_bins))

    for i in range(num_rows):
        # Compute histogram for each row
        hist, _ = np.histogram(image[i, :], bins=num_bins, range=(0, 255))
        histograms[i, :] = hist

    # Plotting the 2D histogram as a heatmap
    plt.figure(figsize=(10, 8))
    plt.imshow(histograms, aspect='auto', extent=[0, 10, 0, num_rows], cmap='hot')
    plt.colorbar(label='Frequency')
    plt.xlabel('Pixel Value')
    plt.ylabel('Row Number')
    plt.title('2D Histogram of Image Rows')
    plt.show()
    print("a")


def average_out_boxes_over_images(boxes_time_list, image_time_list, confidences_time_list):
    def correct_boxes_boundaries(boxes):
        """
        Makes sure the box boundaies are inside the image.
        Being outside the image causes problems with parts of the code
        """
        for box in boxes:
            if box[0] < 0:
                box[0] = 0
            if box[1] < 0:
                box[1] = 0
            if box[2] > 1024:
                box[2] = 1024
            if box[3] > 512:
                box[3] = 512
        return boxes

    mask = np.zeros_like(image_time_list[0])
    # round box boundareis
    boxes_time_list_int = []
    for boxes in boxes_time_list:
        rounded_boxes = np.copy(boxes)
        rounded_boxes[:, [0, 2]] = np.floor(boxes[:, [0, 2]])
        rounded_boxes[:, [1, 3]] = np.ceil(boxes[:, [1, 3]])
        rounded_boxes = rounded_boxes.astype(int)
        rounded_boxes = correct_boxes_boundaries(rounded_boxes)
        boxes_time_list_int.append(rounded_boxes)

        """averages over all 20 images"""
        for box in rounded_boxes:
            mask[box[1]:box[3], box[0]:box[2]] += 1

    plt.imshow(mask, cmap="hot")
    plt.colorbar()
    plt.show()

    """average over 3 images"""
    avg_time = 3
    n_images = len(boxes_time_list_int)
    steps = n_images - 2 * (avg_time - 1)
    for i in range(steps):
        mask1 = np.zeros_like(image_time_list[0])
        for boxes in boxes_time_list_int[i:i + avg_time]:
            for box in boxes:
                mask1[box[1]:box[3], box[0]:box[2]] += 1
        mask1[mask1 < np.ceil(avg_time / 2)] = 0

        fig, axs = plt.subplots(3, 1, figsize=(5, 5))
        im1 = axs[0].imshow(mask1, cmap="hot")
        # fig.title(im1, str(i) + " to " + str(i+avg_time))
        fig.colorbar(im1, ax=axs[0])

        im2 = axs[1].imshow(image_time_list[i + np.ceil(avg_time / 2).astype(int)], cmap="gray", vmin=0, vmax=1000)
        fig.colorbar(im2, ax=axs[1])

        single_image = image_time_list[i + np.ceil(avg_time / 2).astype(int)]
        single_image[single_image > 1000] = 1000
        single_image[single_image < 0] = 0

        rgb_image = np.stack((single_image,) * 3, axis=-1)
        mask1_rgb = np.stack((rgb_image,) * 3, axis=-1)
        # Create a semi-transparent overlay
        # Assuming your image is RGB (without alpha channel)
        overlay = np.zeros_like(rgb_image)
        overlay_color = [255, 0, 0]  # Red color overlay
        alpha = 0.5  # Transparency factor

        overlay[mask1 != 0] = overlay_color  # Modify this to target your specific region
        image_with_overlay = rgb_image.copy()
        image_with_overlay[mask1 != 0] = image_with_overlay[mask1 != 0] * (1 - alpha) + overlay[mask1 != 0] * alpha

        im3 = axs[2].imshow(image_with_overlay)

        plt.show()

    print("a")


def radial_mean_custom_bins(image, num_bins):
    height, width = image.shape
    x, y = np.meshgrid(np.arange(width), np.arange(height))

    # Calculate the distance of each pixel from the bottom left corner
    dist = distance.cdist(np.column_stack((x.ravel(), y.ravel())), [[0, 0]], 'euclidean').reshape(height, width)

    # Maximum possible distance
    max_dist = np.hypot(width - 1, height - 1)

    # Create bins for distances
    bins = np.linspace(0, max_dist, num=num_bins + 1)
    digitized = np.digitize(dist.ravel(), bins, right=True)

    # Get image values and set weights to 0 where image value is 0
    image_values = image.ravel()
    weights = np.where(image_values != 0, image_values, 0)

    # Calculate sum and count for each bin
    sum_bins = np.bincount(digitized, weights=image.ravel(), minlength=num_bins + 1)

    # Adjust count_bins to count only non-zero pixels
    nonzero_counts = np.where(image_values != 0, 1, 0)
    count_bins = np.bincount(digitized, weights=nonzero_counts, minlength=num_bins + 1)

    # Calculate mean for each bin, now correctly ignoring zeros
    with np.errstate(divide='ignore', invalid='ignore'):
        mean_bins = sum_bins / count_bins
        mean_bins = np.nan_to_num(mean_bins)  # Convert NaNs to zero

    return mean_bins[1:]  # Exclude last bin which is empty or partial


def simplified_get_exp_metrics(truth_boxes: Tensor, predicted_boxes: Tensor, matcher) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray]:
    """
    Simplified function to match truth and predicted bounding boxes using a specified matcher.

    Args:
        truth_boxes (Tensor): Tensor of truth bounding boxes.
        predicted_boxes (Tensor): Tensor of predicted bounding boxes.
        matcher (Matcher): Matcher instance to use for matching boxes.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: Match matrix, row indices, and column indices.
    """

    # Use the matcher to match truth and predicted bounding boxes
    match_matrix, row_indices, col_indices = matcher(truth_boxes, predicted_boxes)

    return match_matrix, row_indices, col_indices


def sort_and_filter_predicted_boxes(
        truth_boxes: torch.Tensor,
        predicted_boxes: torch.Tensor,
        row_indices: List[int],
        col_indices: List[int]
) -> torch.Tensor:
    """
    Sorts and filters predicted bounding boxes based on matching with truth boxes.

    Args:
        truth_boxes (torch.Tensor): Tensor of truth bounding boxes.
        predicted_boxes (torch.Tensor): Tensor of predicted bounding boxes.
        row_indices (List[int]): Row indices from the matching process.
        col_indices (List[int]): Column indices from the matching process.

    Returns:
        torch.Tensor: Sorted and filtered tensor of predicted bounding boxes.
    """

    # Create a placeholder for sorted and filtered predicted boxes
    sorted_predicted_boxes = torch.zeros_like(truth_boxes)

    # Iterate over the matched indices and sort the predicted boxes
    for i, row_idx in enumerate(row_indices):
        col_idx = col_indices[i]
        sorted_predicted_boxes[row_idx] = predicted_boxes[col_idx]

    return sorted_predicted_boxes


def draw_marks_on_mids(peak_param, color='r', alpha=1.0):
    if peak_param.any():
        for k in peak_param:
            plt.scatter(k[1], k[2], color=color, marker='x', label='Point', s=50, alpha=alpha)


def draw_lines_on_mids(line_param, color="r"):
    if line_param.any():
        for k in line_param:
            plt.axvline(x=k[2], color=color, linestyle='--', label='Vertical Line')


def draw_sigmas(peak_param, n_sigmas=1, color=(1, 0, 0)):
    axis = plt.gca()
    for fit_i in peak_param:
        # for n in range(n_sigmas):

        # amplitude, xo, yo, sigma_x, sigma_y, theta, offset
        #     0      1   2     3        4       5        6
        ellipse = Ellipse(xy=(fit_i[1], fit_i[2]), width=2 * fit_i[3], height=2 * fit_i[4],
                          angle=np.degrees(-fit_i[5]),
                          edgecolor=color, fc='None', lw=2, ls='--')
        axis.add_patch(ellipse)

        # ellipse = Ellipse(xy=(fit_i[1], fit_i[2]), width=4 * fit_i[3], height=4 * fit_i[4],
        #                   angle=np.degrees(-fit_i[5]),
        #                   edgecolor=color, fc='None', lw=2, ls='--')
        # axis.add_patch(ellipse)


def postprocessing_images(predictions, scores):
    min_score = 0.01

    # Initialize postprocessing steps with desired parameters
    standard_postprocessing = StandardPostprocessing(nms_level=0.01, score_level=min_score)
    small_q_filter = SmallQFilter(min_q_pix=50.0)
    merge_boxes_postprocessing = MergeBoxesPostprocessing(
        min_score=min_score)  # , min_iou=0.5, max_q=5.0, mode='mean-quantile',
    # quantile=0.8)

    # Combine postprocessing steps into a pipeline
    pipeline = PostprocessingPipeline(
        standard_postprocessing,
        small_q_filter,
        merge_boxes_postprocessing
    )

    refined_predictions, refined_scores = pipeline(predictions, scores)

    return refined_predictions, refined_scores


def compute_iou_matrix(boxes1, boxes2):
    """
    Compute a matrix of IoU values between two sets of boxes.

    Parameters:
    boxes1 -- an array of shape (N, 4) of boxes (xmin, ymin, xmax, ymax)
    boxes2 -- an array of shape (M, 4) of boxes (xmin, ymin, xmax, ymax)

    Returns:
    iou_matrix -- a NumPy array of shape (N, M) containing IoU values.
    """
    # Expand the boxes to compare every box in boxes1 with every box in boxes2
    boxes1 = np.expand_dims(boxes1, axis=1)  # Shape becomes (N, 1, 4)
    boxes2 = np.expand_dims(boxes2, axis=0)  # Shape becomes (1, M, 4)

    # Compute intersection coordinates
    inter_max_xy = np.minimum(boxes1[..., 2:], boxes2[..., 2:])
    inter_min_xy = np.maximum(boxes1[..., :2], boxes2[..., :2])
    inter_wh = np.maximum(inter_max_xy - inter_min_xy, 0)

    # Compute intersection and union areas
    inter_area = inter_wh[..., 0] * inter_wh[..., 1]
    box1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    box2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])
    union_area = box1_area + box2_area - inter_area

    # Compute IoU values
    iou_matrix = inter_area / union_area

    return iou_matrix


def find_corresponding_boxes(frame1_boxes, frame2_boxes, iou_threshold=0.5):
    """
    Find the corresponding boxes in two consecutive frames using NumPy for efficiency.

    Parameters:
    frame1_boxes -- NumPy array of shape (N, 4) for the first frame
    frame2_boxes -- NumPy array of shape (M, 4) for the second frame
    iou_threshold -- threshold above which boxes are considered corresponding

    Returns:
    matches -- list of tuples (index_frame1, index_frame2) indicating corresponding boxes
    """
    iou_matrix = compute_iou_matrix(np.array(frame1_boxes), np.array(frame2_boxes))
    matches = []

    # Iterate over each row (box in frame1), finding the column (box in frame2) with the highest IoU
    for i, row in enumerate(iou_matrix):
        max_iou_idx = np.argmax(row)
        if row[max_iou_idx] > iou_threshold:
            matches.append((i, max_iou_idx))

    return matches


def postprocessing_images2(predictions, scores):
    min_score = 0.01

    # Initialize postprocessing steps with desired parameters
    standard_postprocessing = StandardPostprocessing(nms_level=0.01, score_level=min_score)
    small_q_filter = SmallQFilter(min_q_pix=50.0)
    merge_boxes_postprocessing = MergeBoxesPostprocessing(
        min_score=min_score)  # , min_iou=0.5, max_q=5.0, mode='mean-quantile',
    # quantile=0.8)

    # Combine postprocessing steps into a pipeline
    pipeline = PostprocessingPipeline(
        standard_postprocessing,
        small_q_filter,
        merge_boxes_postprocessing
    )

    refined_predictions, refined_scores = pipeline(predictions, scores)

    return refined_predictions, refined_scores


def calculate_bounding_boxes(peaks):
    """
    Calculate bounding boxes for multiple 2D Gaussian peaks ignoring the theta.

    Args:
    peaks (numpy array): An array with rows representing parameters of each Gaussian peak.
                         The columns are ["amp", "x0", "y0", "sigma_x", "sigma_y", "theta", "offset", "id"].

    Returns:
    numpy array: An array where each row contains the bounding box [x_min, y_min, x_max, y_max] of each peak.
    """
    # Factor to convert sigma to FWHM
    fwhm_factor = 2 * np.sqrt(2 * np.log(2))

    # Initialize the output array
    bounding_boxes = np.zeros((peaks.shape[0], 4))  # Each row will have [x_min, y_min, x_max, y_max]

    # Calculate bounding box for each peak
    for i, peak in enumerate(peaks):
        sigma_x, sigma_y = peak[3], peak[4]
        fwhm_x = sigma_x * fwhm_factor
        fwhm_y = sigma_y * fwhm_factor

        x0, y0 = peak[1], peak[2]

        x_min = x0 - fwhm_x / 2
        x_max = x0 + fwhm_x / 2
        y_min = y0 - fwhm_y / 2
        y_max = y0 + fwhm_y / 2

        bounding_boxes[i] = [x_min, y_min, x_max, y_max]

    return bounding_boxes



if __name__ == '__main__':
    # a_test_set_hagen()
    with open('gixd-fit/object_arrays.pkl', 'rb') as file:
        own_predictions = pickle.load(file)
    data = H5GIWAXSDataset("hagen/35_flipped_quazi_labeled_512.h5", buffer_size=5, unskewed_polar=True,
                           contrast_correction=True)
    # data = H5GIWAXSDataset("35_flipped.h5", buffer_size=5, unskewed_polar=True)
    # data = H5GIWAXSDataset("revised_25.h5", buffer_size=5, unskewed_polar=True)
    # save_plot_to_mp4()
    # data = H5GIWAXSDataset_h("hagen/batch_1_labeled.h5", buffer_size=5, unskewed_polar=False)

    image_time_list = []
    boxes_time_list = []
    confidences_time_list = []

    peak_param_p_relative_eval_list = np.zeros((1, 8))
    peak_param_p_eval_list = np.zeros((1, 16))

    # load prediction model
    fp_detector = SimpleBinaryClassifier_12()
    fp_detector.load_state_dict(torch.load("fp_detector_12_better.pth"))
    fp_detector.eval()

    images_np = []
    boxes_np = []
    scores_np = []

    dif_pred_lab_x_list = []
    dif_pred_lab_y_list = []

    dif_fit_lab_x_list = []
    dif_fit_lab_y_list = []

    for i, giwaxs_img_container in enumerate(data.iter_images()):

        giwaxs_img = giwaxs_img_container.converted_polar_image
        raw_giwaxs_img = giwaxs_img_container.raw_polar_image
        img_name = giwaxs_img_container.polar_labels.img_name
        boxes_labeled = giwaxs_img_container.polar_labels.boxes
        scores_labeled = giwaxs_img_container.polar_labels.confidences

        # filter out low confidence boxes
        boxes_predicted = np.delete(giwaxs_img_container.reciprocal_labels.predictions,
                                    np.where(giwaxs_img_container.reciprocal_labels.scores < 0), axis=0)

        scores_predicted = np.delete(giwaxs_img_container.reciprocal_labels.scores,
                                     np.where(giwaxs_img_container.reciprocal_labels.scores < 0), axis=0)

        # boxes_predicted = own_predictions[i]
        # boxes_predicted = np.array(boxes_predicted, dtype=float)

        # print(boxes_predicted)

        # plt.figure(figsize=(10, 10/2))
        # plt.imshow(giwaxs_img, cmap='gray')
        # plt.axis('off')
        # plt.savefig(str(i) + "_example_fig.pdf", bbox_inches='tight')
        # draw_boxes_basic(boxes_labeled)
        # draw_boxes_basic(boxes_predicted, color='b')
        # plt.savefig(str(i) + "_example_fig_boxes.pdf", bbox_inches='tight')
        # plt.show()
        # plt.close()

        """MATCH THE IDs"""
        # matcher = QMatcher(1., rel=True, min_iou=0.1)
        matcher = QMatcher(10., rel=False, min_iou=0.01)

        match_matrix, row_indices, col_indices = matcher(torch.from_numpy(boxes_labeled),
                                                         torch.from_numpy(boxes_predicted))

        boxes_predicted_sorted = sort_and_filter_predicted_boxes(torch.from_numpy(boxes_labeled),
                                                                 torch.from_numpy(boxes_predicted),
                                                                 row_indices, col_indices)

        boxes_predicted_sorted = np.array(boxes_predicted_sorted)

        """MATCH LABELED BOXES TO PREDICTED BOXES"""
        # Initialize arrays to hold IDs for both labeled and predicted boxes, setting initial IDs.
        # This time, every box gets an ID regardless of being matched or not.
        labeled_ids_with_matches = np.arange(1, len(boxes_labeled) + 1)
        predicted_ids_with_matches = np.arange(1, len(boxes_predicted) + 1)

        # Reset IDs for predicted to ensure they start after the last labeled ID to avoid initial overlap
        predicted_ids_with_matches += labeled_ids_with_matches[-1]

        # Go through the matches and assign the same ID to both matched labeled and predicted boxes
        for row_idx, col_idx in zip(row_indices, col_indices):
            # Use the labeled box's ID for both to ensure matched pairs share the same ID
            predicted_ids_with_matches[col_idx] = labeled_ids_with_matches[row_idx]

        boxes_labeled_ids = np.concatenate([boxes_labeled, labeled_ids_with_matches.reshape(-1, 1)], axis=1)
        boxes_predicted_ids = np.concatenate([boxes_predicted, predicted_ids_with_matches.reshape(-1, 1)], axis=1)


        def match_ids(array1, array2):
            ids1 = array1[:, -1]
            ids2 = array2[:, -1]

            # Find common IDs
            common_ids = np.intersect1d(ids1, ids2)

            # Filter arrays based on common IDs
            filtered_array1 = np.array([row for row in array1 if row[-1] in common_ids])
            filtered_array2 = np.array([row for row in array2 if row[-1] in common_ids])

            # Ensure the arrays are sorted by ID to align them for difference calculation
            filtered_array1 = filtered_array1[np.argsort(filtered_array1[:, -1])]
            filtered_array2 = filtered_array2[np.argsort(filtered_array2[:, -1])]

            # Calculate differences
            array_matched = np.concatenate([filtered_array1[:, :-1], filtered_array2], axis=1)

            return array_matched

        """
        DO THE CALCULATIONS
        """

        line_params_l, line_params_err_l, line_errors_l, peak_param_l, peak_param_err_l, peak_errors_l = \
            FitBoxes(raw_giwaxs_img, giwaxs_img, img_name, boxes_labeled_ids,
                     plot_fits=False, dataframe=False).fit()

        line_params_p, line_params_err_p, line_errors_p, peak_param_p, peak_param_err_p, peak_errors_p = \
            FitBoxes(raw_giwaxs_img, giwaxs_img, img_name, boxes_predicted_ids,
                     plot_fits=False, dataframe=False).fit()

        def calculate_mids_and_concatenate_ids(boxes):
            """Calculate the midpoints and concatenate with the ID column."""
            x_mids = ((boxes[:, 0] + boxes[:, 2]) / 2).reshape(-1, 1)
            y_mids = ((boxes[:, 1] + boxes[:, 3]) / 2).reshape(-1, 1)
            ids = boxes[:, -1].reshape(-1, 1)
            return np.concatenate([x_mids, y_mids, ids], axis=1)

        def filter_and_sort_by_ids(data, keeper_ids):
            """Filter out entries that don't match keeper_ids and sort by IDs."""
            filtered_data = data[np.isin(data[:, -1], keeper_ids)]
            return filtered_data[np.argsort(filtered_data[:, -1])]

        labeled_mids = calculate_mids_and_concatenate_ids(boxes_labeled_ids)
        predicted_mids = calculate_mids_and_concatenate_ids(boxes_predicted_ids)

        keeper_ids = np.intersect1d(labeled_mids[:, -1], predicted_mids[:, -1])

        labeled_mids_filtered_sorted = filter_and_sort_by_ids(labeled_mids, keeper_ids)
        predicted_mids_filtered_sorted = filter_and_sort_by_ids(predicted_mids, keeper_ids)

        # process the fit mids
        print(line_params_p)
        print(peak_param_p)
        if (line_params_p.size > 0) and (peak_param_p.size > 0):
            p_x_mids = np.concatenate((line_params_p[:, (2, 5)], peak_param_p[:, (1, 7)]), axis=0)
        elif (peak_param_p.size > 0):
            p_x_mids = peak_param_p[:, (1, 7)]
        p_y_mids = peak_param_p[:, (2, 7)]

        p_x_mids_filtered = filter_and_sort_by_ids(p_x_mids, keeper_ids)

        dif_pred_lab = abs(labeled_mids_filtered_sorted[:, 0] - predicted_mids_filtered_sorted[:, 0])
        dif_fit_lab = abs(labeled_mids_filtered_sorted[:, 0] - p_x_mids_filtered[:, 0])

        print("fit x: " + str(np.mean(dif_fit_lab)))
        print("pred x: " + str(np.mean(dif_pred_lab)))

        dif_pred_lab_x_list.append(dif_pred_lab)
        dif_fit_lab_x_list.append(dif_fit_lab)

        keeper_ids_y = np.intersect1d(labeled_mids[:, -1], p_y_mids[:, -1])

        labeled_mids_filtered_sorted = filter_and_sort_by_ids(labeled_mids, keeper_ids_y)
        predicted_mids_filtered_sorted = filter_and_sort_by_ids(predicted_mids, keeper_ids_y)
        p_y_mids_filtered = filter_and_sort_by_ids(p_y_mids, keeper_ids_y)

        dif_pred_lab_y = abs(labeled_mids_filtered_sorted[:, 1] - predicted_mids_filtered_sorted[:, 1])
        dif_fit_lab_y = abs(labeled_mids_filtered_sorted[:, 1] - p_y_mids_filtered[:, 0])

        print("fit y: " + str(np.mean(dif_fit_lab_y)))
        print("pred y: " + str(np.mean(dif_pred_lab_y)))

        dif_pred_lab_y_list.append(dif_pred_lab_y)
        dif_fit_lab_y_list.append(dif_fit_lab_y)

        boxes_fit = calculate_bounding_boxes(peak_param_p)

        plt.figure(figsize=(10, 10/2))
        plt.imshow(giwaxs_img, cmap='gray')
        plt.axis('off')
        draw_boxes_basic(boxes_labeled, color='r', linewidth=3)
        draw_boxes_basic(boxes_predicted, color='b', linewidth=3)
        draw_boxes_basic(boxes_fit, color='g', linewidth=3)
        plt.show()
        plt.close()


    print("final")
    print("pred x: " + str(np.mean(np.concatenate(dif_pred_lab_x_list))))
    print("fit x: " + str(np.mean(np.concatenate(dif_fit_lab_x_list))))
    print("pred y: " + str(np.mean(np.concatenate(dif_pred_lab_y_list))))
    print("fit y: " + str(np.mean(np.concatenate(dif_fit_lab_y_list))))
