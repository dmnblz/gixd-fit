import numpy as np
import pickle
from Dataset.dataset.dataset import H5GIWAXSDataset
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
import torch.optim as optim
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


def save_fit_image(image, fit_params_line, fit_params_peak, image_cont):
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

    # Create subplots
    fig, axs = plt.subplots(2, 2, figsize=(20, 12))  # Adjust the figsize as needed

    # Plot for fit_image
    im0 = axs[0, 0].imshow(fit_image, origin='lower', cmap='hot', vmax=500)
    axs[0, 0].set_ylim(512, 0)
    fig.colorbar(im0, ax=axs[0, 0])

    # Plot for giwaxs_img (assuming giwaxs_img is defined)
    im1 = axs[0, 1].imshow(image, cmap="gray", vmin=0, vmax=2000)
    axs[0, 1].set_ylim(512, 0)
    fig.colorbar(im1, ax=axs[0, 1])

    # Plot for residual_image
    residual_image = image - fit_image
    im2 = axs[1, 0].imshow(residual_image, cmap='gray', vmin=0, vmax=2000)
    axs[1, 0].set_ylim(512, 0)
    fig.colorbar(im2, ax=axs[1, 0])

    # Plot for images with boxes
    im2 = axs[1, 1].imshow(image_cont, cmap='gray')
    axs[1, 1].set_ylim(512, 0)
    draw_boxes(boxes=boxes, ratios=boxes_ratios, limit=0.8)
    fig.colorbar(im2, ax=axs[1, 1])

    # save the image
    plt.savefig(f'labeled_fit_data/frame_{i:03d}.png')
    plt.close()


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
    steps = n_images - 2 * (avg_time-1)
    for i in range(steps):
        mask1 = np.zeros_like(image_time_list[0])
        for boxes in boxes_time_list_int[i:i+avg_time]:
            for box in boxes:
                mask1[box[1]:box[3], box[0]:box[2]] += 1
        mask1[mask1 < np.ceil(avg_time/2)] = 0

        fig, axs = plt.subplots(3, 1, figsize=(5, 5))
        im1 = axs[0].imshow(mask1, cmap="hot")
        # fig.title(im1, str(i) + " to " + str(i+avg_time))
        fig.colorbar(im1, ax=axs[0])

        im2 = axs[1].imshow(image_time_list[i+np.ceil(avg_time/2).astype(int)], cmap="gray", vmin=0, vmax=1000)
        fig.colorbar(im2, ax=axs[1])

        single_image = image_time_list[i+np.ceil(avg_time/2).astype(int)]
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
    max_dist = np.hypot(width-1, height-1)

    # Create bins for distances
    bins = np.linspace(0, max_dist, num=num_bins+1)
    digitized = np.digitize(dist.ravel(), bins, right=True)

    # Get image values and set weights to 0 where image value is 0
    image_values = image.ravel()
    weights = np.where(image_values != 0, image_values, 0)

    # Calculate sum and count for each bin
    sum_bins = np.bincount(digitized, weights=image.ravel(), minlength=num_bins+1)

    # Adjust count_bins to count only non-zero pixels
    nonzero_counts = np.where(image_values != 0, 1, 0)
    count_bins = np.bincount(digitized, weights=nonzero_counts, minlength=num_bins + 1)

    # Calculate mean for each bin, now correctly ignoring zeros
    with np.errstate(divide='ignore', invalid='ignore'):
        mean_bins = sum_bins / count_bins
        mean_bins = np.nan_to_num(mean_bins)  # Convert NaNs to zero

    return mean_bins[1:]  # Exclude last bin which is empty or partial


def compare_intensities(raw_reciprocal_image, raw_polar_image):
    # calculate the intensity
    raw_radial_mean = radial_mean_custom_bins(raw_reciprocal_image, num_bins=1024)

    masked_array = np.ma.masked_equal(raw_polar_image, 0)
    raw_polar_mean = np.mean(masked_array, axis=0)

    max_intensity = np.max([np.max(raw_radial_mean), np.max(raw_polar_mean)]) * 1.1
    min_intensity = np.min([np.min(raw_radial_mean), np.min(raw_polar_mean)]) * 1.1

    """PLOT"""
    # Create figure and specify grid layout
    fig = plt.figure(figsize=(10, 9))
    gs = gridspec.GridSpec(3, 2, width_ratios=[2, 1], height_ratios=[1, 1, 1])

    # Create subplots
    ax1 = fig.add_subplot(gs[0, 0])  # Top left
    ax2 = fig.add_subplot(gs[1, 0])  # Bottom left
    ax3 = fig.add_subplot(gs[0, 1])  # Top right
    ax4 = fig.add_subplot(gs[1, 1])  # Bottom right
    ax5 = fig.add_subplot(gs[2, 0])
    ax6 = fig.add_subplot(gs[2, 1])

    # Example plotting commands
    # You can replace these with your actual plotting code
    ax1.plot(raw_radial_mean)
    ax1.set_title("Radial mean")
    ax1.set_ylim(min_intensity, max_intensity)
    ax2.plot(raw_polar_mean)
    ax2.set_title("Polar mean")
    ax2.set_ylim(min_intensity, max_intensity)
    ax3.imshow(raw_giwaxs_reciprocal_img, cmap="gray", vmin=0, vmax=500)
    ax3.set_title("reciprocal image")
    ax4.imshow(raw_giwaxs_img, cmap="gray", vmin=0, vmax=500)
    ax4.set_title("polar image")
    ax5.plot(raw_radial_mean - raw_polar_mean)
    ax5.set_title("difference")
    ax5.set_ylim(min_intensity, max_intensity)
    ax6.remove()

    # Adjust layout
    plt.tight_layout()

    # Show plot
    plt.show()


def compare_intensities2(raw_reciprocal_image):

    im_polar = calc_polar_image(raw_reciprocal_image)
    im_quasipolar = calc_quazipolar_image(raw_reciprocal_image)

    im_polar_mask = im_polar != 0
    im_polar_mean_values = np.sum(im_polar, axis=0) / np.sum(im_polar_mask, axis=0)

    im_quasipolar_mask = im_quasipolar != 0
    im_quasipolar_mean_values = np.sum(im_quasipolar, axis=0) / np.sum(im_quasipolar_mask, axis=0)

    # calculate the intensity
    raw_radial_mean = radial_mean_custom_bins(raw_reciprocal_image, num_bins=1024)


    max_intensity = np.max([np.max(raw_radial_mean), np.max(np.nan_to_num(im_polar_mean_values))]) * 1.1
    min_intensity = np.min([np.min(raw_radial_mean), np.min(np.nan_to_num(im_polar_mean_values))]) * 1.1

    fig = plt.figure(figsize=(10, 7))
    plt.plot(raw_radial_mean, label="Q space", lw=3)
    plt.plot(im_polar_mean_values, label="polar space", lw=2)
    plt.plot(im_quasipolar_mean_values, label="quasipolar space", lw=1)
    plt.legend()
    plt.title(img_name)
    print(img_name)
    plt.show()

    print("radial mean: " + str(np.sum(raw_radial_mean)))
    print("polar mean: " + str(np.sum(np.nan_to_num(im_polar_mean_values))))
    print("quasipolar mean: " + str(np.sum(np.nan_to_num(im_quasipolar_mean_values))))

    """PLOT POLAR"""
    # Create figure and specify grid layout
    fig = plt.figure(figsize=(10, 9))
    gs = gridspec.GridSpec(3, 2, width_ratios=[2, 1], height_ratios=[1, 1, 1])

    # Create subplots
    ax1 = fig.add_subplot(gs[0, 0])  # Top left
    ax2 = fig.add_subplot(gs[1, 0])  # Bottom left
    ax3 = fig.add_subplot(gs[0, 1])  # Top right
    ax4 = fig.add_subplot(gs[1, 1])  # Bottom right
    ax5 = fig.add_subplot(gs[2, 0])
    ax6 = fig.add_subplot(gs[2, 1])

    # Example plotting commands
    # You can replace these with your actual plotting code
    ax1.plot(raw_radial_mean)
    ax1.set_title("Radial mean")
    ax1.set_ylim(min_intensity, max_intensity)
    ax1.grid()
    ax2.plot(im_polar_mean_values)
    ax2.set_title("Polar mean")
    ax2.set_ylim(min_intensity, max_intensity)
    ax2.grid()
    ax3.imshow(raw_giwaxs_reciprocal_img, cmap="inferno", vmin=0, vmax=500)
    ax3.set_title("reciprocal image")
    ax4.imshow(im_polar, cmap="inferno", vmin=0, vmax=500)
    ax4.set_title("polar image")
    ax5.plot(raw_radial_mean - im_polar_mean_values)
    ax5.set_title("difference")
    ax5.set_ylim(-max_intensity/2, max_intensity)
    ax6.remove()

    # Adjust layout
    plt.tight_layout()

    # Show plot
    plt.show()

    """PLOT QUASIPOLAR"""
    # Create figure and specify grid layout
    fig = plt.figure(figsize=(10, 9))
    gs = gridspec.GridSpec(3, 2, width_ratios=[2, 1], height_ratios=[1, 1, 1])

    # Create subplots
    ax1 = fig.add_subplot(gs[0, 0])  # Top left
    ax2 = fig.add_subplot(gs[1, 0])  # Bottom left
    ax3 = fig.add_subplot(gs[0, 1])  # Top right
    ax4 = fig.add_subplot(gs[1, 1])  # Bottom right
    ax5 = fig.add_subplot(gs[2, 0])
    ax6 = fig.add_subplot(gs[2, 1])

    # Example plotting commands
    # You can replace these with your actual plotting code
    ax1.plot(raw_radial_mean)
    ax1.set_title("Radial mean")
    ax1.set_ylim(min_intensity, max_intensity)
    ax1.grid()
    ax2.plot(im_quasipolar_mean_values)
    ax2.set_title("Quasipolar mean")
    ax2.set_ylim(min_intensity, max_intensity)
    ax2.grid()
    ax3.imshow(raw_reciprocal_image, cmap="inferno", vmin=0, vmax=500)
    ax3.set_title("reciprocal image")
    ax4.imshow(im_quasipolar, cmap="inferno", vmin=0, vmax=500)
    ax4.set_title("quasipolar image")
    ax5.plot(raw_radial_mean - im_quasipolar_mean_values)
    ax5.set_title("difference")
    ax5.set_ylim(-max_intensity, max_intensity)
    ax6.remove()

    # Adjust layout
    plt.tight_layout()

    # Show plot
    plt.show()


if __name__ == '__main__':
    # a_test_set_hagen()
    data = H5GIWAXSDataset("../35_flipped.h5", buffer_size=5, unskewed_polar=True)
    # save_plot_to_mp4()
    # data = H5GIWAXSDataset_h("hagen/batch_1_labeled.h5", buffer_size=5, unskewed_polar=False)

    image_time_list = []
    boxes_time_list = []
    confidences_time_list = []

    for i, giwaxs_img_container in enumerate(data.iter_images()):

        line_p, line_e, peak_p = FitBoxes(giwaxs_img_container, plot_fits=True).fit()

        print("a")
        continue
        # if i == 20:
        #     break
        #
        # image_time_list.append(giwaxs_img_container.raw_polar_image)
        # boxes_time_list.append(giwaxs_img_container.reciprocal_labels.boxes)
        # confidences_time_list.append(giwaxs_img_container.reciprocal_labels.confidences)
        # continue

        giwaxs_img = giwaxs_img_container.converted_polar_image
        raw_giwaxs_img = giwaxs_img_container.raw_polar_image
        raw_giwaxs_reciprocal_img = giwaxs_img_container.raw_reciprocal

        # plt.imshow(raw_giwaxs_reciprocal_img, vmin=0, vmax=500, cmap="inferno")
        # plt.ylim([0, raw_giwaxs_reciprocal_img.shape[0]])
        # plt.show()
        #
        # im_polar_test = calc_polar_image(raw_giwaxs_reciprocal_img)
        # plt.imshow(im_polar_test, vmin=0, vmax=500, cmap="inferno")
        # plt.show()
        #
        # plt.imshow(raw_giwaxs_img, vmin=0, vmax=500, cmap="inferno")
        # plt.show()

        # labels = giwaxs_img_container.reciprocal_labels
        labels = giwaxs_img_container.polar_labels
        fits = giwaxs_img_container.fits
        boxes = labels.boxes
        img_name = labels.img_name

        # compare_intensities2(raw_reciprocal_image=raw_giwaxs_reciprocal_img)

        # get the image name
        # drop confidences
        # boxes = np.delete(labels.boxes, np.where(labels.confidences < 0.9), axis=0)
        # calculate ratios
        boxes_ratios = calculate_box_to_height_ratios(image=raw_giwaxs_img, boxes=boxes)

        # show whole image
        # image_show(giwaxs_img)
        # plt.title(img_name)
        # draw_boxes(boxes=boxes, ratios=boxes_ratios, limit=0.8)
        # draw_boxes_ratios(boxes=boxes, ratios=boxes_ratios, limit=0.8)
        # draw_enumerate_boxes(boxes=boxes, ratios=boxes_ratios, limit=0.8)

        # plt.show()

        """CLUSTERING"""
        start_time = time.time()
        # groups peaks boxes togehter into a big box. for better fitting
        (peak_cluster_boxes, peak_boxes_inside_clusters,
         line_cluster_boxes, line_boxes_inside_clusters,
         peak_boxes, line_boxes, image_line_data_y_flat) = cluster_peak_boxes2(raw_giwaxs_img, boxes, 7, 1)
        end_time = time.time()
        print(f"The clustering process took {end_time - start_time:.4f} seconds to complete.")

        """LINE FIT"""
        start_time = time.time()
        fit_params_line = []
        for cluster, boxes_in_cluster in zip(line_cluster_boxes, line_boxes_inside_clusters):
            fit_params_line.extend(N1DGaussian(image_line_data_y_flat, boxes_in_cluster, cluster).fit_n_1d_gaussians())
        end_time = time.time()
        duration = end_time - start_time
        print(f"The line fitting process took {duration:.4f} seconds to complete.")

        plt.show()

        """2D PEAKS FIT"""
        peak_param_lst = []
        errors_lst = []
        # n-2d-gaussian on every peak cluster
        start_time_peak = time.time()
        for c_b, b_in_c in zip(enumerate(peak_cluster_boxes), peak_boxes_inside_clusters):
            # count the round
            print(str(c_b[0] + 1) + "/" + str(len(peak_cluster_boxes)) + ", n(gaussians): " + str(len(b_in_c)))
            fit_func = LMFunctions(raw_giwaxs_img, b_in_c, c_b[1], plot_result=False)
            best_fit_param, errors = fit_func.fit_n_2d_gaussian()
            peak_param_lst.extend(best_fit_param)
            errors_lst.append(errors)

        end_time_peak = time.time()
        duration_peak = end_time_peak - start_time_peak
        print(f"The peak fitting process took {duration_peak:.4f} seconds to complete.")
        peak_params = np.array(peak_param_lst)
        if errors_lst:
            df_errors = pd.DataFrame(np.array(errors_lst))
            df_errors.columns = ["RMSE", "MAE", "MAPE", "R^2", "R^2_adj", "Q^2", "n(pixel)", "gaussians"]
            # amplitude, xo, yo, sigma_x, sigma_y, theta, offset
            #     0      1   2     3        4       5        6

        plot_peak = False
        if plot_peak:
            if peak_param_lst:
                # plot_fit_image(raw_giwaxs_img, fit_results)
                fig, ax = plt.subplots()
                image_show(giwaxs_img)
                plt.title(img_name)
                draw_marks_on_peaks(peak_params[:, 1:3])
                # draw_simga_ellipses(peak_params, ax)
                draw_boxes(boxes=boxes, ratios=boxes_ratios, limit=0.8)
                draw_groupboxes(peak_cluster_boxes)
                plt.show()

        # display fits
        fit_params_line = np.array(fit_params_line)
        # draw_fit_image(raw_giwaxs_img, fit_params_line, peak_param_lst)
        save_fit_image(raw_giwaxs_img, fit_params_line, peak_param_lst, giwaxs_img)

    # average_out_boxes_over_images(boxes_time_list, image_time_list, confidences_time_list)
