import numpy as np
import lmfit
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Ellipse
import time
from scipy.ndimage import label
from cv2 import GaussianBlur
import pandas as pd
import multiprocessing
from gauss_func_base import OneDGaussians, TwoDGaussians
from datetime import datetime
import json


def json_config_laoder(config_path):

    with open(config_path) as config_file:
        config = json.load(config_file)

    setting1 = config


def plot_error_calc_2d(data, data_fit, fit_param, boxes, abs_error, square_error, abs_error_y_flat,
                       square_error_y_flat,
                       abs_error_x_lst, square_error_x_lst, abs_error_y_lst, square_error_y_lst, mean_square_error,
                       root_mean_square_error, mean_abs_error, mean_abs_perc_error, r_sqr, r_sqr_adj, q_sqr,
                       error_type='abs'):
    """
    Plots the calculated errors, there are two types of errors to plot 'abs' or 'square'
    """
    if error_type == 'abs':
        p_err_img = abs_error
        p_err_y = abs_error_y_lst
        p_err_x = abs_error_x_lst
        p_err_flat_y = abs_error_y_flat
        p_err_name = 'Absolute'
    else:
        p_err_img = square_error
        p_err_y = square_error_y_lst
        p_err_x = square_error_x_lst
        p_err_flat_y = square_error_y_flat
        p_err_name = 'Square'

    # Find the min and max values of the 2D data for determining the colorbar limits
    min_error = np.min(p_err_img)
    max_error = np.max(p_err_img)

    min_data = np.min(data)
    max_data = np.max(data)
    vmax = np.nanpercentile(data, 99.9)

    img_ratio = data.shape[0] / data.shape[1]

    # Visualization
    fig, axs = plt.subplots(2, 4, figsize=(15, 6), gridspec_kw={'width_ratios': [1, 1, 1, 1]})

    # Original Data
    data_img = axs[0, 0].imshow(data, cmap='viridis', aspect='auto', vmin=min_data, vmax=vmax)
    axs[0, 0].set_title('Data')
    axs[0, 0].set_xlabel('Column Index')
    axs[0, 0].set_ylabel('Row Index')
    # draw in the prediction boxes
    for br in boxes:
        x__ = br[0]
        y__ = br[3]
        x_width__ = br[2] - br[0]
        y_width__ = br[3] - br[1]
        rect = patches.Rectangle((x__, y__), x_width__, -y_width__, linewidth=1, edgecolor='r', facecolor='none')
        axs[0, 0].add_patch(rect)

    # create color bar
    colorbar_ax = fig.add_axes([0.05, 0.45, 0.4, 0.03])  # Position for the color bar (x, y, width, height)
    fig.colorbar(data_img, cax=colorbar_ax, orientation='horizontal')

    # Fit Data
    fit_data_img = axs[0, 1].imshow(data_fit, cmap='viridis', aspect='auto', vmin=min_data, vmax=vmax)
    axs[0, 1].set_title('Fit')
    axs[0, 1].set_xlabel('Column Index')
    axs[0, 1].set_ylabel('Row Index')
    # plots the fits sigmas onto the image
    for fit_i in fit_param:
        # amplitude, xo, yo, sigma_x, sigma_y, theta, offset
        #     0      1   2     3        4       5        6
        axs[0, 1].scatter(fit_i[1], fit_i[2], color='red', marker='x', label='Point')
        ellipse = Ellipse(xy=(fit_i[1], fit_i[2]), width=2 * fit_i[3], height=2 * fit_i[4],
                          angle=float(np.degrees(-fit_i[5])),
                          edgecolor=(1, 0, 0), fc='None', lw=2, ls='--')
        axs[0, 1].add_patch(ellipse)
        ellipse = Ellipse(xy=(fit_i[1], fit_i[2]), width=4 * fit_i[3], height=4 * fit_i[4],
                          angle=float(np.degrees(-fit_i[5])),
                          edgecolor=(0.5, 0, 0), fc='None', lw=2, ls='--')
        axs[0, 1].add_patch(ellipse)

    # Error 2D
    axs[0, 2].imshow(p_err_img, cmap='viridis', aspect='auto', vmin=min_data, vmax=vmax)
    axs[0, 2].set_title('2D ' + p_err_name + ' Error')
    axs[0, 2].set_xlabel('')  # Remove x axis label
    axs[0, 2].set_xticks([])  # Remove x axis tics
    axs[0, 2].set_ylabel('Row Index')
    subtitle_text = "MSE: " + str(round(mean_square_error, 4))
    axs[0, 2].text(0.5, 0.92, subtitle_text, ha='center', va='center',
                   transform=axs[0, 2].transAxes, fontsize=12, color="r")

    # Error y-direction (mean per column)
    axs[1, 2].plot(p_err_y)
    axs[1, 2].set_title('Mean ' + p_err_name + ' Error per Column')
    axs[1, 2].set_xlabel('Column Index')
    axs[1, 2].set_ylabel('Error')
    axs[1, 2].set_ylim([min_error, max_error])

    # Error x-direction (mean per row)
    axs[0, 3].plot(p_err_x, range(len(p_err_x)))
    axs[0, 3].set_title('Mean ' + p_err_name + ' Error per Row')
    axs[0, 3].set_xlabel('Error')
    axs[0, 3].set_ylabel('Row Index')
    axs[0, 3].set_xlim([min_error, max_error])

    # Error y_flat
    axs[1, 3].plot(p_err_flat_y)
    axs[1, 3].set_title('Mean ' + p_err_name + ' Error (y-axis "flattened")')
    axs[1, 3].set_xlabel('Column Index')
    axs[1, 3].set_ylabel('Error')
    axs[1, 3].set_ylim([min_error, max_error])

    # add text about errors
    mae_text = "MAE: " + str(round(mean_abs_error, 4))
    axs[1, 1].text(0.5, 0.72, mae_text, ha='center', va='center',
                   transform=axs[1, 1].transAxes, fontsize=12, color="black")
    mape_text = "MAPE: " + str(round(mean_abs_perc_error, 4))
    axs[1, 1].text(0.5, 0.62, mape_text, ha='center', va='center',
                   transform=axs[1, 1].transAxes, fontsize=12, color="black")
    rmse_text = "RMSE: " + str(round(root_mean_square_error, 4))
    axs[1, 1].text(0.5, 0.52, rmse_text, ha='center', va='center',
                   transform=axs[1, 1].transAxes, fontsize=12, color="black")
    r_sqr_text = "R^2: " + str(round(r_sqr, 4))
    axs[1, 1].text(0.5, 0.42, r_sqr_text, ha='center', va='center',
                   transform=axs[1, 1].transAxes, fontsize=12, color="black")
    r_sqr_adj_text = "R^2_adj: " + str(round(r_sqr_adj, 4))
    axs[1, 1].text(0.5, 0.32, r_sqr_adj_text, ha='center', va='center',
                   transform=axs[1, 1].transAxes, fontsize=12, color="black")
    q_sqr_text = "Q^2: " + str(round(q_sqr, 4))
    axs[1, 1].text(0.5, 0.22, q_sqr_text, ha='center', va='center',
                   transform=axs[1, 1].transAxes, fontsize=12, color="black")

    # Hide the empty subplots
    for i in [0, 1]:
        axs[1, i].axis('off')

    plt.tight_layout()
    plt.show()


def error_calc_2d(data, data_fit, fit_param, boxes, n_param, plot_error=False, error_type='abs'):
    """
    Function that calculates the different types of errors
    """
    # mask the detector occupied pixels
    data_fit = np.where(data == 0, 0, data_fit)
    # number of datapoints, number of parameters
    n_data = data.size
    # n_param   <- get the parameter amount separately, because fit_param contains offset for every peak,
    #              even-tough fit has one offset for whole fit

    # 2D errors
    abs_error = np.abs(data - data_fit)
    square_error = abs_error ** 2

    # y flat error
    abs_error_y_flat = np.abs(np.mean(data, axis=0) - np.mean(data_fit, axis=0))
    square_error_y_flat = abs_error_y_flat ** 2

    # flat 2d x errors
    abs_error_x_lst = np.mean(abs_error, axis=1)
    square_error_x_lst = np.mean(square_error, axis=1)

    # flat 2d y errors
    abs_error_y_lst = np.mean(abs_error, axis=0)
    square_error_y_lst = np.mean(square_error, axis=0)

    mean_abs_error = np.mean(abs_error)
    mean_square_error = np.mean(square_error)
    root_mean_square_error = mean_square_error ** 0.5

    np.seterr(invalid='ignore')

    mean_abs_perc_error = np.mean(np.divide(abs_error, data))
    r_sqr = 1 - np.sum(square_error) / np.sum((data - np.mean(data)) ** 2)
    r_sqr_adj = 1 - (1 - r_sqr) * (n_data - 1) / (n_data - n_param - 1)
    q_sqr = 1 - mean_square_error / np.var(data)

    pixel_count = np.count_nonzero(data)

    if plot_error:
        plot_error_calc_2d(data, data_fit, fit_param, boxes, abs_error, square_error, abs_error_y_flat,
                           square_error_y_flat, abs_error_x_lst,
                           square_error_x_lst, abs_error_y_lst, square_error_y_lst,
                           mean_square_error, root_mean_square_error,
                           mean_abs_error, mean_abs_perc_error, r_sqr, r_sqr_adj, q_sqr,
                           error_type=error_type)

    return root_mean_square_error, mean_abs_error, mean_abs_perc_error, r_sqr, r_sqr_adj, q_sqr, pixel_count


def error_calc_1d(data, data_fit, n_param, plot_error=False, error_type='abs'):
    """
    Function that calculates different types of errors for 1D data
    """
    # Number of datapoints
    n_data = data.size

    # Errors
    abs_error = np.abs(data - data_fit)
    square_error = abs_error ** 2

    # Mean Errors
    mean_abs_error = np.mean(abs_error)
    mean_square_error = np.mean(square_error)
    root_mean_square_error = np.sqrt(mean_square_error)

    # Mean Absolute Percentage Error
    with np.errstate(divide='ignore', invalid='ignore'):
        mean_abs_perc_error = np.mean(np.where(data != 0, np.abs((data - data_fit) / data), 0))

    # Coefficient of Determination, R^2
    ss_res = np.sum(square_error)
    ss_tot = np.sum((data - np.mean(data)) ** 2)
    r_sqr = 1 - (ss_res / ss_tot)

    # Adjusted R^2
    r_sqr_adj = 1 - (1 - r_sqr) * (n_data - 1) / (n_data - n_param - 1)

    # Quality of Variance Captured by the Model, Q^2
    q_sqr = 1 - mean_square_error / np.var(data)

    # Pixel Count Equivalent for 1D (Count of Non-Zero Data Points)
    pixel_count = np.count_nonzero(data)

    return root_mean_square_error, mean_abs_error, mean_abs_perc_error, r_sqr, r_sqr_adj, q_sqr, pixel_count

def find_zero_ranges(lst):
    """Find start and end of a list where it stops and stars being 0"""
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


def calculate_available_image_height(image):
    """Calculates the height in pixels of the actual measurement"""
    h = []
    for p in np.transpose(image):
        zeros = find_zero_ranges(p)
        if len(zeros) != 0:
            len_to_remove = []
            for z in zeros:
                len_to_remove.append(z[1] + 1 - z[0])
            h.append(len(p) - sum(len_to_remove))
        else:
            h.append(len(p))
    return h


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
        if i == 0:
            ratios.append(0.000000001)
        else:
            ratios.append(b / i)
    return ratios


def calculate_box_height_to_width_ratio(boxes):
    """
    Calculates the height / width ratio of the box
    """
    ratios = []
    for box in boxes:
        ratios.append((np.round(box[3] - box[1]) / np.round(box[2] - box[0])))
    return ratios


def gaussian_blur(image, size):
    """Apply gaussian on the image."""
    kernel_size = (size, size)
    sigma = 0
    return GaussianBlur(image, kernel_size, sigma)


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


def cluster_peak_boxes2(image, boxes, r, groupbox_extend):
    """
    Groups peaks and lines together based on their vicinity

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


def cluster_peak_boxes3(image, boxes, boxes_id, r, groupbox_extend, im0):
    """
    Groups peaks and lines together based on their vicinity

    r = range, the range of the gaussian blur that clusters peaks together
    groupbox_extend = after determining the clusters, extend the box even further
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

    def separate_peaks_lines_advanced(image, boxes, ratio):
        """
        Takes the image and the boxes and separates them into two lists of boxes, either peak or line
        Returns list of boxes of peaks and list of boxes of lines
        """
        box_ratios = calculate_box_to_height_ratios(image, boxes)
        box_height_width_ratio = calculate_box_height_to_width_ratio(boxes)
        boxes_peaks = []
        boxes_lines = []
        for bo, r, r_h_w in zip(boxes, box_ratios, box_height_width_ratio):
            # filter out peaks that are not lines
            if (r > ratio) or ((r_h_w >= 10) and r > 0.4):
                boxes_lines.append(bo)
            else:
                boxes_peaks.append(bo)

        return np.array(boxes_peaks), np.array(boxes_lines)


    def correct_boxes_boundaries(boxes):
        """
        Makes sure the box boundaries are inside the image.
        Being outside the image causes problems with parts of the code
        """
        for box in boxes:
            if box[0] < 0:
                box[0] = 0
            if box[1] < 0:
                box[1] = 0
            if box[2] > image.shape[1]:
                box[2] = image.shape[1]
            if box[3] > image.shape[0]:
                box[3] = image.shape[0]
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

    # make sure boxes are in bounds of the image
    boxes = correct_boxes_boundaries(boxes)

    # Create a mapping between boxes and their IDs
    box_id_map = {tuple(box): box_id for box, box_id in zip(boxes, boxes_id)}

    # Modify the get_boxes_inside_clusters function to include IDs
    def get_boxes_inside_clusters_with_ids(initial_boxes, cluster_boxes):
        boxes_inside_clusters_with_ids = []

        for cluster_box in cluster_boxes:
            boxes_inside_with_ids = [(init_box, box_id_map[tuple(init_box)]) for init_box in initial_boxes if is_box_inside(init_box, cluster_box)]
            boxes_inside_clusters_with_ids.append(boxes_inside_with_ids)

        return boxes_inside_clusters_with_ids

    def get_boxes_inside_clusters_with_ids_correct(initial_boxes, cluster_boxes):
        boxes_inside_clusters_with_ids = []
        assigned_boxes = set()

        for cluster_box in cluster_boxes:
            boxes_inside_with_ids = []
            for init_box in initial_boxes:
                init_box_tuple = tuple(init_box)
                if init_box_tuple not in assigned_boxes and is_box_inside(init_box, cluster_box):
                    boxes_inside_with_ids.append((init_box, box_id_map[init_box_tuple]))
                    assigned_boxes.add(init_box_tuple)
            boxes_inside_clusters_with_ids.append(boxes_inside_with_ids)

        return boxes_inside_clusters_with_ids

    # separate peaks and lines
    boxes_peak, boxes_line = separate_peaks_lines_advanced(image, boxes, ratio=0.8)

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
    # print(boxes_peak_int)
    for box in boxes_peak_int:
        box = box.astype(int)
        mask_peak[box[1]:box[3], box[0]:box[2]] = 1

    # mask everything that is not a peak
    mask_peak_blur = gaussian_blur(mask_peak, r)
    mask_peak2 = np.copy(mask_peak_blur)
    mask_peak2[mask_peak2 != 0] = 1

    # cluster the peaks
    mask_peak3, p_cluster_boxes = fill_clusters_with_ones(np.copy(mask_peak2))
    p_boxes_inside_clusters_with_ids = get_boxes_inside_clusters_with_ids_correct(boxes_peak, p_cluster_boxes)

    # Filter out clusters with no boxes inside them for peaks
    p_cluster_boxes = [cluster_box for cluster_box, boxes_inside in
                       zip(p_cluster_boxes, p_boxes_inside_clusters_with_ids) if len(boxes_inside) > 0]
    p_boxes_inside_clusters_with_ids = [boxes_inside for boxes_inside in p_boxes_inside_clusters_with_ids if
                                        len(boxes_inside) > 0]

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

    image_line_data0 = np.copy(im0)
    image_line_data0[mask_peak2 == 1] = 0
    image_line_data0[mask_line2 == 0] = 0

    # plt.imshow(image_line_data0, cmap='gray')
    # plt.show()

    # cluster the lines
    mask_line3, l_cluster_boxes = fill_clusters_with_ones(np.copy(mask_line2))
    l_boxes_inside_clusters_with_ids = get_boxes_inside_clusters_with_ids_correct(boxes_line, l_cluster_boxes)

    # Filter out clusters with no boxes inside them for lines
    l_cluster_boxes = [cluster_box for cluster_box, boxes_inside in zip(l_cluster_boxes, l_boxes_inside_clusters_with_ids) if len(boxes_inside) > 0]
    l_boxes_inside_clusters_with_ids = [boxes_inside for boxes_inside in l_boxes_inside_clusters_with_ids if len(boxes_inside) > 0]

    image_line_data_y_flat = flatten_y_axis(image_line_data)

    # Return the additional information with IDs
    return (p_cluster_boxes, p_boxes_inside_clusters_with_ids, l_cluster_boxes, l_boxes_inside_clusters_with_ids,
            boxes_peak, boxes_line, image_line_data_y_flat)

# Example usage (assuming necessary functions and data are defined):
# image = np.array(...)  # Load or define the image
# boxes = np.array(...)  # Define the boxes
# boxes_id = np.array(...)  # Define the boxes IDs
# r = ...  # Define the range for gaussian blur
# groupbox_extend = ...  # Define the groupbox extend value
# results = cluster_peak_boxes3(image, boxes, boxes_id, r, groupbox_extend)





def draw_marks_on_peaks(locs):
    """
    draws an "x" at the center of peaks
    """
    for (x0, y0) in locs:
        plt.scatter(x0, y0, color='red', marker='x', label='Point')


def draw_boxes(boxes, ratios, limit):
    """
    draws boxes based on their "classification"
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


def draw_boxes_basic(boxes, color = 'r'):
    """
    draws boxes
    """
    for box in boxes:
        x = box[0]
        y = box[3]
        x_width = box[2] - box[0]
        y_width = box[3] - box[1]
        rect = patches.Rectangle((x, y), x_width, -y_width, linewidth=1, edgecolor=color, facecolor='none')
        plt.gca().add_patch(rect)


def draw_groupboxes(boxes):
    """
    draws pink boxes for groupboxes
    """
    for box in boxes:
        x = box[0]
        y = box[3]
        x_width = box[2] - box[0]
        y_width = box[3] - box[1]
        rect = patches.Rectangle((x, y), x_width, -y_width, linewidth=1, edgecolor='pink', facecolor='none')
        plt.gca().add_patch(rect)


def draw_simga_ellipses(fits, axis):
    """
    Takes the fit params of peaks and plots two ellipses, one at std=1 and second and std=2
    """
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


def draw_fit_image(image, image_contrast, fit_params_line, fit_params_peak):
    """
    Takes the parameters from the fits and plots them
    Returns plot with fits and a "contrast enhanced" image for coparison
    """
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
        gg = rotated_gaussian_constant(x, y, fit[0], fit[2], fit[1], 0)
        fit_image = fit_image + gg

    # amplitude, xo, yo, sigma_x, sigma_y, theta, offset
    #     0      1   2     3        4       5        6
    # display peak fits
    for fit in fit_params_peak:
        gg = gaussian(x, y, fit[0], fit[1], fit[2], fit[3], fit[4], fit[5])
        fit_image = fit_image + gg

    # remove parts of image where there should be no data
    fit_image[image == 0] = 0

    # plot fits
    fig1 = plt.figure(figsize=(14, 8))
    plt.imshow(fit_image, origin='lower', cmap='hot', vmax=500)
    plt.title("Fits")
    plt.ylim(512, 0)
    plt.colorbar()

    # plot comparison image
    fig2 = plt.figure(figsize=(14, 8))
    plt.imshow(image_contrast, cmap="inferno")
    plt.title("comparison image")
    plt.ylim(512, 0)
    plt.colorbar()

    # fig3 = plt.figure(3)
    # residual_image = image - fit_image
    # plt.imshow(residual_image, cmap="inferno")
    # plt.ylim(512, 0)
    # plt.colorbar()

    plt.show()


def draw_fit_image_id(image, image_contrast, fit_params_line, fit_params_peak, boxes):
    """
    Takes the parameters from the fits and plots them, tags with id
    Returns plot with fits and a "contrast enhanced" image for coparison
    """
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
        gg = rotated_gaussian_constant(x, y, fit[0], fit[2], fit[1], 0)
        fit_image = fit_image + gg

    # amplitude, xo, yo, sigma_x, sigma_y, theta, offset
    #     0      1   2     3        4       5        6
    # display peak fits
    for fit in fit_params_peak:
        gg = gaussian(x, y, fit[0], fit[1], fit[2], fit[3], fit[4], fit[5])
        fit_image = fit_image + gg

    # remove parts of image where there should be no data
    fit_image[image == 0] = 0

    fig, axs = plt.subplots(2, 1, figsize=(15, 15))

    # plot fits
    # fig1 = plt.figure(figsize=(14, 8))
    axs[0].imshow(fit_image, origin='lower', cmap='hot', vmax=500)
    axs[0].set_title("Fits")
    axs[0].set_ylim(512, 0)

    draw_id_on_plot = False
    if draw_id_on_plot:
        for fit in fit_params_line:
            # ["amp", "sigma", "x0", "m", "b", "id"]
            text = str(fit[5])
            axs[0].gca().text(fit[2], 10, text, fontsize=10, color="red")
        for fit in fit_params_peak:
            # amplitude, xo, yo, sigma_x, sigma_y, theta, offset, id
            #     0      1   2     3        4       5        6     7
            text = str(fit[7])
            axs[0].gca().text(fit[1], fit[2], text, fontsize=10, color="red")
    # axs[0].colorbar()

    dbox = False

    if dbox:
        draw_boxes_basic(boxes)



    # plot comparison image
    # fig2 = plt.figure(figsize=(14, 8))
    axs[1].imshow(image, cmap="hot", vmin=0, vmax=1000)
    axs[1].set_title("comparison image")
    axs[1].set_ylim(512, 0)
    # axs[1].colorbar()

    dbox2 = True

    if dbox2:
        for box in boxes:
            x = box[0]
            y = box[3]
            x_width = box[2] - box[0]
            y_width = box[3] - box[1]
            rect = patches.Rectangle((x, y), x_width, -y_width, linewidth=1, edgecolor='g', facecolor='none')
            axs[1].add_patch(rect)

    # fig3 = plt.figure(3)
    # residual_image = image - fit_image
    # plt.imshow(residual_image, cmap="inferno")
    # plt.ylim(512, 0)
    # plt.colorbar()

    current_time = datetime.now()
    # ct0 = f"gixd-fit/timeseries_images/classic_{current_time.hour:d02}_{current_time.minute:d02}_{current_time.second:d02}.png"
    # ct = f"gixd-fit/timeseries_images/th3_{current_time.hour:02d}_{current_time.minute:02d}_{current_time.second:02d}.png"
    plt.savefig(f"gixd-fit/timeseries_images/fit_with_postprocessing_pred{current_time.hour:02d}_{current_time.minute:02d}_{current_time.second:02d}.png")

    plt.close()
    # plt.show()





