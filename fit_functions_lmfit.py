import numpy as np
import lmfit
import matplotlib.pyplot as plt
import time
import pandas as pd
import multiprocessing
from gauss_func_base import OneDGaussians, TwoDGaussians
from utensil import (json_config_laoder, plot_error_calc_2d, error_calc_2d, error_calc_1d, find_zero_ranges,
                     calculate_available_image_height, calculate_box_to_height_ratios, gaussian_blur, flatten_y_axis,
                     cluster_peak_boxes2, cluster_peak_boxes3, draw_marks_on_peaks, draw_boxes, draw_boxes_basic,
                     draw_groupboxes, draw_simga_ellipses, draw_fit_image, draw_fit_image_id)


class N2DGaussian:
    """Class for fitting 2D gaussians"""

    __slots__ = "raw_image", "boxes", "image_boundaries", "plot_result"

    def __init__(self, raw_image, boxes, image_boundaries, plot_result):
        self.raw_image = raw_image
        self.boxes = boxes
        self.image_boundaries = image_boundaries
        self.plot_result = plot_result

    def fit_n_2d_gaussian(self):
        """
        n 2d gaussian fitting with a rotation angle theta
        can fit up to max 10 gaussians at the same time
        """
        # image boundaries
        ib = np.array([np.floor(self.image_boundaries[0]), np.floor(self.image_boundaries[1]),
                       np.ceil(self.image_boundaries[2]), np.ceil(self.image_boundaries[3])]).astype(int)

        # make sure the image boudaries are not out of range
        for i in range(len(ib)):
            if ib[i] < 0:
                ib[i] = 0

        # get the part of the image
        image_cut = self.raw_image[ib[1]:ib[3], ib[0]:ib[2]]

        # obsolete for now
        # boxes_relative = []
        # for box in self.boxes:
        #     br = box[0]
        #     boxes_relative.append([br[0] - ib[0], br[1] - ib[1],
        #                            br[2] - ib[0], br[3] - ib[1]])
        # boxes_relative = np.array(boxes_relative)

        # get the relative position of the boxes in the image as int
        boxes_relative_int = []
        for box in self.boxes:
            br = box[0]
            boxes_relative_int.append([np.floor(br[0] - ib[0]), np.floor(br[1] - ib[1]),
                                       np.ceil(br[2] - ib[0]), np.ceil(br[3] - ib[1])])
        boxes_relative_int = np.array(boxes_relative_int).astype(int)

        # mask every value == 0
        mask = image_cut != 0

        # prepare the image for fitting
        y_len, x_len = image_cut.shape
        x, y = np.meshgrid(np.arange(x_len), np.arange(y_len))
        data_1d = image_cut[mask].ravel()
        X, Y = x[mask].ravel(), y[mask].ravel()

        # get starting parameters for initial guess
        amp_initial_lst = []
        x0_initial_lst = []
        y0_initial_lst = []
        sigma_x_initial_lst = []
        sigma_y_initial_lst = []
        theta_initial_lst = []

        if image_cut.size > 0:
            pass
        else:
            print("boxes:")
            print(boxes_relative_int)
            print("array:")
            print(image_cut)

        for br in boxes_relative_int:
            # Amplitude as max value - min value of the image
            # DO THIS BETTER -> media filter might remove the salt and pepper style noise
            # -> better finding the actually not noised max
            amp_initial = np.max(image_cut[br[1]:br[3], br[0]:br[2]]) - 0
            amp_initial_lst.append(amp_initial)

            # x0 and y0 as point of where the max value of the box inside the image is
            max_position = np.unravel_index(image_cut[br[1]:br[3], br[0]:br[2]].argmax(),
                                            image_cut[br[1]:br[3], br[0]:br[2]].shape)

            x0_initial = br[0] + max_position[1]
            x0_initial_lst.append(x0_initial)

            y0_initial = br[1] + max_position[0]
            y0_initial_lst.append(y0_initial)

            # print(br[2]-br[0]/10)
            # print(br[3]-br[1]/10)

            # test with box sizes
            # sigma_x_initial_lst.append((br[2]-br[0])/10)
            # sigma_y_initial_lst.append((br[3]-br[1])/10)

            # TODO: do it trough FWHM? although 1 seems to work fine
            sigma_x_initial_lst.append(1)
            sigma_y_initial_lst.append(1)

            theta_initial_lst.append(0)

        initial_guess_lst = np.array([amp_initial_lst,
                                      x0_initial_lst,
                                      y0_initial_lst,
                                      sigma_x_initial_lst,
                                      sigma_y_initial_lst,
                                      theta_initial_lst])
        initial_guess_lst = np.transpose(initial_guess_lst)

        # offset c
        c_initial = 0

        # VERY IMPORTANT PARAMETER
        # mean variability = how many pixel is the mean allowed to exceed its predicted box
        mv = 0.5

        n = len(boxes_relative_int)  # number of gaussians

        """PREPARE INITIAL PARAMETERS FOR FITTING"""
        params = lmfit.Parameters()
        for br, ig, no in zip(boxes_relative_int, initial_guess_lst, np.arange(n)):
            # initial guess order: amp, x0, y0, sigma_x, sigma_y, theta
            #                       0    1   2     3        4       5
            params.add('amplitude' + str(no), value=ig[0], min=0)
            params.add('xo' + str(no), value=ig[1], min=br[0] - mv, max=br[2] + mv)
            params.add('yo' + str(no), value=ig[2], min=br[1] - mv, max=br[3] + mv)
            params.add('sigma_x' + str(no), value=ig[3], min=(br[2] - br[0]) / 6, max=int(br[2] - br[0]))
            params.add('sigma_y' + str(no), value=ig[4], min=(br[3] - br[1]) / 6, max=int(br[3] - br[1]))
            params.add('theta' + str(no), value=0, min=-np.pi / 4, max=np.pi / 4)
        params.add('offset', value=c_initial, min=0)

        """ FIT """
        models = TwoDGaussians
        model_lst = [models.one_twoD_gaussians, models.two_twoD_gaussians, models.three_twoD_gaussians,
                     models.four_twoD_gaussians, models.five_twoD_gaussians, models.six_twoD_gaussians,
                     models.seven_twoD_gaussians, models.eight_twoD_gaussians, models.nine_twoD_gaussians,
                     models.ten_twoD_gaussians]
        model = lmfit.Model(model_lst[n - 1], independent_vars=['x', 'y'])
        result = model.fit(data_1d, params=params, x=X, y=Y)
        fit_data = model.func(x, y, **result.best_values).reshape(y_len, x_len)

        """ GET THE FIT PARAMETERS """
        best_fit_param = [
            [
                result.params[f"amplitude{no}"].value,
                result.params[f"xo{no}"].value + ib[0],
                result.params[f"yo{no}"].value + ib[1],
                result.params[f"sigma_x{no}"].value,
                result.params[f"sigma_y{no}"].value,
                result.params[f"theta{no}"].value,
                result.params["offset"].value,
                self.boxes[no][1]
            ]
            for no in range(n)
        ]

        best_fit_param_relative = [
            [
                result.params[f"amplitude{no}"].value,
                result.params[f"xo{no}"].value,
                result.params[f"yo{no}"].value,
                result.params[f"sigma_x{no}"].value,
                result.params[f"sigma_y{no}"].value,
                result.params[f"theta{no}"].value,
                result.params["offset"].value,
                self.boxes[no][1]
            ]
            for no in range(n)
        ]
        """ GET THE FIT PARAMETER'S STD ERRORS """
        fit_param_sdt_err = [
            [
                result.params[f"amplitude{no}"].stderr,
                result.params[f"xo{no}"].stderr,
                result.params[f"yo{no}"].stderr,
                result.params[f"sigma_x{no}"].stderr,
                result.params[f"sigma_y{no}"].stderr,
                result.params[f"theta{no}"].stderr,
                result.params["offset"].stderr,
                self.boxes[no][1]
            ]
            for no in range(n)
        ]

        """ STATISTICS """
        errors = list(error_calc_2d(image_cut, fit_data, best_fit_param_relative, boxes_relative_int, len(params),
                                    plot_error=self.plot_result, error_type='abs')[:])
        errors.append(n)
        return best_fit_param, fit_param_sdt_err, errors


class N1DGaussian:
    """
    Class for fitting lines
    """
    __slots__ = "image", "boxes", "cluster_box"

    def __init__(self, image, boxes, cluster_box):
        self.image = image
        self.boxes = boxes
        self.cluster_box = cluster_box

    def fit_n_1d_gaussians(self):
        """
        Fits n line gaussians (gaussian in x, const in y)
        """
        y = np.array(self.image[int(np.floor(self.cluster_box[0] + 1)):int(np.ceil(self.cluster_box[2]))])
        x = np.arange(len(y))

        # prepare the box coordinates
        boxes_relative = []
        for box_id in self.boxes:
            box = box_id[0]
            boxes_relative.append([box[0] - self.cluster_box[0], box[1] - self.cluster_box[1],
                                   box[2] - self.cluster_box[0], box[3] - self.cluster_box[1]])
        boxes_relative = np.array(boxes_relative)
        boxes_relative_int = np.concatenate([np.floor(boxes_relative[:, :2]), np.ceil(boxes_relative[:, 2:])],
                                            axis=1).astype(int)

        """PREPARE INITIAL PARAMETERS"""
        """THIS NEEDS MORE FINE-TUNING"""
        params = lmfit.Parameters()
        # initial guess order: amp, sigma, x0, m, b
        #                       0    1      2  3  4
        for no, box in enumerate(boxes_relative_int):
            params.add('amplitude' + str(no), value=max(y[box[0]:box[2]]) - min(y),
                       min=(max(y[box[0]:box[2]]) - min(y)) / 2 - 0.0001,
                       max=(max(y[box[0]:box[2]]) - min(y)) * 2)
            params.add('sigma' + str(no), value=(box[2] - box[0]) / 2, min=(box[2] - box[0]) / 6, max=box[2] - box[0])
            params.add('xo' + str(no), value=x[np.argmax(y[box[0]:box[2]])], min=box[0], max=box[2])
        params.add('m', value=min(y), min=0, max=max(y))
        params.add('b', value=y[0])

        # Create the model for the fit based on the amount of lines ("1D gaussians")
        n_gaussians = len(self.boxes)
        models = OneDGaussians
        model_lst = [models.one_oneD_gaussians, models.two_oneD_gaussians, models.three_oneD_gaussians,
                     models.four_oneD_gaussians, models.five_oneD_gaussians, models.six_oneD_gaussians,
                     models.seven_oneD_gaussians, models.eight_oneD_gaussians]
        model = lmfit.Model(model_lst[n_gaussians - 1])

        # Fit the model to the data
        result = model.fit(y, params, x=x, method='leastsq')
        data_fit = result.best_fit

        """ GET THE FIT PARAMETERS """
        best_fit_param = [
            [
                result.params[f"amplitude{no}"].value,
                result.params[f"sigma{no}"].value,
                result.params[f"xo{no}"].value + self.cluster_box[0],
                result.params["m"].value,
                result.params["b"].value,
                self.boxes[no][1]  # box id
            ]
            for no in range(n_gaussians)
        ]

        """ GET THE FIT PARAMETER'S STD ERRORS """
        fit_param_std_err = [
            [
                result.params[f"amplitude{no}"].stderr,
                result.params[f"sigma{no}"].stderr,
                result.params[f"xo{no}"].stderr,
                result.params["m"].stderr,
                result.params["b"].stderr,
                self.boxes[no][1]  # box id
            ]
            for no in range(n_gaussians)
        ]
        """STATISTICS"""

        errors = list(error_calc_1d(data=x, data_fit=data_fit,
                                    n_param=len(result.params), plot_error=False, error_type='abs'))

        errors.append(n_gaussians)

        # Optionally, plot the results
        plot = False
        if plot:
            for ppp in range(len(best_fit_param)):
                mid = best_fit_param[ppp][1]
                plt.vlines(mid + 1, min(y), max(y), colors="g")
                plt.plot(x + self.cluster_box[0] + 1, result.best_fit, 'r--')  # Fitted curve

        return best_fit_param, fit_param_std_err, errors


def fit_gaussian_wrapper(args):
    raw_giwaxs_img, box_in_cluster, cluster_box = args
    best_fit_param, fit_param_err, errors = N2DGaussian(raw_giwaxs_img, box_in_cluster, cluster_box,
                                                        plot_result=False).fit_n_2d_gaussian()
    return best_fit_param, fit_param_err, errors


class FitBoxes:
    __slots__ = "raw_giwaxs_img", "giwaxs_img", "img_name", "boxes", "plot_fits", "dataframe"  # , "config"

    def __init__(self, raw_giwaxs_img, giwaxs_img, img_name, boxes, plot_fits, dataframe):  #, config):

        self.raw_giwaxs_img = raw_giwaxs_img
        self.giwaxs_img = giwaxs_img
        self.img_name = img_name
        self.boxes = boxes
        self.plot_fits = plot_fits
        self.dataframe = dataframe
        # self.config = config

    def fit(self):
        print("Single process")

        """GET VALUES"""
        boxes = self.boxes
        img_name = self.img_name
        giwaxs_img = self.giwaxs_img
        raw_giwaxs_img = self.raw_giwaxs_img

        """ASSIGN ID TO BOXES"""
        assign_id_to_boxes = False
        if assign_id_to_boxes:
            boxes_id = np.arange(len(boxes))
        else:
            boxes_id = boxes[:, -1]
            boxes = boxes[:, 0:-1]

        """CLUSTERING"""
        start_time = time.time()
        (peak_cluster_boxes, peak_boxes_inside_clusters,
         line_cluster_boxes, line_boxes_inside_clusters,
         peak_boxes, line_boxes, image_line_data_y_flat) = cluster_peak_boxes3(raw_giwaxs_img, boxes, boxes_id, 7, 1,
                                                                               giwaxs_img)
        end_time = time.time()
        print(f"The clustering process took {end_time - start_time:.4f} seconds to complete.")

        """LINE FIT"""
        start_time = time.time()
        line_params_lst = []
        line_params_err_lst = []
        line_errors = []
        for cluster, boxes_in_cluster in zip(line_cluster_boxes, line_boxes_inside_clusters):
            param, param_err, error = N1DGaussian(image_line_data_y_flat, boxes_in_cluster,
                                                  cluster).fit_n_1d_gaussians()
            line_params_lst.extend(param)
            line_params_err_lst.extend(param_err)
            line_errors.append(error)

        end_time = time.time()
        duration = end_time - start_time
        print(f"The line fitting process took {duration:.4f} seconds to complete.")

        # convert to df
        line_params = np.array(line_params_lst)
        line_params_err = np.array(line_params_err_lst)

        line_params_df = pd.DataFrame(np.array(line_params))
        if not line_params_df.empty:
            line_params_df.columns = ["amp", "sigma", "x0", "m", "b", "id"]

        line_params_err_df = pd.DataFrame(np.array(line_params_err))
        if not line_params_err_df.empty:
            line_params_err_df.columns = ["amp ±", "sigma ±", "x0 ±", "m ±", "b ±", "id"]

        line_errors = np.array(line_errors)
        if np.any(line_errors):
            line_errors_df = pd.DataFrame(line_errors)
            line_errors_df.columns = ["RMSE", "MAE", "MAPE", "R^2", "R^2_adj", "Q^2", "n(pixel)", "gaussians"]
        else:
            line_errors_df = []

        """2D PEAKS FIT"""
        peak_param_lst = []
        peak_param_err_lst = []
        peak_errors_lst = []
        # n-2d-gaussian on every peak cluster
        start_time_peak = time.time()
        for c_b, b_in_c in zip(enumerate(peak_cluster_boxes), peak_boxes_inside_clusters):
            # count the round
            print(str(c_b[0] + 1) + "/" + str(len(peak_cluster_boxes)) + ", n(gaussians): " + str(len(b_in_c)))
            best_fit_param, fit_param_err, errors = N2DGaussian(raw_giwaxs_img, b_in_c, c_b[1],
                                                                plot_result=False).fit_n_2d_gaussian()
            peak_param_lst.extend(best_fit_param)
            peak_param_err_lst.extend(fit_param_err)
            peak_errors_lst.append(errors)
        end_time_peak = time.time()
        duration_peak = end_time_peak - start_time_peak
        print(f"The peak fitting process took {duration_peak:.4f} seconds to complete.")

        # convert to df
        peak_param = np.array(peak_param_lst)
        peak_param_err = np.array(peak_param_err_lst)
        peak_errors = np.array(peak_errors_lst)

        if np.any(peak_param):
            peak_param_df = pd.DataFrame(peak_param)
            peak_param_df.columns = ["amp", "x0", "y0", "sigma_x", "sigma_y", "theta", "offset", "id"]

            peak_param_err_df = pd.DataFrame(peak_param_err)
            peak_param_err_df.columns = ["amp ±", "x0 ±", "y0 ±", "sigma_x ±", "sigma_y ±", "theta ±", "offset ±", "id"]
        else:
            peak_param_df = []
            peak_param_err_df = []

        """CALCULATE 2D PEAK ERRORS"""
        peak_params = np.array(peak_param_lst)
        if np.any(peak_errors):
            peak_errors_df = pd.DataFrame(peak_errors)
            peak_errors_df.columns = ["RMSE", "MAE", "MAPE", "R^2", "R^2_adj", "Q^2", "n(pixel)", "gaussians"]
        else:
            peak_errors_df = []

        """DISPLAY FIT"""
        plot_peak = False

        if plot_peak:
            if peak_param_lst:
                # plot_fit_image(raw_giwaxs_img, fit_results)
                fig, ax = plt.subplots()
                plt.imshow(giwaxs_img, cmap="gray")
                plt.title(img_name)
                draw_marks_on_peaks(peak_params[:, 1:3])
                draw_simga_ellipses(peak_params, ax)
                # draw_boxes(boxes=boxes, ratios=boxes_ratios, limit=0.8)
                draw_groupboxes(peak_cluster_boxes)
                plt.show()

        if self.plot_fits:
            # display fits
            draw_fit_image_id(raw_giwaxs_img, giwaxs_img, line_params_lst, peak_param_lst, boxes)

        if self.dataframe:
            return line_params_df, line_params_err_df, line_errors_df, peak_param_df, peak_param_err_df, peak_errors_df
        else:
            return line_params, line_params_err, line_errors, peak_param, peak_param_err, peak_errors

    def fit_parallel(self):
        print("\nMultiprocessing")
        """GET VALUES"""
        boxes = self.giwaxs_img_container.polar_labels.boxes
        img_name = self.giwaxs_img_container.polar_labels.img_name
        giwaxs_img = self.giwaxs_img_container.converted_polar_image
        raw_giwaxs_img = self.giwaxs_img_container.raw_polar_image

        boxes_ratios = calculate_box_to_height_ratios(image=raw_giwaxs_img, boxes=boxes)

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
        line_params_lst = []
        line_params_err_lst = []
        for cluster, boxes_in_cluster in zip(line_cluster_boxes, line_boxes_inside_clusters):
            param, err = N1DGaussian(image_line_data_y_flat, boxes_in_cluster, cluster).fit_n_1d_gaussians()
            line_params_lst.extend(param)
            line_params_err_lst.extend(err)

        end_time = time.time()
        duration = end_time - start_time
        print(f"The line fitting process took {duration:.4f} seconds to complete.")

        # convert to df
        line_params = np.array(line_params_lst)
        line_params_err = np.array(line_params_err_lst)

        line_params_df = pd.DataFrame(np.array(line_params))
        line_params_df.columns = ["amp", "sigma", "x0", "m", "b"]

        line_params_err_df = pd.DataFrame(np.array(line_params_err))
        line_params_err_df.columns = ["amp ±", "sigma ±", "x0 ±", "m ±", "b ±"]

        """2D PEAKS FIT"""
        peak_param_lst = []
        peak_param_err_lst = []
        peak_errors_lst = []
        # n-2d-gaussian on every peak cluster
        start_time_peak = time.time()

        arguments = [(raw_giwaxs_img, b_in_c, c_b[1]) for c_b, b_in_c in
                     zip(enumerate(peak_cluster_boxes), peak_boxes_inside_clusters)]

        with multiprocessing.Pool(processes=4) as pool:
            results = pool.map(fit_gaussian_wrapper, arguments)

        for result in results:
            best_fit_param, fit_param_err, errors = result
            peak_param_lst.extend(best_fit_param)
            peak_param_err_lst.extend(fit_param_err)
            peak_errors_lst.append(errors)

        end_time_peak = time.time()
        duration_peak = end_time_peak - start_time_peak
        print(f"The peak fitting process took {duration_peak:.4f} seconds to complete.")

        # convert to df
        peak_param = np.array(peak_param_lst)
        peak_param_err = np.array(peak_param_err_lst)
        peak_errors = np.array(peak_errors_lst)

        if np.any(peak_param):
            peak_param_df = pd.DataFrame(peak_param)
            peak_param_df.columns = ["amp", "x0", "y0", "sigma_x", "sigma_y", "theta", "offset"]

            peak_param_err_df = pd.DataFrame(peak_param_err)
            peak_param_err_df.columns = ["amp ±", "x0 ±", "y0 ±", "sigma_x ±", "sigma_y ±", "theta ±", "offset ±"]
        else:
            peak_param_df = []
            peak_param_err_df = []

        """CALCULATE 2D PEAK ERRORS"""
        peak_params = np.array(peak_param_lst)
        if np.any(peak_errors):
            peak_errors_df = pd.DataFrame(peak_errors)
            peak_errors_df.columns = ["RMSE", "MAE", "MAPE", "R^2", "R^2_adj", "Q^2", "n(pixel)", "gaussians"]
        else:
            peak_errors_df = []

        """DISPLAY FIT"""
        plot_peak = False

        if plot_peak:
            if peak_param_lst:
                # plot_fit_image(raw_giwaxs_img, fit_results)
                fig, ax = plt.subplots()
                plt.imshow(giwaxs_img, cmap="gray")
                plt.title(img_name)
                draw_marks_on_peaks(peak_params[:, 1:3])
                draw_simga_ellipses(peak_params, ax)
                # draw_boxes(boxes=boxes, ratios=boxes_ratios, limit=0.8)
                draw_groupboxes(peak_cluster_boxes)
                plt.show()

        if self.plot_fits:
            # display fits
            draw_fit_image(raw_giwaxs_img, giwaxs_img, line_params_lst, peak_param_lst)

        return line_params_df, line_params_err_df, peak_param_df, peak_param_err_df, peak_errors_df
