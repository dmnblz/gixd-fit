from Dataset.dataset.dataset import H5GIWAXSDataset
from fit_functions_lmfit import FitBoxes
import numpy as np

if __name__ == '__main__':
    # a_test_set_hagen()
    data = H5GIWAXSDataset("35_flipped.h5", buffer_size=5, unskewed_polar=True)
    for i, giwaxs_img_container in enumerate(data.iter_images()):
        giwaxs_img = giwaxs_img_container.converted_polar_image
        raw_giwaxs_img = giwaxs_img_container.raw_polar_image
        img_name = giwaxs_img_container.polar_labels.img_name
        boxes = giwaxs_img_container.polar_labels.boxes
        boxes_ids = np.arange(1, len(boxes) + 1)
        boxes_with_ids = np.concatenate([boxes, boxes_ids.reshape(-1, 1)], axis=1)
        FitBoxes(raw_giwaxs_img, giwaxs_img, img_name, boxes_with_ids, plot_fits=False, dataframe=False).fit()

    """
    line_params_df          line fit parameters
    line_params_err_df      line fit parameter uncertainties
    line_errors_df          line fit error; MSE, MAE... etc.
    peak_param_df           peak fit parameters
    peak_param_err_df       peak fit parameter uncertainties
    peak_errors_df          peak fit error; MSE, MAE... etc.
    """