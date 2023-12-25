from Dataset.dataset.dataset import H5GIWAXSDataset
from fit_functions_lmfit import FitBoxes

if __name__ == '__main__':
    # a_test_set_hagen()
    data = H5GIWAXSDataset("35_flipped.h5", buffer_size=5, unskewed_polar=True)
    for i, giwaxs_img_container in enumerate(data.iter_images()):
        line_params_df, line_params_err_df, peak_param_df, peak_param_err_df, peak_errors_df = FitBoxes(
            giwaxs_img_container, plot_fits=True).fit()

    """
    line_params_df          line fit parameters
    line_params_err_df      line fit parameter uncertainties
    peak_param_df           peak fit parameters
    peak_param_err_df       peak fit parameter uncertainties
    peak_errors_df          peak fit error; MSE, MAE... etc.
    """