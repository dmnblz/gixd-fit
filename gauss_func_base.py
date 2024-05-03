import numpy as np
from numba import jit


@jit(nopython=True)
def gaussian(x, y, amplitude, xo, yo, sigma_x, sigma_y, theta):
    xo = float(xo)
    yo = float(yo)
    a = (np.cos(theta) ** 2) / (2 * sigma_x ** 2) + (np.sin(theta) ** 2) / (2 * sigma_y ** 2)
    b = -(np.sin(2 * theta)) / (4 * sigma_x ** 2) + (np.sin(2 * theta)) / (4 * sigma_y ** 2)
    c = (np.sin(theta) ** 2) / (2 * sigma_x ** 2) + (np.cos(theta) ** 2) / (2 * sigma_y ** 2)
    g = amplitude * np.exp(- (a * ((x - xo) ** 2) + 2 * b * (x - xo) * (y - yo) + c * ((y - yo) ** 2)))
    return g


class TwoDGaussians:

    @jit(nopython=True)
    def ten_twoD_gaussians(x, y,
                           amplitude0, xo0, yo0, sigma_x0, sigma_y0, theta0,
                           amplitude1, xo1, yo1, sigma_x1, sigma_y1, theta1,
                           amplitude2, xo2, yo2, sigma_x2, sigma_y2, theta2,
                           amplitude3, xo3, yo3, sigma_x3, sigma_y3, theta3,
                           amplitude4, xo4, yo4, sigma_x4, sigma_y4, theta4,
                           amplitude5, xo5, yo5, sigma_x5, sigma_y5, theta5,
                           amplitude6, xo6, yo6, sigma_x6, sigma_y6, theta6,
                           amplitude7, xo7, yo7, sigma_x7, sigma_y7, theta7,
                           amplitude8, xo8, yo8, sigma_x8, sigma_y8, theta8,
                           amplitude9, xo9, yo9, sigma_x9, sigma_y9, theta9,
                           offset):
        g0 = gaussian(x, y, amplitude0, xo0, yo0, sigma_x0, sigma_y0, theta0)
        g1 = gaussian(x, y, amplitude1, xo1, yo1, sigma_x1, sigma_y1, theta1)
        g2 = gaussian(x, y, amplitude2, xo2, yo2, sigma_x2, sigma_y2, theta2)
        g3 = gaussian(x, y, amplitude3, xo3, yo3, sigma_x3, sigma_y3, theta3)
        g4 = gaussian(x, y, amplitude4, xo4, yo4, sigma_x4, sigma_y4, theta4)
        g5 = gaussian(x, y, amplitude5, xo5, yo5, sigma_x5, sigma_y5, theta5)
        g6 = gaussian(x, y, amplitude6, xo6, yo6, sigma_x6, sigma_y6, theta6)
        g7 = gaussian(x, y, amplitude7, xo7, yo7, sigma_x7, sigma_y7, theta7)
        g8 = gaussian(x, y, amplitude8, xo8, yo8, sigma_x8, sigma_y8, theta8)
        g9 = gaussian(x, y, amplitude9, xo9, yo9, sigma_x9, sigma_y9, theta9)

        return (g0 + g1 + g2 + g3 + g4 + g5 + g6 + g7 + g8 + g9 + offset).ravel()


    @jit(nopython=True)
    def nine_twoD_gaussians(x, y,
                            amplitude0, xo0, yo0, sigma_x0, sigma_y0, theta0,
                            amplitude1, xo1, yo1, sigma_x1, sigma_y1, theta1,
                            amplitude2, xo2, yo2, sigma_x2, sigma_y2, theta2,
                            amplitude3, xo3, yo3, sigma_x3, sigma_y3, theta3,
                            amplitude4, xo4, yo4, sigma_x4, sigma_y4, theta4,
                            amplitude5, xo5, yo5, sigma_x5, sigma_y5, theta5,
                            amplitude6, xo6, yo6, sigma_x6, sigma_y6, theta6,
                            amplitude7, xo7, yo7, sigma_x7, sigma_y7, theta7,
                            amplitude8, xo8, yo8, sigma_x8, sigma_y8, theta8,
                            offset):
        g0 = gaussian(x, y, amplitude0, xo0, yo0, sigma_x0, sigma_y0, theta0)
        g1 = gaussian(x, y, amplitude1, xo1, yo1, sigma_x1, sigma_y1, theta1)
        g2 = gaussian(x, y, amplitude2, xo2, yo2, sigma_x2, sigma_y2, theta2)
        g3 = gaussian(x, y, amplitude3, xo3, yo3, sigma_x3, sigma_y3, theta3)
        g4 = gaussian(x, y, amplitude4, xo4, yo4, sigma_x4, sigma_y4, theta4)
        g5 = gaussian(x, y, amplitude5, xo5, yo5, sigma_x5, sigma_y5, theta5)
        g6 = gaussian(x, y, amplitude6, xo6, yo6, sigma_x6, sigma_y6, theta6)
        g7 = gaussian(x, y, amplitude7, xo7, yo7, sigma_x7, sigma_y7, theta7)
        g8 = gaussian(x, y, amplitude8, xo8, yo8, sigma_x8, sigma_y8, theta8)

        return (g0 + g1 + g2 + g3 + g4 + g5 + g6 + g7 + g8 + offset).ravel()


    @jit(nopython=True)
    def eight_twoD_gaussians(x, y,
                             amplitude0, xo0, yo0, sigma_x0, sigma_y0, theta0,
                             amplitude1, xo1, yo1, sigma_x1, sigma_y1, theta1,
                             amplitude2, xo2, yo2, sigma_x2, sigma_y2, theta2,
                             amplitude3, xo3, yo3, sigma_x3, sigma_y3, theta3,
                             amplitude4, xo4, yo4, sigma_x4, sigma_y4, theta4,
                             amplitude5, xo5, yo5, sigma_x5, sigma_y5, theta5,
                             amplitude6, xo6, yo6, sigma_x6, sigma_y6, theta6,
                             amplitude7, xo7, yo7, sigma_x7, sigma_y7, theta7,
                             offset):
        g0 = gaussian(x, y, amplitude0, xo0, yo0, sigma_x0, sigma_y0, theta0)
        g1 = gaussian(x, y, amplitude1, xo1, yo1, sigma_x1, sigma_y1, theta1)
        g2 = gaussian(x, y, amplitude2, xo2, yo2, sigma_x2, sigma_y2, theta2)
        g3 = gaussian(x, y, amplitude3, xo3, yo3, sigma_x3, sigma_y3, theta3)
        g4 = gaussian(x, y, amplitude4, xo4, yo4, sigma_x4, sigma_y4, theta4)
        g5 = gaussian(x, y, amplitude5, xo5, yo5, sigma_x5, sigma_y5, theta5)
        g6 = gaussian(x, y, amplitude6, xo6, yo6, sigma_x6, sigma_y6, theta6)
        g7 = gaussian(x, y, amplitude7, xo7, yo7, sigma_x7, sigma_y7, theta7)

        return (g0 + g1 + g2 + g3 + g4 + g5 + g6 + g7 + offset).ravel()


    @jit(nopython=True)
    def seven_twoD_gaussians(x, y,
                             amplitude0, xo0, yo0, sigma_x0, sigma_y0, theta0,
                             amplitude1, xo1, yo1, sigma_x1, sigma_y1, theta1,
                             amplitude2, xo2, yo2, sigma_x2, sigma_y2, theta2,
                             amplitude3, xo3, yo3, sigma_x3, sigma_y3, theta3,
                             amplitude4, xo4, yo4, sigma_x4, sigma_y4, theta4,
                             amplitude5, xo5, yo5, sigma_x5, sigma_y5, theta5,
                             amplitude6, xo6, yo6, sigma_x6, sigma_y6, theta6,
                             offset):
        g0 = gaussian(x, y, amplitude0, xo0, yo0, sigma_x0, sigma_y0, theta0)
        g1 = gaussian(x, y, amplitude1, xo1, yo1, sigma_x1, sigma_y1, theta1)
        g2 = gaussian(x, y, amplitude2, xo2, yo2, sigma_x2, sigma_y2, theta2)
        g3 = gaussian(x, y, amplitude3, xo3, yo3, sigma_x3, sigma_y3, theta3)
        g4 = gaussian(x, y, amplitude4, xo4, yo4, sigma_x4, sigma_y4, theta4)
        g5 = gaussian(x, y, amplitude5, xo5, yo5, sigma_x5, sigma_y5, theta5)
        g6 = gaussian(x, y, amplitude6, xo6, yo6, sigma_x6, sigma_y6, theta6)

        return (g0 + g1 + g2 + g3 + g4 + g5 + g6 + offset).ravel()


    @jit(nopython=True)
    def six_twoD_gaussians(x, y,
                           amplitude0, xo0, yo0, sigma_x0, sigma_y0, theta0,
                           amplitude1, xo1, yo1, sigma_x1, sigma_y1, theta1,
                           amplitude2, xo2, yo2, sigma_x2, sigma_y2, theta2,
                           amplitude3, xo3, yo3, sigma_x3, sigma_y3, theta3,
                           amplitude4, xo4, yo4, sigma_x4, sigma_y4, theta4,
                           amplitude5, xo5, yo5, sigma_x5, sigma_y5, theta5,
                           offset):
        g0 = gaussian(x, y, amplitude0, xo0, yo0, sigma_x0, sigma_y0, theta0)
        g1 = gaussian(x, y, amplitude1, xo1, yo1, sigma_x1, sigma_y1, theta1)
        g2 = gaussian(x, y, amplitude2, xo2, yo2, sigma_x2, sigma_y2, theta2)
        g3 = gaussian(x, y, amplitude3, xo3, yo3, sigma_x3, sigma_y3, theta3)
        g4 = gaussian(x, y, amplitude4, xo4, yo4, sigma_x4, sigma_y4, theta4)
        g5 = gaussian(x, y, amplitude5, xo5, yo5, sigma_x5, sigma_y5, theta5)

        return (g0 + g1 + g2 + g3 + g4 + g5 + offset).ravel()


    @jit(nopython=True)
    def five_twoD_gaussians(x, y,
                            amplitude0, xo0, yo0, sigma_x0, sigma_y0, theta0,
                            amplitude1, xo1, yo1, sigma_x1, sigma_y1, theta1,
                            amplitude2, xo2, yo2, sigma_x2, sigma_y2, theta2,
                            amplitude3, xo3, yo3, sigma_x3, sigma_y3, theta3,
                            amplitude4, xo4, yo4, sigma_x4, sigma_y4, theta4,
                            offset):
        g0 = gaussian(x, y, amplitude0, xo0, yo0, sigma_x0, sigma_y0, theta0)
        g1 = gaussian(x, y, amplitude1, xo1, yo1, sigma_x1, sigma_y1, theta1)
        g2 = gaussian(x, y, amplitude2, xo2, yo2, sigma_x2, sigma_y2, theta2)
        g3 = gaussian(x, y, amplitude3, xo3, yo3, sigma_x3, sigma_y3, theta3)
        g4 = gaussian(x, y, amplitude4, xo4, yo4, sigma_x4, sigma_y4, theta4)

        return (g0 + g1 + g2 + g3 + g4 + offset).ravel()


    @jit(nopython=True)
    def four_twoD_gaussians(x, y,
                            amplitude0, xo0, yo0, sigma_x0, sigma_y0, theta0,
                            amplitude1, xo1, yo1, sigma_x1, sigma_y1, theta1,
                            amplitude2, xo2, yo2, sigma_x2, sigma_y2, theta2,
                            amplitude3, xo3, yo3, sigma_x3, sigma_y3, theta3,
                            offset):
        g0 = gaussian(x, y, amplitude0, xo0, yo0, sigma_x0, sigma_y0, theta0)
        g1 = gaussian(x, y, amplitude1, xo1, yo1, sigma_x1, sigma_y1, theta1)
        g2 = gaussian(x, y, amplitude2, xo2, yo2, sigma_x2, sigma_y2, theta2)
        g3 = gaussian(x, y, amplitude3, xo3, yo3, sigma_x3, sigma_y3, theta3)

        return (g0 + g1 + g2 + g3 + offset).ravel()


    @jit(nopython=True)
    def three_twoD_gaussians(x, y,
                             amplitude0, xo0, yo0, sigma_x0, sigma_y0, theta0,
                             amplitude1, xo1, yo1, sigma_x1, sigma_y1, theta1,
                             amplitude2, xo2, yo2, sigma_x2, sigma_y2, theta2,
                             offset):
        g0 = gaussian(x, y, amplitude0, xo0, yo0, sigma_x0, sigma_y0, theta0)
        g1 = gaussian(x, y, amplitude1, xo1, yo1, sigma_x1, sigma_y1, theta1)
        g2 = gaussian(x, y, amplitude2, xo2, yo2, sigma_x2, sigma_y2, theta2)

        return (g0 + g1 + g2 + offset).ravel()


    @jit(nopython=True)
    def two_twoD_gaussians(x, y,
                           amplitude0, xo0, yo0, sigma_x0, sigma_y0, theta0,
                           amplitude1, xo1, yo1, sigma_x1, sigma_y1, theta1,
                           offset):
        g0 = gaussian(x, y, amplitude0, xo0, yo0, sigma_x0, sigma_y0, theta0)
        g1 = gaussian(x, y, amplitude1, xo1, yo1, sigma_x1, sigma_y1, theta1)

        return (g0 + g1 + offset).ravel()


    @jit(nopython=True)
    def one_twoD_gaussians(x, y,
                           amplitude0, xo0, yo0, sigma_x0, sigma_y0, theta0,
                           offset):
        g0 = gaussian(x, y, amplitude0, xo0, yo0, sigma_x0, sigma_y0, theta0)

        return (g0 + offset).ravel()


def gaussian_1d(x, amplitude, sigma, xo):
    g = amplitude * np.exp(-(x - xo) ** 2 / (2 * sigma ** 2))
    return g


def lin_1d(x, m, b):
    lin = m * x + b
    return lin


class OneDGaussians:

    def one_oneD_gaussians(x,
                           amplitude0, sigma0, xo0,
                           m, b):
        g0 = gaussian_1d(x, amplitude0, sigma0, xo0)
        lin = lin_1d(x, m, b)

        return g0 + lin


    def two_oneD_gaussians(x,
                           amplitude0, sigma0, xo0,
                           amplitude1, sigma1, xo1,
                           m, b):
        g0 = gaussian_1d(x, amplitude0, sigma0, xo0)
        g1 = gaussian_1d(x, amplitude1, sigma1, xo1)
        lin = lin_1d(x, m, b)

        return g0 + g1 + lin


    def three_oneD_gaussians(x,
                             amplitude0, sigma0, xo0,
                             amplitude1, sigma1, xo1,
                             amplitude2, sigma2, xo2,
                             m, b):
        g0 = gaussian_1d(x, amplitude0, sigma0, xo0)
        g1 = gaussian_1d(x, amplitude1, sigma1, xo1)
        g2 = gaussian_1d(x, amplitude2, sigma2, xo2)
        lin = lin_1d(x, m, b)

        return g0 + g1 + g2 + lin


    def four_oneD_gaussians(x,
                            amplitude0, sigma0, xo0,
                            amplitude1, sigma1, xo1,
                            amplitude2, sigma2, xo2,
                            amplitude3, sigma3, xo3,
                            m, b):
        g0 = gaussian_1d(x, amplitude0, sigma0, xo0)
        g1 = gaussian_1d(x, amplitude1, sigma1, xo1)
        g2 = gaussian_1d(x, amplitude2, sigma2, xo2)
        g3 = gaussian_1d(x, amplitude3, sigma3, xo3)
        lin = lin_1d(x, m, b)

        return g0 + g1 + g2 + g3 + lin


    def five_oneD_gaussians(x,
                            amplitude0, sigma0, xo0,
                            amplitude1, sigma1, xo1,
                            amplitude2, sigma2, xo2,
                            amplitude3, sigma3, xo3,
                            amplitude4, sigma4, xo4,
                            m, b):
        g0 = gaussian_1d(x, amplitude0, sigma0, xo0)
        g1 = gaussian_1d(x, amplitude1, sigma1, xo1)
        g2 = gaussian_1d(x, amplitude2, sigma2, xo2)
        g3 = gaussian_1d(x, amplitude3, sigma3, xo3)
        g4 = gaussian_1d(x, amplitude4, sigma4, xo4)
        lin = lin_1d(x, m, b)
        return g0 + g1 + g2 + g3 + g4 + lin


    def six_oneD_gaussians(x,
                           amplitude0, sigma0, xo0,
                           amplitude1, sigma1, xo1,
                           amplitude2, sigma2, xo2,
                           amplitude3, sigma3, xo3,
                           amplitude4, sigma4, xo4,
                           amplitude5, sigma5, xo5,
                           m, b):
        g0 = gaussian_1d(x, amplitude0, sigma0, xo0)
        g1 = gaussian_1d(x, amplitude1, sigma1, xo1)
        g2 = gaussian_1d(x, amplitude2, sigma2, xo2)
        g3 = gaussian_1d(x, amplitude3, sigma3, xo3)
        g4 = gaussian_1d(x, amplitude4, sigma4, xo4)
        g5 = gaussian_1d(x, amplitude5, sigma5, xo5)
        lin = lin_1d(x, m, b)
        return g0 + g1 + g2 + g3 + g4 + g5 + lin

    def seven_oneD_gaussians(x,
                           amplitude0, sigma0, xo0,
                           amplitude1, sigma1, xo1,
                           amplitude2, sigma2, xo2,
                           amplitude3, sigma3, xo3,
                           amplitude4, sigma4, xo4,
                           amplitude5, sigma5, xo5,
                           amplitude6, sigma6, xo6,
                           m, b):
        g0 = gaussian_1d(x, amplitude0, sigma0, xo0)
        g1 = gaussian_1d(x, amplitude1, sigma1, xo1)
        g2 = gaussian_1d(x, amplitude2, sigma2, xo2)
        g3 = gaussian_1d(x, amplitude3, sigma3, xo3)
        g4 = gaussian_1d(x, amplitude4, sigma4, xo4)
        g5 = gaussian_1d(x, amplitude5, sigma5, xo5)
        g6 = gaussian_1d(x, amplitude6, sigma6, xo6)
        lin = lin_1d(x, m, b)
        return g0 + g1 + g2 + g3 + g4 + g5 + g6 + lin

    def eight_oneD_gaussians(x,
                           amplitude0, sigma0, xo0,
                           amplitude1, sigma1, xo1,
                           amplitude2, sigma2, xo2,
                           amplitude3, sigma3, xo3,
                           amplitude4, sigma4, xo4,
                           amplitude5, sigma5, xo5,
                           amplitude6, sigma6, xo6,
                           amplitude7, sigma7, xo7,
                           m, b):
        g0 = gaussian_1d(x, amplitude0, sigma0, xo0)
        g1 = gaussian_1d(x, amplitude1, sigma1, xo1)
        g2 = gaussian_1d(x, amplitude2, sigma2, xo2)
        g3 = gaussian_1d(x, amplitude3, sigma3, xo3)
        g4 = gaussian_1d(x, amplitude4, sigma4, xo4)
        g5 = gaussian_1d(x, amplitude5, sigma5, xo5)
        g6 = gaussian_1d(x, amplitude6, sigma6, xo6)
        g7 = gaussian_1d(x, amplitude7, sigma7, xo7)
        lin = lin_1d(x, m, b)
        return g0 + g1 + g2 + g3 + g4 + g5 + g6 + g7 + lin
