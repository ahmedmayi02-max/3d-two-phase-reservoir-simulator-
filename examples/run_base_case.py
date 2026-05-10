from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import matplotlib.ticker as mtick

try:
    from scipy.sparse import lil_matrix
    from scipy.sparse.linalg import spsolve
    SPARSE_OK = True
except Exception:
    SPARSE_OK = False


PRINT_FULL_REPORT = True
PRINT_TIME_SERIES_ARRAYS = True


FT3_PER_STB = 5.615
C = 0.001127  # standard field Darcy constant -> rates in STB/day when B in mobility
WI_CONST = 2.0 * np.pi * C


# 1) SMALL HELPER FUNCTIONS
def clamp(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.minimum(np.maximum(x, lo), hi)


def harmonic(a: float, b: float) -> float:
    if a <= 0.0 or b <= 0.0:
        return 0.0
    return 2.0 * a * b / (a + b)


def relperm_corey(Sw: np.ndarray, Swi: float, Sor: float) -> tuple[np.ndarray, np.ndarray]:
    denom = 1.0 - Swi - Sor
    Sstar = (Sw - Swi) / denom
    Sstar = clamp(Sstar, 0.0, 1.0)
    krw = 0.22 * (Sstar ** 3)
    kro = (1.0 - Sstar) ** 3
    return krw, kro


def mobilities(
    Sw: np.ndarray,
    Swi: float,
    Sor: float,
    mu_w: float,
    mu_o: float,
    Bw: float,
    Bo: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    With B in the denominator: fluxes come out as SURFACE rates (STB/day).
    fw = water fraction in SURFACE rate sense.
    """
    krw, kro = relperm_corey(Sw, Swi, Sor)
    lam_w = krw / (mu_w * Bw)
    lam_o = kro / (mu_o * Bo)
    lam_t = lam_w + lam_o
    fw = np.where(lam_t > 0.0, lam_w / lam_t, 0.0)
    return lam_w, lam_o, lam_t, fw


def lin3(i: int, j: int, k: int, Nx: int, Ny: int) -> int:
    # flatten order: k -> j -> i
    return (k * Ny + j) * Nx + i


# 2) MODEL INPUTS
Nx, Ny, Nz = 25, 25, 3

k_md_3d = np.array([
    # Layer k=0
    [
        [7.80, 20.60, 39.94, 6.52, 3.43, 5.46, 9.43, 15.49, 0.00, 0.00, 0.00, 2.15, 8.55, 21.68, 3.69, 20.41, 2.17, 45.79, 9.53, 0.00, 0.00, 16.32, 41.62, 6.02, 6.65],
        [8.01, 56.35, 16.86, 19.32, 44.55, 8.79, 19.95, 7.65, 7.41, 16.06, 0.00, 48.83, 0.00, 20.70, 11.32, 13.83, 0.00, 9.43, 3.20, 40.93, 61.02, 0.00, 6.16, 33.84, 18.31],
        [2.94, 27.39, 8.10, 6.61, 9.37, 0.00, 8.97, 47.23, 0.00, 19.86, 8.11, 16.41, 14.22, 0.00, 32.93, 0.00, 17.08, 34.01, 17.60, 37.71, 13.52, 12.71, 63.09, 17.60, 15.82],
        [13.44, 0.00, 10.42, 0.00, 7.50, 6.72, 15.16, 13.22, 12.99, 0.00, 5.59, 4.61, 5.34, 32.38, 20.44, 5.69, 3.41, 0.00, 48.93, 24.63, 12.23, 18.67, 5.16, 9.03, 8.62],
        [38.45, 9.25, 21.21, 39.11, 5.18, 7.26, 11.38, 0.00, 0.00, 5.43, 5.03, 22.69, 29.59, 0.00, 0.00, 0.67, 10.04, 0.00, 4.75, 15.97, 5.37, 16.10, 0.00, 16.75, 17.56],
        [15.64, 2.79, 26.17, 2.36, 23.06, 19.39, 4.11, 6.06, 1.42, 21.22, 24.75, 38.79, 21.70, 6.13, 18.16, 14.62, 28.81, 0.00, 30.47, 22.42, 3.88, 33.65, 50.80, 12.08, 5.62],
        [10.16, 18.79, 0.00, 4.34, 16.25, 5.59, 0.00, 0.00, 3.05, 21.24, 2.68, 13.06, 5.39, 37.45, 14.94, 22.04, 4.23, 10.90, 26.25, 5.74, 42.86, 0.00, 46.61, 36.60, 15.76],
        [2.22, 12.57, 12.57, 44.05, 8.19, 0.00, 0.00, 5.17, 65.52, 29.88, 14.18, 13.93, 2.45, 0.00, 5.81, 0.00, 16.35, 5.20, 6.87, 23.86, 3.10, 17.11, 0.00, 9.12, 10.65],
        [33.32, 2.15, 10.94, 14.72, 22.29, 31.38, 4.83, 25.86, 17.72, 9.36, 28.86, 34.02, 58.41, 0.00, 21.44, 13.69, 0.00, 25.02, 6.57, 12.16, 31.91, 7.82, 18.99, 1.53, 18.74],
        [4.59, 5.14, 6.68, 0.00, 28.75, 15.34, 18.32, 0.00, 0.00, 14.76, 21.90, 71.60, 0.00, 0.00, 19.50, 11.63, 12.33, 26.86, 22.75, 26.87, 14.69, 9.41, 43.09, 60.72, 5.16],
        [17.96, 56.80, 28.25, 22.38, 16.65, 14.23, 0.00, 9.22, 0.00, 0.00, 38.01, 2.07, 40.72, 5.05, 2.88, 7.25, 38.97, 3.16, 7.71, 11.14, 0.00, 14.35, 25.53, 17.56, 0.00],
        [6.39, 0.00, 33.69, 12.54, 0.00, 11.36, 0.00, 0.00, 15.56, 18.44, 32.76, 0.00, 8.10, 25.92, 51.44, 7.06, 15.04, 16.56, 1.57, 5.27, 16.19, 44.91, 33.19, 39.66, 0.00],
        [7.53, 0.00, 5.51, 14.30, 5.95, 0.00, 77.85, 4.91, 8.46, 1.38, 54.67, 12.88, 2.32, 32.75, 29.19, 65.14, 0.00, 18.79, 13.21, 8.14, 22.64, 0.00, 6.17, 0.00, 18.84],
        [0.00, 3.13, 27.82, 9.31, 0.00, 2.12, 0.00, 0.00, 0.00, 76.52, 7.03, 0.00, 12.08, 1.87, 5.26, 6.17, 13.12, 5.87, 40.63, 0.00, 5.08, 18.56, 51.10, 27.80, 0.00],
        [6.34, 2.28, 0.00, 0.00, 71.72, 30.72, 23.22, 17.33, 0.00, 9.37, 2.05, 19.69, 27.06, 0.00, 8.40, 21.96, 19.06, 4.14, 4.57, 35.60, 52.26, 5.66, 29.91, 27.97, 0.00],
        [3.08, 5.56, 14.22, 17.54, 10.17, 4.44, 2.99, 0.00, 29.91, 6.39, 12.45, 6.81, 10.43, 0.00, 12.79, 9.55, 10.77, 59.64, 3.52, 11.59, 2.69, 6.62, 0.00, 1.59, 10.30],
        [7.27, 5.39, 14.12, 5.42, 20.03, 16.29, 0.00, 0.00, 10.33, 0.00, 51.74, 0.00, 0.00, 20.15, 30.50, 21.22, 2.21, 5.29, 7.72, 13.06, 0.00, 0.00, 29.85, 0.00, 15.78],
        [15.89, 0.00, 23.36, 42.50, 35.77, 14.67, 12.82, 32.44, 2.38, 0.00, 0.00, 3.00, 0.00, 17.93, 23.13, 25.47, 51.02, 0.00, 40.81, 31.05, 31.16, 11.64, 11.72, 0.00, 16.35],
        [16.47, 3.04, 0.00, 9.40, 4.04, 0.00, 10.18, 21.67, 16.07, 0.00, 58.40, 22.25, 12.08, 21.85, 12.64, 0.00, 30.99, 26.62, 21.26, 21.88, 6.33, 9.02, 43.55, 0.00, 25.43],
        [19.26, 16.34, 14.70, 4.51, 15.91, 12.07, 0.00, 0.00, 25.94, 14.44, 43.88, 16.42, 13.66, 10.15, 4.50, 3.21, 19.39, 0.00, 32.89, 2.05, 27.34, 9.11, 16.72, 21.78, 11.28],
        [22.64, 66.94, 14.66, 12.86, 6.50, 63.55, 7.11, 5.44, 35.81, 2.25, 17.43, 16.30, 46.55, 24.49, 0.00, 14.02, 24.23, 0.00, 44.36, 55.48, 22.62, 0.00, 0.00, 0.00, 0.00],
        [5.69, 0.00, 22.09, 10.90, 32.11, 19.09, 2.47, 26.98, 36.37, 19.37, 24.81, 4.59, 43.09, 12.18, 24.94, 0.00, 0.00, 18.25, 4.52, 0.00, 27.81, 73.80, 9.52, 39.20, 0.00],
        [38.59, 10.10, 7.08, 6.62, 6.76, 24.86, 26.61, 20.19, 0.00, 17.86, 28.55, 0.00, 3.89, 1.19, 4.56, 0.00, 18.37, 27.03, 0.00, 29.60, 8.09, 40.12, 0.00, 0.00, 6.77],
        [27.61, 14.73, 8.10, 0.00, 31.96, 21.58, 9.27, 5.63, 5.73, 14.91, 3.17, 0.00, 19.10, 0.00, 43.58, 0.00, 0.00, 24.80, 30.52, 6.93, 27.87, 0.00, 21.46, 17.49, 0.00],
        [13.85, 4.19, 12.78, 15.59, 6.78, 8.70, 7.30, 22.16, 0.00, 10.16, 51.98, 26.73, 10.14, 35.20, 8.49, 12.33, 16.85, 9.29, 16.74, 18.49, 20.30, 0.00, 8.46, 15.47, 70.02],
    ],

    # Layer k=1
    [
        [16.01, 5.05, 0.00, 0.00, 0.00, 32.80, 11.52, 17.25, 6.09, 0.00, 54.47, 22.73, 47.41, 0.00, 7.57, 0.00, 13.33, 11.32, 0.00, 11.35, 0.00, 16.32, 41.62, 13.22, 23.51],
        [55.67, 22.15, 11.57, 0.00, 0.00, 15.94, 16.24, 60.23, 0.00, 3.43, 24.05, 10.57, 19.72, 21.37, 0.00, 20.15, 11.05, 17.09, 0.00, 17.56, 10.75, 0.00, 6.16, 0.00, 5.74],
        [21.64, 27.19, 9.51, 3.50, 11.60, 0.00, 5.51, 0.00, 47.13, 11.98, 66.59, 31.21, 0.00, 9.75, 23.29, 2.81, 0.00, 4.42, 43.13, 8.27, 17.03, 12.71, 63.09, 14.60, 2.56],
        [11.76, 0.00, 5.57, 51.15, 6.32, 18.40, 48.35, 4.79, 52.10, 33.04, 29.08, 0.00, 72.66, 16.07, 14.30, 51.54, 0.00, 8.85, 0.00, 34.45, 28.07, 18.67, 5.16, 11.87, 26.10],
        [0.00, 6.82, 0.00, 0.00, 17.30, 4.59, 0.00, 12.57, 13.41, 39.58, 9.00, 14.64, 9.56, 54.19, 0.00, 54.46, 3.32, 0.00, 4.55, 0.00, 0.00, 16.10, 0.00, 15.85, 9.03],
        [0.00, 2.68, 25.69, 38.03, 8.25, 11.07, 42.94, 12.25, 39.34, 12.72, 56.61, 23.72, 26.78, 0.00, 10.06, 10.67, 41.21, 14.28, 11.90, 0.00, 18.49, 33.65, 50.80, 12.08, 5.09],
        [5.42, 27.47, 8.64, 0.00, 12.31, 53.15, 6.10, 19.00, 0.00, 8.02, 0.00, 3.57, 12.65, 0.00, 31.02, 4.48, 0.00, 8.31, 3.02, 2.05, 17.62, 0.00, 46.61, 11.79, 8.45],
        [48.33, 29.15, 4.94, 0.00, 2.87, 12.03, 0.00, 5.03, 4.40, 4.88, 19.24, 6.59, 13.91, 11.19, 1.27, 0.00, 37.84, 13.74, 2.32, 0.00, 5.95, 17.11, 0.00, 28.32, 12.96],
        [9.83, 19.23, 6.20, 0.00, 29.96, 0.00, 13.42, 51.43, 15.16, 11.46, 15.61, 2.45, 7.67, 41.09, 0.00, 0.00, 0.00, 27.70, 0.00, 2.64, 14.49, 7.82, 18.99, 21.61, 19.59],
        [0.00, 36.94, 0.00, 13.00, 27.44, 88.20, 12.85, 56.00, 11.63, 23.10, 4.03, 0.00, 0.00, 21.04, 18.74, 7.23, 9.75, 40.17, 14.91, 0.00, 6.99, 9.41, 43.09, 39.14, 36.67],
        [42.74, 11.88, 15.92, 21.04, 17.08, 39.10, 59.43, 32.74, 43.27, 0.00, 0.00, 0.00, 5.04, 18.38, 0.00, 35.86, 26.27, 0.00, 6.37, 33.40, 21.06, 14.35, 25.53, 15.35, 0.00],
        [0.00, 23.70, 5.54, 0.00, 6.75, 14.70, 34.82, 14.44, 0.00, 22.54, 23.99, 22.50, 5.32, 6.78, 3.92, 12.70, 0.00, 6.59, 0.00, 0.00, 20.86, 44.91, 33.19, 0.00, 1.37],
        [0.00, 48.94, 46.68, 23.39, 40.21, 62.10, 8.26, 3.86, 0.00, 37.74, 6.77, 4.55, 9.20, 42.21, 30.15, 20.70, 45.94, 5.33, 6.80, 6.00, 43.19, 0.00, 6.17, 37.13, 24.62],
        [0.00, 32.77, 34.42, 7.64, 8.99, 15.24, 0.00, 0.00, 9.05, 32.53, 43.08, 9.30, 11.82, 5.24, 2.12, 14.97, 32.69, 13.05, 18.72, 29.45, 1.16, 18.56, 51.10, 22.98, 4.32],
        [64.88, 2.67, 29.29, 34.58, 8.13, 46.77, 8.88, 45.83, 23.10, 6.83, 0.00, 1.65, 10.46, 29.68, 20.07, 0.00, 27.99, 0.00, 22.17, 60.31, 0.00, 5.66, 29.91, 27.97, 0.00],
        [16.27, 24.73, 1.98, 0.00, 8.97, 0.00, 6.20, 1.89, 0.00, 7.27, 0.00, 0.00, 6.13, 35.05, 14.76, 42.29, 1.57, 8.82, 11.25, 3.37, 6.80, 6.62, 0.00, 1.59, 0.00],
        [0.00, 0.00, 1.76, 12.15, 38.87, 30.11, 15.61, 25.93, 51.20, 6.09, 26.21, 0.00, 0.00, 47.71, 16.55, 9.67, 19.05, 27.39, 0.00, 9.56, 8.14, 0.00, 29.85, 0.00, 6.69],
        [14.82, 12.22, 4.37, 33.89, 0.00, 14.18, 21.40, 23.38, 60.32, 6.71, 5.23, 6.01, 14.24, 0.00, 20.83, 22.20, 50.20, 2.91, 20.36, 0.00, 40.54, 11.64, 11.72, 0.00, 43.09],
        [11.86, 15.05, 14.39, 0.00, 66.15, 7.03, 0.00, 0.00, 22.12, 6.81, 5.11, 3.84, 22.27, 38.68, 42.13, 0.00, 5.12, 0.00, 27.20, 8.04, 0.00, 9.02, 43.55, 7.99, 8.68],
        [6.43, 6.83, 40.57, 1.66, 17.36, 0.00, 0.00, 18.57, 0.00, 35.65, 30.47, 0.00, 29.83, 22.51, 23.43, 2.57, 11.45, 6.56, 4.57, 57.63, 24.80, 9.11, 16.72, 7.92, 15.31],
        [6.79, 8.99, 38.10, 0.00, 2.41, 6.78, 51.89, 40.49, 10.69, 4.34, 5.84, 11.29, 9.12, 7.38, 29.80, 3.37, 17.60, 14.10, 6.10, 0.00, 8.06, 0.00, 0.00, 17.90, 11.34],
        [27.81, 14.17, 19.28, 7.83, 33.52, 16.81, 0.00, 0.00, 0.00, 0.00, 0.00, 3.07, 12.45, 1.69, 0.00, 24.23, 4.72, 6.02, 4.28, 26.94, 6.78, 73.80, 9.52, 32.30, 18.13],
        [15.43, 5.60, 4.29, 0.00, 76.55, 12.38, 0.00, 35.18, 63.16, 0.00, 16.75, 0.00, 9.98, 49.23, 46.00, 0.00, 21.36, 0.00, 22.41, 16.92, 48.93, 40.12, 0.00, 21.97, 0.00],
        [8.23, 17.17, 82.64, 69.48, 35.89, 13.73, 15.11, 5.41, 10.76, 8.25, 7.50, 0.00, 7.79, 10.34, 32.22, 4.27, 3.18, 37.47, 26.16, 3.66, 20.48, 0.00, 21.46, 17.49, 21.83],
        [0.00, 11.41, 0.94, 3.28, 3.27, 10.03, 9.08, 0.00, 6.44, 9.46, 0.00, 0.00, 12.60, 13.27, 12.84, 48.45, 4.57, 18.92, 10.55, 16.16, 5.65, 0.00, 8.46, 13.75, 12.49],
    ],

    # Layer k=2
    [
        [26.48, 0.00, 0.00, 7.21, 7.15, 0.00, 0.00, 19.75, 5.87, 0.00, 15.22, 18.20, 3.44, 3.68, 11.14, 18.25, 11.04, 0.00, 0.00, 1.64, 3.92, 29.58, 16.13, 0.00, 17.99],
        [54.30, 5.45, 14.45, 0.00, 6.58, 20.33, 31.51, 16.64, 4.74, 2.19, 5.71, 61.46, 9.38, 18.75, 1.32, 4.98, 28.96, 0.00, 1.78, 63.41, 18.66, 11.73, 50.30, 0.00, 6.35],
        [13.82, 4.94, 5.15, 8.93, 8.84, 13.56, 5.56, 0.00, 3.22, 0.00, 2.85, 16.21, 40.89, 32.31, 14.18, 7.46, 12.71, 15.89, 28.84, 3.64, 13.26, 10.60, 22.61, 10.16, 5.57],
        [15.47, 5.42, 36.36, 4.33, 2.08, 15.52, 6.06, 52.47, 5.58, 20.28, 12.86, 13.95, 44.72, 11.58, 17.58, 31.11, 0.00, 3.98, 0.00, 0.00, 8.69, 19.66, 9.38, 1.88, 17.82],
        [0.00, 6.70, 5.29, 5.60, 7.19, 4.74, 4.16, 13.69, 9.20, 0.00, 46.54, 2.42, 12.35, 8.10, 17.93, 0.00, 38.70, 29.39, 8.27, 0.00, 19.36, 6.77, 21.10, 59.51, 10.31],
        [7.90, 9.01, 19.78, 8.00, 19.89, 0.00, 21.05, 14.32, 8.94, 7.49, 7.59, 34.60, 0.00, 0.00, 0.00, 3.85, 0.00, 12.03, 2.75, 3.29, 4.43, 5.72, 0.00, 32.08, 0.00],
        [6.90, 19.77, 8.40, 21.57, 6.41, 10.30, 22.64, 0.00, 44.47, 6.21, 55.74, 18.74, 12.53, 3.38, 3.87, 12.48, 16.20, 49.59, 18.44, 6.82, 1.62, 6.42, 14.88, 43.68, 0.00],
        [10.58, 9.48, 50.93, 0.00, 5.74, 11.05, 5.48, 9.59, 5.77, 5.72, 4.79, 6.59, 34.84, 9.43, 0.00, 20.93, 26.46, 15.45, 8.26, 11.87, 3.23, 1.68, 13.15, 27.24, 10.13],
        [41.47, 27.39, 11.18, 0.00, 0.00, 14.65, 8.26, 7.33, 22.08, 3.18, 7.71, 2.45, 0.00, 13.96, 17.57, 26.37, 13.77, 14.98, 8.07, 5.70, 3.58, 13.20, 0.00, 32.24, 9.76],
        [12.14, 15.18, 0.00, 0.00, 30.93, 14.35, 33.83, 0.00, 38.13, 0.00, 4.62, 0.00, 0.00, 15.78, 30.80, 55.41, 7.24, 4.60, 5.50, 5.70, 6.78, 46.68, 5.65, 0.00, 7.11],
        [15.77, 74.90, 51.48, 43.08, 15.58, 39.99, 5.87, 0.00, 35.03, 13.85, 0.00, 0.00, 51.00, 9.01, 16.95, 7.82, 65.57, 12.73, 32.15, 5.03, 11.38, 13.55, 0.00, 11.31, 8.11],
        [37.37, 0.00, 10.89, 0.00, 8.96, 3.70, 0.00, 0.00, 45.27, 12.13, 0.00, 2.76, 20.86, 0.00, 14.35, 17.60, 69.86, 25.66, 3.98, 3.81, 17.80, 0.00, 8.81, 13.69, 7.56],
        [1.32, 33.71, 0.00, 0.00, 0.00, 15.93, 38.98, 24.44, 8.55, 6.03, 20.29, 21.38, 13.28, 0.00, 26.12, 30.15, 12.85, 1.52, 28.02, 21.13, 6.73, 11.41, 0.00, 12.85, 21.81],
        [12.61, 45.25, 0.00, 18.06, 43.11, 0.00, 14.26, 29.45, 1.39, 16.21, 44.13, 17.57, 24.29, 0.00, 0.00, 0.00, 6.74, 4.55, 39.69, 6.75, 1.96, 20.04, 51.80, 0.00, 4.79],
        [10.53, 53.23, 36.27, 10.12, 12.21, 19.40, 40.39, 23.39, 31.07, 35.32, 0.00, 0.00, 35.66, 0.00, 8.30, 29.21, 10.77, 0.00, 2.77, 22.76, 0.00, 27.35, 0.00, 2.79, 8.29],
        [0.00, 0.00, 12.19, 22.11, 8.37, 11.56, 0.00, 46.20, 0.00, 0.00, 17.71, 0.00, 17.46, 5.31, 8.17, 2.95, 11.09, 22.38, 17.16, 0.00, 24.77, 9.16, 10.03, 0.00, 12.83],
        [21.95, 2.32, 11.67, 44.62, 4.58, 0.00, 33.97, 6.96, 21.61, 9.90, 0.00, 7.97, 0.00, 21.26, 45.04, 15.76, 16.35, 6.20, 19.48, 8.02, 0.00, 0.00, 15.47, 2.55, 17.20],
        [9.43, 7.27, 0.00, 22.08, 37.97, 5.53, 0.00, 5.48, 7.80, 0.00, 23.25, 78.14, 7.12, 4.68, 7.67, 0.00, 5.51, 18.17, 27.74, 0.00, 0.00, 5.65, 7.04, 2.00, 5.57],
        [0.00, 22.88, 0.00, 44.69, 0.00, 13.38, 40.12, 38.86, 7.19, 5.48, 0.00, 68.88, 25.08, 14.71, 30.34, 0.00, 25.29, 0.00, 56.34, 8.04, 5.34, 10.46, 15.64, 0.00, 3.38],
        [3.09, 0.00, 21.88, 7.24, 14.25, 42.47, 11.95, 0.00, 0.00, 5.98, 0.00, 9.16, 14.25, 27.04, 14.11, 0.00, 31.21, 1.20, 0.00, 57.63, 28.86, 33.51, 0.00, 22.55, 47.92],
        [0.00, 30.06, 8.83, 8.89, 0.00, 29.50, 19.48, 10.98, 17.03, 67.46, 29.30, 4.50, 0.00, 16.60, 10.38, 18.41, 0.00, 0.00, 27.09, 0.00, 14.27, 39.95, 32.75, 22.85, 3.91],
        [8.02, 6.64, 40.40, 12.36, 1.84, 0.00, 21.53, 0.00, 29.38, 7.67, 0.00, 0.00, 0.00, 5.54, 24.38, 40.06, 22.37, 0.00, 53.37, 26.94, 5.61, 8.40, 0.00, 6.62, 1.37],
        [4.78, 1.92, 18.05, 4.45, 7.90, 34.32, 13.01, 29.04, 12.76, 12.69, 16.19, 15.42, 11.54, 0.00, 36.52, 40.78, 11.01, 6.49, 9.17, 16.92, 0.00, 5.26, 34.11, 14.12, 18.78],
        [7.83, 9.51, 6.89, 32.06, 13.85, 4.96, 39.17, 92.82, 11.51, 0.00, 12.94, 40.43, 0.00, 2.39, 0.00, 25.05, 1.40, 34.38, 8.42, 3.66, 11.44, 6.72, 32.71, 24.94, 0.00],
        [91.09, 12.48, 0.00, 8.79, 0.00, 6.79, 10.94, 13.96, 13.93, 24.39, 26.73, 0.00, 30.04, 0.00, 6.47, 5.25, 33.87, 0.00, 15.15, 16.16, 11.40, 18.25, 21.81, 2.71, 0.00],
    ],
], dtype=float)

# keep your tweak
k_md_3d[1, 10, 14] = 76.994   # (all k, j=10, i=14)


alpha_kz = 0.1
kx = k_md_3d.copy()
ky = k_md_3d.copy()
kz = alpha_kz * k_md_3d

# Geometry
dx = 60.0  # ft
dy = 60.0  # ft
dz = 90.0  # ft

A_x = dz * dy  # ft^2
A_y = dz * dx  # ft^2
A_z = dx * dy  # ft^2

V_bulk_ft3 = dx * dy * dz
V_bulk_rb = V_bulk_ft3 / FT3_PER_STB
PV_rb = 0.1 * V_bulk_rb

# Rock / fluid
phi = 0.10

mu_w = 0.31
mu_o = 1.1445
Bw = 1.029
Bo = 1.2426

Swi, Sor = 0.22, 0.20

# Compressibilities (1/psi)
cr = 3e-6
cw = 3.13e-6
co = 1.00e-5

USE_TOTAL_CT = True
ct_effective = 1.10e-5  # fallback constant if you want

# Initial conditions
P0 = 5035.0
Sw0 = 0.25


t_final = 200
dt = 1.0
nsteps = int(round(t_final / dt))


P_W = 5000.0
P_E = 5037.0
P_S = 5047.0
P_BOT = 5052.0

BC_WEST = False
BC_EAST = False
BC_NORTH = False
BC_SOUTH = False
BC_BOTTOM = True

# Aquifer strength multipliers
AQUIFER_BOTTOM_MULT = 0.001


inj_wells = [
]

q_prod_target = 300.0  # STB/day (surface)

prod_wells = [
    {"name":"PRD1","i":14,"j":10,"k":1,"q_total":300.0}
]

# Radial WI params
rw = 0.25
skin = 0.0
re = 0.14 * np.sqrt(dx**2 + dy**2)
CT_MULT = 0.71


# 3) TRANSMISSIBILITY (3-D)
def build_face_T_total_3d(
    kx_: np.ndarray,
    ky_: np.ndarray,
    kz_: np.ndarray,
    lam_t: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Tx: (Nz, Ny, Nx+1)
    Ty: (Nz, Ny+1, Nx)
    Tz: (Nz+1, Ny, Nx)

    Rates in STB/day, pressures in psi.
    """
    Tx = np.zeros((Nz, Ny, Nx + 1), dtype=float)
    Ty = np.zeros((Nz, Ny + 1, Nx), dtype=float)
    Tz = np.zeros((Nz + 1, Ny, Nx), dtype=float)

    # x-faces
    for k in range(Nz):
        for j in range(Ny):
            for i_face in range(Nx + 1):
                if 0 < i_face < Nx:
                    kf = harmonic(kx_[k, j, i_face - 1], kx_[k, j, i_face])
                    lamf = 0.5 * (lam_t[k, j, i_face - 1] + lam_t[k, j, i_face])
                    Tx[k, j, i_face] = C * (kf * A_x / dx) * lamf
                else:
                    # boundary value still computed; used only when BC is on
                    if i_face == 0:
                        kf, lamf = kx_[k, j, 0], lam_t[k, j, 0]
                    else:
                        kf, lamf = kx_[k, j, Nx - 1], lam_t[k, j, Nx - 1]
                    Tx[k, j, i_face] = C * (kf * A_x / dx) * lamf

    # y-faces
    for k in range(Nz):
        for j_face in range(Ny + 1):
            for i in range(Nx):
                if 0 < j_face < Ny:
                    kf = harmonic(ky_[k, j_face - 1, i], ky_[k, j_face, i])
                    lamf = 0.5 * (lam_t[k, j_face - 1, i] + lam_t[k, j_face, i])
                    Ty[k, j_face, i] = C* (kf * A_y / dy) * lamf
                else:
                    if j_face == 0:
                        kf, lamf = ky_[k, 0, i], lam_t[k, 0, i]
                    else:
                        kf, lamf = ky_[k, Ny - 1, i], lam_t[k, Ny - 1, i]
                    Ty[k, j_face, i] = C * (kf * A_y / dy) * lamf

    # z-faces
    for k_face in range(Nz + 1):
        for j in range(Ny):
            for i in range(Nx):
                if 0 < k_face < Nz:
                    kf = harmonic(kz_[k_face - 1, j, i], kz_[k_face, j, i])
                    lamf = 0.5 * (lam_t[k_face - 1, j, i] + lam_t[k_face, j, i])
                    Tz[k_face, j, i] = C * (kf * A_z / dz) * lamf
                else:
                    if k_face == 0:
                        kf, lamf = kz_[0, j, i], lam_t[0, j, i]
                    else:
                        kf, lamf = kz_[Nz - 1, j, i], lam_t[Nz - 1, j, i]
                    Tz[k_face, j, i] = C * (kf * A_z / dz) * lamf

    return Tx, Ty, Tz


# 4) WELLS (3-D)
def well_WI(k_cell: float, thickness: float) -> float:
    denom = (np.log(re / rw) + skin)
    return WI_CONST * (k_cell * thickness) / denom


def add_wells_to_pressure_3d(A, rhs: np.ndarray, lam_t: np.ndarray) -> None:
    # Injectors: BHP (open in all layers)
    for w in inj_wells:
        i, j = w["i"], w["j"]
        for k in range(Nz):
            WI = well_WI(k_md_3d[k, j, i], dz)
            coeff = WI * lam_t[k, j, i]
            I = lin3(i, j, k, Nx, Ny)
            A[I, I] += coeff
            rhs[I] += coeff * w["bhp"]

    # Producers: rate controlled (sink) distributed by WI*lam_t
    for w in prod_wells:
        i, j = w["i"], w["j"]
        weights = np.zeros(Nz, dtype=float)
        for k in range(Nz):
            WI = well_WI(k_md_3d[k, j, i], dz)
            weights[k] = max(WI * lam_t[k, j, i], 0.0)
        frac = weights / weights.sum() if weights.sum() > 0 else np.full(Nz, 1.0 / Nz)

        for k in range(Nz):
            qk = w["q_total"] * frac[k]
            I = lin3(i, j, k, Nx, Ny)
            rhs[I] -= qk


def compute_producer_pwf(P: np.ndarray, lam_t: np.ndarray, w: dict) -> float:
    """
    Rate-controlled well -> compute implied Pwf:
    q = Σ WI*lam_t*(P_k - Pwf)  =>  Pwf = (Σ WI*lam_t*P_k - q) / Σ WI*lam_t
    """
    i, j = w["i"], w["j"]
    num = 0.0
    den = 0.0
    for k in range(Nz):
        WI = well_WI(k_md_3d[k, j, i], dz)
        wk = max(WI * lam_t[k, j, i], 0.0)
        num += wk * float(P[k, j, i])
        den += wk
    if den <= 0.0:
        return float(np.mean(P[:, j, i]))
    return float((num - w["q_total"]) / den)


def compute_well_rates_3d(
    P: np.ndarray,
    lam_w: np.ndarray,
    lam_o: np.ndarray,
    lam_t: np.ndarray,
    fw: np.ndarray,
) -> dict:
    out = {"injectors": [], "producers": []}

    # Injectors (BHP)
    for w in inj_wells:
        i, j = w["i"], w["j"]
        qlayers = []
        qtot = 0.0
        for k in range(Nz):
            WI = well_WI(k_md_3d[k, j, i], dz)
            qk = WI * lam_t[k, j, i] * (w["bhp"] - P[k, j, i])
            qk = max(qk, 0.0)
            qlayers.append(float(qk))
            qtot += qk
        out["injectors"].append(
            {"name": w["name"], "i": i, "j": j, "q_total": float(qtot), "q_layers": qlayers, "bhp": float(w["bhp"])}
        )

    # Producers (rate controlled)
    for w in prod_wells:
        i, j = w["i"], w["j"]
        weights = np.zeros(Nz, dtype=float)
        for k in range(Nz):
            WI = well_WI(k_md_3d[k, j, i], dz)
            weights[k] = max(WI * lam_t[k, j, i], 0.0)
        frac = weights / weights.sum() if weights.sum() > 0 else np.full(Nz, 1.0 / Nz)

        qt_layers = []
        qwt = 0.0
        qot = 0.0
        qtt = 0.0

        for k in range(Nz):
            qk = float(w["q_total"] * frac[k])
            qt_layers.append(qk)

            fwk = float(fw[k, j, i])
            qw = fwk * qk
            qo = (1.0 - fwk) * qk

            qwt += qw
            qot += qo
            qtt += qk

        wc = qwt / qtt if qtt > 0 else 0.0
        p_block = float(np.mean(P[:, j, i]))
        pwf = compute_producer_pwf(P, lam_t, w)

        out["producers"].append(
            {
                "name": w["name"],
                "i": i,
                "j": j,
                "q_w": float(qwt),
                "q_o": float(qot),
                "q_total": float(qtt),
                "wc": float(wc),
                "p_block_avg": p_block,
                "pwf": float(pwf),
                "q_layers": qt_layers,
            }
        )

    return out


# 5) PRESSURE SOLVE (IMPLICIT) with BCs (3-D)
def pressure_solve_3d(P_old: np.ndarray, Sw: np.ndarray, lam_t: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    Tx, Ty, Tz = build_face_T_total_3d(kx, ky, kz, lam_t)

    # -------- FIX: actually use your "total ct" idea --------
    if USE_TOTAL_CT:
        So = 1.0 - Sw
        ct_cell = cr + Sw * cw + So * co  # simple but effective
        ct_cell = CT_MULT * ct_cell

    else:
        ct_cell = np.full_like(Sw, ct_effective, dtype=float)


    # accumulation coefficient in STB/day/psi (approx): convert bulk to RB then to STB using an avg Bt
    Bt_ref = 0.5 * (Bo + Bw)
    alpha_cell = (phi * ct_cell * V_bulk_rb) / (Bt_ref * dt)
    # -------------------------------------------------------

    N = Nx * Ny * Nz
    rhs = np.zeros(N, dtype=float)
    A = lil_matrix((N, N), dtype=float) if SPARSE_OK else np.zeros((N, N), dtype=float)

    for k in range(Nz):
        for j in range(Ny):
            for i in range(Nx):
                I = lin3(i, j, k, Nx, Ny)

                alpha = float(alpha_cell[k, j, i])
                diag = alpha
                b = alpha * float(P_old[k, j, i])

                # WEST Dirichlet
                if i == 0:
                    if BC_WEST:
                        Tbnd = 2.0 * Tx[k, j, 0]
                        diag += Tbnd
                        b += Tbnd * P_W
                else:
                    Tw = Tx[k, j, i]
                    diag += Tw
                    A[I, lin3(i - 1, j, k, Nx, Ny)] -= Tw

                # EAST Dirichlet
                if i == Nx - 1:
                    if BC_EAST:
                        Tbnd = 2.0 * Tx[k, j, Nx]
                        diag += Tbnd
                        b += Tbnd * P_E
                else:
                    Te = Tx[k, j, i + 1]
                    diag += Te
                    A[I, lin3(i + 1, j, k, Nx, Ny)] -= Te

                # NORTH closed (no Dirichlet)
                if j > 0:
                    Tn = Ty[k, j, i]
                    diag += Tn
                    A[I, lin3(i, j - 1, k, Nx, Ny)] -= Tn

                # SOUTH + optional Dirichlet
                if j < Ny - 1:
                    Ts = Ty[k, j + 1, i]
                    diag += Ts
                    A[I, lin3(i, j + 1, k, Nx, Ny)] -= Ts
                else:
                    if BC_SOUTH:
                        Tbnd = 2.0 * Ty[k, Ny, i]
                        diag += Tbnd
                        b += Tbnd * P_S

                # BOTTOM Dirichlet aquifer (k=0)
                if k == 0:
                    if BC_BOTTOM:
                        Tbnd = 2.0 * Tz[0, j, i] * AQUIFER_BOTTOM_MULT
                        diag += Tbnd
                        b += Tbnd * P_BOT
                else:
                    Tb = Tz[k, j, i]
                    diag += Tb
                    A[I, lin3(i, j, k - 1, Nx, Ny)] -= Tb

                # TOP closed
                if k < Nz - 1:
                    Tt = Tz[k + 1, j, i]
                    diag += Tt
                    A[I, lin3(i, j, k + 1, Nx, Ny)] -= Tt

                A[I, I] = diag
                rhs[I] = b

    add_wells_to_pressure_3d(A, rhs, lam_t)

    if SPARSE_OK:
        P_vec = spsolve(A.tocsr(), rhs)
    else:
        P_vec = np.linalg.solve(A, rhs)

    P_new = P_vec.reshape((Nz, Ny, Nx))
    return P_new, Tx, Ty, Tz


# 6) FLUXES + SATURATION
def compute_face_fluxes_3d(
    P: np.ndarray, Tx: np.ndarray, Ty: np.ndarray, Tz: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ux = np.zeros((Nz, Ny, Nx + 1), dtype=float)
    uy = np.zeros((Nz, Ny + 1, Nx), dtype=float)
    uz = np.zeros((Nz + 1, Ny, Nx), dtype=float)

    # interior x faces
    for k in range(Nz):
        for j in range(Ny):
            for i_face in range(1, Nx):
                ux[k, j, i_face] = Tx[k, j, i_face] * (P[k, j, i_face - 1] - P[k, j, i_face])

    if BC_WEST:
        for k in range(Nz):
            for j in range(Ny):
                Tbnd = 2.0 * Tx[k, j, 0]
                ux[k, j, 0] = Tbnd * (P_W - P[k, j, 0])

    if BC_EAST:
        for k in range(Nz):
            for j in range(Ny):
                Tbnd = 2.0 * Tx[k, j, Nx]
                ux[k, j, Nx] = Tbnd * (P[k, j, Nx - 1] - P_E)

    # interior y faces
    for k in range(Nz):
        for j_face in range(1, Ny):
            for i in range(Nx):
                uy[k, j_face, i] = Ty[k, j_face, i] * (P[k, j_face - 1, i] - P[k, j_face, i])

    if BC_SOUTH:
        for k in range(Nz):
            for i in range(Nx):
                Tbnd = 2.0 * Ty[k, Ny, i]
                uy[k, Ny, i] = Tbnd * (P[k, Ny - 1, i] - P_S)

    # interior z faces
    for k_face in range(1, Nz):
        for j in range(Ny):
            for i in range(Nx):
                uz[k_face, j, i] = Tz[k_face, j, i] * (P[k_face - 1, j, i] - P[k_face, j, i])

    # bottom Dirichlet flux (aquifer)
    if BC_BOTTOM:
        for j in range(Ny):
            for i in range(Nx):
                Tbnd = 2.0 * Tz[0, j, i] * AQUIFER_BOTTOM_MULT
                uz[0, j, i] = Tbnd * (P_BOT - P[0, j, i])  # + into k=0 if P_BOT > P0

    return ux, uy, uz


def transport_update_3d(
    Sw: np.ndarray,
    fw: np.ndarray,
    ux: np.ndarray,
    uy: np.ndarray,
    uz: np.ndarray,
    rates: dict,
    dt_local: float,
) -> np.ndarray:
    """
    net_w is computed in SURFACE water rate (STB/day).
    Convert to RESERVOIR water volume for saturation update using Bw.
    """
    net_w_surf = np.zeros((Nz, Ny, Nx), dtype=float)

    # x faces internal
    for k in range(Nz):
        for j in range(Ny):
            for i_face in range(1, Nx):
                u = ux[k, j, i_face]
                i_w, i_e = i_face - 1, i_face
                fw_up = fw[k, j, i_w] if u >= 0 else fw[k, j, i_e]
                Fw = fw_up * u
                net_w_surf[k, j, i_w] -= Fw
                net_w_surf[k, j, i_e] += Fw

    if BC_WEST:
        for k in range(Nz):
            for j in range(Ny):
                u = ux[k, j, 0]
                net_w_surf[k, j, 0] += fw[k, j, 0] * u

    if BC_EAST:
        for k in range(Nz):
            for j in range(Ny):
                u = ux[k, j, Nx]
                net_w_surf[k, j, Nx - 1] -= fw[k, j, Nx - 1] * u

    # y faces internal
    for k in range(Nz):
        for j_face in range(1, Ny):
            for i in range(Nx):
                u = uy[k, j_face, i]
                j_n, j_s = j_face - 1, j_face
                fw_up = fw[k, j_n, i] if u >= 0 else fw[k, j_s, i]
                Fw = fw_up * u
                net_w_surf[k, j_n, i] -= Fw
                net_w_surf[k, j_s, i] += Fw

    # south aquifer boundary inflow is pure water
    if BC_SOUTH:
        for k in range(Nz):
            for i in range(Nx):
                u = uy[k, Ny, i]
                if u < 0:
                    net_w_surf[k, Ny - 1, i] += (-u) * 1.0
                else:
                    net_w_surf[k, Ny - 1, i] -= fw[k, Ny - 1, i] * u

    # z faces internal
    for k_face in range(1, Nz):
        for j in range(Ny):
            for i in range(Nx):
                u = uz[k_face, j, i]
                k_up = k_face - 1
                k_dn = k_face
                fw_up = fw[k_up, j, i] if u >= 0 else fw[k_dn, j, i]
                Fw = fw_up * u
                net_w_surf[k_up, j, i] -= Fw
                net_w_surf[k_dn, j, i] += Fw

    # bottom aquifer inflow is pure water
    if BC_BOTTOM:
        for j in range(Ny):
            for i in range(Nx):
                u = uz[0, j, i]  # + into k=0 if positive
                if u > 0:
                    net_w_surf[0, j, i] += u * 1.0
                else:
                    net_w_surf[0, j, i] += u * fw[0, j, i]

    # Producers: subtract produced water (surface)
    for w in rates["producers"]:
        i, j = w["i"], w["j"]
        for k in range(Nz):
            qk = w["q_layers"][k]
            net_w_surf[k, j, i] -= fw[k, j, i] * qk

    # Convert surface water rate -> reservoir water volume rate using Bw
    net_w_res = net_w_surf * Bw  # RB/day

    PV_rb_cell = phi * V_bulk_rb
    Sw_new = Sw + (dt_local / PV_rb_cell) * net_w_res
    return clamp(Sw_new, Swi, 1.0 - Sor)


def transport_substeps(dt_big: float, ux: np.ndarray, uy: np.ndarray, uz: np.ndarray, CFL: float = 0.45) -> int:
    """
    crude but effective stability control for explicit upwind transport.
    """
    PV_rb_cell = phi * V_bulk_rb
    max_u = max(float(np.max(np.abs(ux))), float(np.max(np.abs(uy))), float(np.max(np.abs(uz))))
    if max_u <= 1e-12:
        return 1
    # worst-case outflow ~ 6 faces
    dt_cfl = CFL * PV_rb_cell / (6.0 * max_u * Bw)
    nsub = int(np.ceil(dt_big / max(dt_cfl, 1e-12)))
    return int(np.clip(nsub, 1, 200))


# 7) VELOCITIES (ft/day)
def velocity_stats_3d(ux: np.ndarray, uy: np.ndarray, uz: np.ndarray) -> dict:
    # Convert STB/day -> ft^3/day then divide by area -> ft/day
    ux_ft3 = ux * FT3_PER_STB
    uy_ft3 = uy * FT3_PER_STB
    uz_ft3 = uz * FT3_PER_STB

    vx = ux_ft3 / A_x
    vy = uy_ft3 / A_y
    vz = uz_ft3 / A_z

    vpx = vx / phi
    vpy = vy / phi
    vpz = vz / phi

    def stats(arr: np.ndarray) -> dict:
        mag = np.abs(arr)
        return {"mean": float(np.mean(mag)), "max": float(np.max(mag))}

    return {
        "total_velocity_x": stats(vx),
        "total_velocity_y": stats(vy),
        "total_velocity_z": stats(vz),
        "pore_velocity_x": stats(vpx),
        "pore_velocity_y": stats(vpy),
        "pore_velocity_z": stats(vpz),
    }


# 8) MAIN
def main():
    if PRINT_FULL_REPORT:
        np.set_printoptions(precision=3, suppress=True, threshold=np.inf)

    P = np.full((Nz, Ny, Nx), P0, dtype=float)
    Sw = np.full((Nz, Ny, Nx), Sw0, dtype=float)

    # cumulative produced (STB)
    cum_qw = {w["name"]: 0.0 for w in prod_wells}
    cum_qo = {w["name"]: 0.0 for w in prod_wells}

    # history
    t_hist = [0.0]
    p_avg_hist = [float(P.mean())]
    sw_avg_hist = [float(Sw.mean())]

    p_prod_hist = {w["name"]: [float(np.mean(P[:, w["j"], w["i"]]))] for w in prod_wells}
    sw_prod_hist = {w["name"]: [float(np.mean(Sw[:, w["j"], w["i"]]))] for w in prod_wells}

    pwf_hist = {w["name"]: [float(P0)] for w in prod_wells}  # will be overwritten after first rates compute

    q_prod_hist = {w["name"]: [0.0] for w in prod_wells}
    qo_prod_hist = {w["name"]: [0.0] for w in prod_wells}
    qw_prod_hist = {w["name"]: [0.0] for w in prod_wells}
    wc_prod_hist = {w["name"]: [0.0] for w in prod_wells}

    # initial wc/pwf
    lam_w0, lam_o0, lam_t0, fw0 = mobilities(Sw, Swi, Sor, mu_w, mu_o, Bw, Bo)
    rates0 = compute_well_rates_3d(P, lam_w0, lam_o0, lam_t0, fw0)
    for wp in rates0["producers"]:
        name = wp["name"]
        wc_prod_hist[name][0] = float(wp["wc"])
        pwf_hist[name][0] = float(wp["pwf"])

    # TIME LOOP
    for step in range(1, nsteps + 1):
        t = step * dt

        lam_w, lam_o, lam_t, fw = mobilities(Sw, Swi, Sor, mu_w, mu_o, Bw, Bo)

        # pressure solve uses Sw (for ct_total)
        P, Tx, Ty, Tz = pressure_solve_3d(P, Sw, lam_t)

        ux, uy, uz = compute_face_fluxes_3d(P, Tx, Ty, Tz)
        rates = compute_well_rates_3d(P, lam_w, lam_o, lam_t, fw)

        # store rates + wc + pwf
        for wp in rates["producers"]:
            name = wp["name"]
            q_prod_hist[name].append(float(wp["q_total"]))
            qo_prod_hist[name].append(float(wp["q_o"]))
            qw_prod_hist[name].append(float(wp["q_w"]))
            wc_prod_hist[name].append(float(wp["wc"]))
            pwf_hist[name].append(float(wp["pwf"]))

        # stable transport (substeps)
        nsub = transport_substeps(dt, ux, uy, uz)
        dt_sub = dt / nsub
        for _ in range(nsub):
            Sw = transport_update_3d(Sw, fw, ux, uy, uz, rates, dt_local=dt_sub)

        # cumulative production (STB)
        for wp in rates["producers"]:
            name = wp["name"]
            cum_qw[name] += wp["q_w"] * dt
            cum_qo[name] += wp["q_o"] * dt

        # histories
        t_hist.append(t)
        p_avg_hist.append(float(P.mean()))
        sw_avg_hist.append(float(Sw.mean()))
        for w in prod_wells:
            p_prod_hist[w["name"]].append(float(np.mean(P[:, w["j"], w["i"]])))
            sw_prod_hist[w["name"]].append(float(np.mean(Sw[:, w["j"], w["i"]])))

    # FINAL Results
    lam_w, lam_o, lam_t, fw = mobilities(Sw, Swi, Sor, mu_w, mu_o, Bw, Bo)
    Tx, Ty, Tz = build_face_T_total_3d(kx, ky, kz, lam_t)
    ux, uy, uz = compute_face_fluxes_3d(P, Tx, Ty, Tz)
    rates_final = compute_well_rates_3d(P, lam_w, lam_o, lam_t, fw)
    vstats = velocity_stats_3d(ux, uy, uz)

    # In-place in RB (reservoir barrels)
    PV_rb_cell = phi * V_bulk_rb
    WIP_rb = PV_rb_cell * Sw
    OIP_rb = PV_rb_cell * (1.0 - Sw)



    if PRINT_FULL_REPORT:
        print("\nPermeability k (md) by layer:")
        for kk in range(Nz):
            print(f"\nLayer k={kk}:")
            print(k_md_3d[kk])

        print("\nPressure P (psi) by layer:")
        for kk in range(Nz):
            print(f"\nLayer k={kk}:")
            print(P[kk])

        print("\nWater saturation Sw by layer:")
        for kk in range(Nz):
            print(f"\nLayer k={kk}:")
            print(Sw[kk])

    print(f"\nP min/max: {P.min():.3f} / {P.max():.3f}")
    print(f"Sw min/max: {Sw.min():.6f} / {Sw.max():.6f}")

    print("\nProducer results at final time (STB/day) [RATE CONTROLLED]")
    for w in rates_final["producers"]:
        print(
            f"{w['name']} @({w['i']},{w['j']}): "
            f"qw={w['q_w']:.2f}, qo={w['q_o']:.2f}, qt={w['q_total']:.2f}, "
            f"WC={w['wc']:.3f}, P_block_avg={w['p_block_avg']:.2f}, Pwf={w['pwf']:.2f}"
        )

    print("\nCumulative produced (0-t_final) (STB)")
    for w in prod_wells:
        name = w["name"]
        print(
            f"{name}: cum_w={cum_qw[name]:.2f}, cum_o={cum_qo[name]:.2f}, "
            f"cum_liq={cum_qw[name] + cum_qo[name]:.2f}"
        )

    print("\nTotal in-place volumes")
    print(f"Total WIP = {WIP_rb.sum():.2f} RB  ({WIP_rb.sum() * FT3_PER_STB:.2f} ft^3)")
    print(f"Total OIP = {OIP_rb.sum():.2f} RB  ({OIP_rb.sum() * FT3_PER_STB:.2f} ft^3)")

    print("\nVelocity summaries (ft/day)")
    for key, val in vstats.items():
        print(f"{key}: mean={val['mean']:.7f}, max={val['max']:.6f}")

    if PRINT_TIME_SERIES_ARRAYS:
        print("\n==================== TIME SERIES ARRAYS (ALL DETAILS) ====================")
        for w in prod_wells:
            name = w["name"]
            print(f"\n{name} Pwf(t) (psi):")
            print(np.array(pwf_hist[name]))
            print(f"\n{name} q_total(t) (STB/day):")
            print(np.array(q_prod_hist[name]))
            print(f"\n{name} q_oil(t) (STB/day):")
            print(np.array(qo_prod_hist[name]))
            print(f"\n{name} q_water(t) (STB/day):")
            print(np.array(qw_prod_hist[name]))
            print(f"\n{name} watercut(t) (fraction):")
            print(np.array(wc_prod_hist[name]))

    # ---------------- PLOTS ----------------
    t_arr = np.array(t_hist)

    # Pressure
    plt.figure()
    plt.plot(t_arr, np.array(p_avg_hist), label="Average reservoir P")
    for w in prod_wells:
        plt.plot(t_arr, np.array(p_prod_hist[w["name"]]), label=f"P block {w['name']}")
    plt.xlabel("Time (days)")
    plt.ylabel("Pressure (psi)")
    plt.title("3D Pressure vs Time")
    plt.grid(True)
    plt.legend()



    # Sw (your formatting)
    plt.figure()
    ax = plt.gca()
    ax.plot(t_arr, np.array(sw_avg_hist), label="Average reservoir Sw")
    for w in prod_wells:
        ax.plot(t_arr, np.array(sw_prod_hist[w["name"]]), label=f"Sw at {w['name']} (avg perf)")
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Water saturation Sw")
    ax.set_title("3D Water Saturation vs Time")
    ax.grid(True)
    ax.legend()
    ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.4f"))

    y_all = [np.array(sw_avg_hist)]
    for w in prod_wells:
        y_all.append(np.array(sw_prod_hist[w["name"]]))
    y_all = np.concatenate(y_all)
    ymin, ymax = float(y_all.min()), float(y_all.max())
    yr = ymax - ymin
    pad = max(0.05 * yr, 1e-4)
    ax.set_ylim(ymin - pad, ymax + pad)

    # Oil rate
    plt.figure()
    for w in prod_wells:
        name = w["name"]
        plt.plot(t_arr, np.array(qo_prod_hist[name]), label=f"{name} q_oil")
    plt.xlabel("Time (days)")
    plt.ylabel("Oil rate (STB/day)")
    plt.title("3D Oil Production Rate vs Time")
    plt.grid(True)
    plt.legend()

    # Water rate
    plt.figure()
    for w in prod_wells:
        name = w["name"]
        plt.plot(t_arr, np.array(qw_prod_hist[name]), label=f"{name} q_water")
    plt.xlabel("Time (days)")
    plt.ylabel("Water rate (STB/day)")
    plt.title("3D Water Production Rate vs Time")
    plt.grid(True)
    plt.legend()

    # Water cut
    plt.figure()
    axwc = plt.gca()
    for w in prod_wells:
        name = w["name"]
        axwc.plot(t_arr, np.array(wc_prod_hist[name]), label=f"{name} water cut")
    axwc.set_xlabel("Time (days)")
    axwc.set_ylabel("Water cut (%)")
    axwc.set_title("3D Water Cut vs Time")
    axwc.grid(True)
    axwc.legend()
    axwc.set_ylim(0.0, 1.0)
    axwc.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))

    plt.show()


if __name__ == "__main__":
    main()
