from typing import Tuple

from numba import njit, prange
import numpy as np
from numpy.typing import NDArray


@njit(parallel=True)  # use to accelerate the computation if the image is large
def polar_center_hough_line(
    img: NDArray, angle_step: float = 1
) -> Tuple[NDArray, NDArray, NDArray, NDArray]:
    """Hough transfrom of the image with the middle of the image as center and using polar coordinates

    Args:
        img (NDArray): 2d binary numpy array
        angle_step (float, optional): Spacing between angles in degree. Defaults to 1.

    Returns:
        accumulator (NDArray): Hough accumulator array in shape (N,M)
        rhos (NDArray): Rho values used in shape (M,)
        thetas (NDArray): degrees used in radians in shape (N,)
        degrees (NDArray): degrees used in shape (N,)
    """
    assert len(img.shape) == 2, "Input image should be 2D binary numpy array"

    # degrees from 0 to 360
    degrees = np.arange(0, 360, angle_step)
    thetas = np.deg2rad(degrees)

    # rho measure from the center of the image to corners, with step = 1
    half_height, half_width = img.shape[0] // 2, img.shape[1] // 2
    diag_len = int(np.ceil(np.sqrt(half_height**2 + half_width**2)))
    rhos = np.linspace(0, diag_len, diag_len + 1)

    cos_theta = np.cos(thetas)
    sin_theta = np.sin(thetas)
    num_thetas = len(thetas)

    # Hough accumulator array of theta vs rho
    accumulator = np.zeros((num_thetas, diag_len + 1), dtype=np.uint32)

    # (row, col) indexes to edges
    y_idxs, x_idxs = np.nonzero(img)

    # Shift indices to center at the middle of the image
    x_idxs -= half_width
    y_idxs = half_height - y_idxs

    # Vote in the hough accumulator
    for i in prange(len(x_idxs)):
        x = x_idxs[i]
        y = y_idxs[i]

        for t_idx in range(num_thetas):
            rho = int(round(x * cos_theta[t_idx] + y * sin_theta[t_idx]))

            # if rho is negative, there exists the angle that has the positive rho
            if rho < 0:
                continue

            accumulator[t_idx, rho] += 1

    return accumulator, rhos, thetas, degrees


def show_polar_hough_line(
    img: NDArray,
    accumulator: NDArray,
    rhos: NDArray,
    thetas: NDArray,
    cmap: str = "hot",
):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    plt.imshow(img, cmap="gray")

    rhos_mesh, thetas_mesh = np.meshgrid(rhos, thetas)
    fig = plt.figure()
    ax = Axes3D(fig, auto_add_to_figure=False)
    ax.set_axis_off()
    fig.add_axes(ax)

    plt.subplot(projection="polar")
    plt.pcolormesh(thetas_mesh, rhos_mesh, accumulator, cmap=cmap)
    plt.colorbar()
    plt.title("Hough transform")
    plt.ylabel("Angles (degrees)")
    plt.xlabel("Distance (pixels)")

    ax = plt.gca()
    ax.yaxis.set_label_coords(
        -0.1, 0.5
    )  # Adjust the position by specifying the coordinates
    ax.tick_params(axis="y", colors="green")
    plt.show()


if __name__ == "__main__":
    import cv2 as cv

    img = cv.imread("sudoku.png")
    dst = cv.Canny(img, 50, 200)

    accumulator, rhos, thetas, degrees = polar_center_hough_line(dst)

    show_polar_hough_line(dst, accumulator, rhos, thetas)
