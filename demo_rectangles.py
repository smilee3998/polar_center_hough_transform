from hough import polar_center_hough_line, show_polar_hough_line
from PIL import Image, ImageDraw
import numpy as np


def main():
    # Create a black binary image
    width, height = 400, 400
    image = Image.new("1", (width, height), 0)
    draw = ImageDraw.Draw(image)

    center_x, center_y = width // 2, height // 2

    # Define rectangles (width, height)
    sizes = [(100, 50), (200, 100), (300, 150)]

    for size in sizes:
        rect_width, rect_height = size
        top_left = (center_x - rect_width // 2, center_y - rect_height // 2)
        bottom_right = (center_x + rect_width // 2, center_y + rect_height // 2)
        draw.rectangle([top_left, bottom_right], outline=1)

    accumulator, rhos, thetas, degrees = polar_center_hough_line(np.array(image))
    show_polar_hough_line(np.array(image), accumulator, rhos, thetas, cmap="hot")


if __name__ == "__main__":
    main()
