# Hough Transform Implementation

This repository contains an implementation of the Hough Transform for image processing, specifically using polar coordinates with the center of the image as the origin. This approach provides a more intuitive representation of lines, making it easier to detect and analyze lines that are in certain patterns, such as rectangles.

The use of numba library is optional, but it is recommended to speed up the computation of the loop in Hough Transform, especially if your image is large. To disable this feature, simply comment the lines `@njit(parallel=True)` and change `prange` to `range`. 

## Files

- `hough.py`: This script implements the Hough Transform and includes a function to visualize the results in polar coordinates.
- `demo_rectangles.py`: This script demonstrates the Hough Transform by drawing multiple rectangles and applying the transform to visualize the detected lines.

## Requirements

- `pip install -r requirements.txt`

# Usage
1. Hough Transform on image:

- Run `hough.py` to apply the Hough Transform to an image of your choice.
- Modify the script to load your desired image.
2. Demo with Rectangles:

- Run `demo_rectangles.py` to see the Hough Transform in action with multiple rectangles.

## License
This repository is a modified version of codes under 
[MIT](https://alyssaq.github.io/mit-license).
