The colorsort python library provides algorithms for sorting colors.

You can use colorsort to:
* Convert an image into a color vector
* Reduce the number of colors in a color vector while maintaining the overall pallete
* Sort the colors in a color vector

For an interactive demo check out [this jupyter notebook](https://github.com/rowland-208/colorsort/blob/main/etc/examples.ipynb).

Example converting image into color vector:

    >>> import colorsort as csort
    >>> num_x_pixels = 16
    >>> num_y_pixels = 16
    >>> num_channels = 3
    >>> image_shape = (num_x_pixels, num_y_pixels, num_channels)
    >>> image_rgb = np.random.randint(0,255,image_shape).astype(np.uint8)
    >>> vec_rgb = csort.image_to_vec(image_rgb)

Casting the array to uint8 is crucial.
Colorsort uses opencv under the hood and expects uint8.

Example reducing the number of colors in a color vector

    >>> print(vec_rgb.shape)
    (256, 3)
    >>> vec_rgb_reduced = csort.reduce_colors(vec_rgb, 10)
    >>> print(vec_rgb_reduced.shape)
    (10, 3)

Colors are reduced using K-means clustering in the LAB colorspace.
If the input has fewer than 10 colors,
a copy of the input color vector is returned.

Example sorting colors in a color vector

    >>> vec_rgb_sorted = csort.sort_colors(vec_rgb_reduced)
    >>> print(vec_rgb_sorted.shape)
    (10, 3)

The color sorting algorighm maps the problem of sorting colors onto
a travelling salesman problem.
The route starts at the brightest color,
and ends at the darkest color.
The distance between two colors is calculated using the
Delta E CIE2000 standard.
By finding the shortest path that visits all colors
the resulting colors are perceptually sorted,
with minimal jumps between neighboring colors.
This is useful for constructing colormaps from arbitrary images.
