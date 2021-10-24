import numpy as np
import pytest

from colorsort.colorsort import argsort_iterable, image_to_vec, reduce_colors, sort_colors

@pytest.fixture
def image_rgb():
    num_x_pixels = 20
    num_y_pixels = 20
    num_channels = 3
    shape = (num_x_pixels,num_y_pixels,num_channels)
    np.random.seed(0)
    return np.random.randint(0,255,shape).astype(np.uint8)

@pytest.fixture(params=[{'num_colors': 10},{'num_colors':100}])
def vec_rgb(request):
    num_colors = request.param['num_colors']
    num_channels = 3
    shape = (num_colors,num_channels)
    np.random.seed(0)
    return np.random.randint(0,255,shape).astype(np.uint8)

@pytest.mark.parametrize(
    'vals', [
        [],
        [0,0,0],
        list(range(10)),
        list(reversed(range(10))),
        [0,9,1,8,2,7,3,6,4,5],
        ['c','b','a'],
    ]
)
def test_argsort_iterable(vals):
    for i, val_i in zip(argsort_iterable(vals), sorted(vals)):
        assert vals[i] == val_i

def test_image_to_vec(image_rgb: np.ndarray):
    vec_rgb = image_to_vec(image_rgb)
    num_colors, num_channels = vec_rgb.shape
    assert num_colors == 400
    assert num_channels == 3

@pytest.mark.parametrize('max_num_colors', [10, 20])
def test_reduce_colors(vec_rgb: np.ndarray, max_num_colors):
    # reduce_colors uses k-means which uses numpy random
    np.random.seed(0)
    vec_rgb_reduced = reduce_colors(vec_rgb, max_num_colors)

    num_colors_start, num_channels_start = vec_rgb.shape
    num_colors_reduced, num_channels_reduced = vec_rgb_reduced.shape

    assert num_channels_start == num_channels_reduced

    if max_num_colors<num_colors_start:
        assert num_colors_reduced==max_num_colors
    # no reduction is performed if the number of colors is already
    # less than max_num_colors
    else:
        assert num_colors_reduced==num_colors_start


def test_sort_colors(vec_rgb: np.ndarray):
    vec_rgb_sorted = sort_colors(vec_rgb)

    num_colors_start, num_channels_start = vec_rgb.shape
    num_colors_sorted, num_channels_sorted = vec_rgb_sorted.shape

    assert num_colors_start == num_colors_sorted
    assert num_channels_start == num_channels_sorted
