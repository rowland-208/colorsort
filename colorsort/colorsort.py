from typing import Any, Iterable

from colormath.color_objects import LabColor
from colormath.color_diff import delta_e_cie2000
import cv2
import numpy as np
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
from scipy.cluster.vq import kmeans

def argsort_iterable(x: Iterable[Any]) -> Iterable[int]:
    """Iterable of indexes that sort an array.

    This is equivalent to numpy argsort but works on any sortable python iterable
    and returns a python iterable.

    Evaluation is lazy, evaluating x only as needed for sorting.

    Args:
        x (Iterable[Any]): The array to be sorted; must be sortable

    Returns:
        Iterable[int]: The indexes that sort x
    """
    return (i for _,i in sorted((xi,i) for i,xi in enumerate(x)))

def image_to_vec(image_rgb: np.ndarray) -> np.ndarray:
    """Construct a RGB color vector from a 2D RGB image.

    Args:
        image_data (np.ndarray): image data as a ndarray or ndarray-like
            image_data.shape = (num_x_pixels, num_y_pixels, num_channels)

    Returns:
        np.ndarray: one row for each color in the input image
            shape = (num_x_pixels*num_y_pixels, num_channels)
    """
    num_rows, num_cols, num_channels = image_rgb.shape
    return np.reshape(image_rgb, (num_rows*num_cols,num_channels))

def reduce_colors(vec_rgb: np.ndarray, max_num_colors: int) -> np.ndarray:
    """Find representative colors to reduce the size of a RGB color vector.

    Uses K-means clustering to find at most max_num_colors that are representative of the colors
    in the input RGB color vector.
    The scipy.cluster.vq.kmeans method is used.
    The random seed for kmeans can be set using the gloabl numpy.random.seed method.

    Args:
        vec_rgb (np.ndarray): the color vector to reduce
            vec_rgb.shape = (num_colors, num_channels)
        max_num_colors (int, optional): The max number of colors in the final image. Defaults to 10.

    Returns:
        np.ndarray: a color vector with number of colors reduced
            shape = (min(num_colors, max_num_colors), num_channels)

    """
    if len(vec_rgb)<=max_num_colors:
        return vec_rgb.copy()

    vec_lab = cv2.cvtColor(vec_rgb[None], cv2.COLOR_RGB2LAB)[0]
    kmeans_lab,_ = kmeans(vec_lab/255., max_num_colors)
    kmeans_rgb = cv2.cvtColor((kmeans_lab[None]*255).astype(np.uint8), cv2.COLOR_LAB2RGB)[0]

    return kmeans_rgb

def sort_colors(vec_rgb: np.ndarray) -> np.ndarray:
    """Sort colors in a RGB color vector to be perceptually uniform

    Args:
        vec_rgb (np.ndarray): the color vector to sort
            vec_rgb.shape = (num_colors, num_channels)

    Returns:
        np.ndarray: colors from the original color vector sorted
            shape = (num_colors, num_channels)
    """
    vec_lab = cv2.cvtColor(vec_rgb[None], cv2.COLOR_RGB2LAB)[0]
    vec_lab = np.array(list(sorted(map(tuple,vec_lab))))
    colors = [LabColor(l/10,a,b) for l,a,b in vec_lab]
    colors = [colors[0], *colors]
    dists = np.array([[delta_e_cie2000(col1,col2) for col1 in colors] for col2 in colors])
    dists[0,-1] = 0.
    dists[-1,0] = 0.
    dists[0,2:-1] = 1e8
    dists[2:-1,0] = 1e8
    dists = dists.astype(np.int32)

    def get_route(solution, routing, manager):
        index = routing.Start(0)
        route = [manager.IndexToNode(index)]
        while not routing.IsEnd(index):
            index = solution.Value(routing.NextVar(index))
            route.append(manager.IndexToNode(index))
        return route

    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return dists[from_node][to_node]

    manager = pywrapcp.RoutingIndexManager(len(dists), 1, 0)
    routing = pywrapcp.RoutingModel(manager)
    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    solution = routing.SolveWithParameters(search_parameters)
    if solution:
        route = np.array(get_route(solution, routing, manager))
        route = route[1:-1] - 1
        vec_lab = vec_lab[route]

    sorted_rgb = cv2.cvtColor(vec_lab[None], cv2.COLOR_LAB2RGB)[0]
    return sorted_rgb
