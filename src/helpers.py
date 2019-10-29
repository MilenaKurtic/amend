# helper functions for algorithms.py file

import numpy as np
import numpy.linalg as linalg

def convert_to_homogeneous(points):
    data_size = len(points[0])
    if data_size == 2:
        homogeneous = np.zeros((len(points), 3))
        for i in range(len(points)):
            homogeneous[i] = np.append(points[i], [1])
        return homogeneous
    else:
        return points

def get_p_matrix(points):
    A = points[0]
    B = points[1]
    C = points[2]
    D = points[3]
    
    delta = linalg.det(np.array([
        [A[0], B[0], C[0]],
        [A[1], B[1], C[1]],
        [A[2], B[2], C[2]]
    ]))

    delta1 = linalg.det(np.array([
        [D[0], B[0], C[0]],
        [D[1], B[1], C[1]],
        [D[2], B[2], C[2]]
    ]))

    delta2 = linalg.det(np.array([
        [A[0], C[0], D[0]],
        [A[1], C[1], D[1]],
        [A[2], C[2], D[2]]
    ]))

    delta3 = linalg.det(np.array([
        [A[0], B[0], D[0]],
        [A[1], B[1], D[1]],
        [A[2], B[2], D[2]]
    ]))

    l1 = delta1 / delta 
    l2 = delta2 / delta 
    l3 = delta3 / delta 

    return np.array([
        [A[0] * l1, B[0] * l2, C[0] * l3],
        [A[1] * l1, B[1] * l2, C[1] * l3],
        [A[2] * l1, B[2] * l2, C[2] * l3]
    ])

def make_2x9_matrix(points):
    return np.array([
        [                             0,                            0,                            0,
           -points[1][2] * points[0][0], -points[1][2] * points[0][1], -points[1][2] * points[0][2],
            points[1][1] * points[0][0],  points[1][1] * points[0][1],  points[1][1] * points[0][2]
        ],
        [
            points[1][2] * points[0][0],  points[1][2] * points[0][1],  points[1][2] * points[0][2],
                                      0,                            0,                            0,
           -points[1][0] * points[0][0], -points[1][0] * points[0][1], -points[1][0] * points[0][2]
        ]
    ])

def normalize_points(points):
    data_size = len(points)

    # get points center of gravity 
    sum_x = np.sum(point[0] / point[2] for point in points) / data_size
    sum_y = np.sum(point[1] / point[2] for point in points) / data_size

    cog = np.array([sum_x, sum_y, 1])

    # translate points to origin
    G = np.array([
        [1, 0, -cog[0]],
        [0, 1, -cog[1]],
        [0, 0,    1   ]
    ])

    # scale the points so that the average distance of 
    # the point from origin is sqrt(2)
    sum = 0

    for point in points:
        sum += euclidean_distance(cog, point)

    avg_sum = sum / data_size

    d = np.sqrt(2) / avg_sum
    S = np.array([
        [d, 0, 0],
        [0, d, 0],
        [0, 0, 1]
    ])

    return np.dot(S, G)

def apply_transformation(points, transformation):
    result = []
    for point in points:
        result.append(np.dot(transformation, np.transpose(point)))
    
    return result

def euclidean_distance(a, b):
    return np.sqrt(((a[0] / a[2] - b[0] / b[2]) ** 2) + ((a[1] / a[2] - b[1] / b[2]) ** 2))