import numpy as np
import numpy.linalg as linalg
import matplotlib.pyplot as plt

import helpers as helper

def naive_algorithm(points, points_img):
    if __debug__:
        print(" ##### Naive algorithm #####\n")

    # convert points coordinates to homogeneous coordinates if needed
    points     = helper.convert_to_homogeneous(points)
    points_img = helper.convert_to_homogeneous(points_img)

    # set points names to make things more readable
    A = points[0]
    B = points[1]
    C = points[2]
    D = points[3]

    Ap = points_img[0]
    Bp = points_img[1]
    Cp = points_img[2]
    Dp = points_img[3]

    # configure plot and add subplots
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.title("Points")
    plt.plot(
        [A[0], B[0], C[0], D[0], A[0]],
        [A[1], B[1], C[1], D[1], A[1]],
    )

    plt.subplot(1, 2, 2)
    plt.title("Points images")
    plt.plot(
        [Ap[0], Bp[0], Cp[0], Dp[0], Ap[0]],
        [Ap[1], Bp[1], Cp[1], Dp[1], Ap[1]],
    )
    
    plt.show()

    P  = helper.get_p_matrix(points)
    Pp = helper.get_p_matrix(points_img)
    if __debug__:
        print("Transformation matrix P =")
        print(P)
        print("")

        print("Transformation matrix P\' =")
        print(Pp)
        print("")

    result = np.dot(Pp, linalg.inv(P))
    
    if __debug__:
        print("Naive algorithm result = ")
        print(np.around(result, 4))
        print("")

    return result

def dlt_algorithm(points, points_img):
    if __debug__:
        print(" ##### DLT algorithm #####\n")

    # convert points coordinates to homogeneous coordinates if needed
    points     = helper.convert_to_homogeneous(points)
    points_img = helper.convert_to_homogeneous(points_img)

    # prepare data for SVD decomposition
    data_size = np.shape(points)[0]
    prep_data = []

    for i in range(data_size):
        prep_data.append((points[i], points_img[i]))

    # create A matrix for svd 
    A = helper.make_2x9_matrix(prep_data[0]) # np.concatenate requires equal array shapes
    for i in range(1, data_size):
        tmp = helper.make_2x9_matrix(prep_data[i])
        A   = np.concatenate((A, tmp))
    
    # apply svd decomposition on matrix A
    u, d, vt = linalg.svd(A)

    # print out result
    result = vt[-1].reshape(3, 3)

    if __debug__:
        print("DLT algorithm result =")
        print(np.around(result, 4))
        print("")

    return result

def dlt_normalized_algorithm(points, points_img):
    if __debug__:
        print(" ##### Normalized DLT algorithm #####\n")

    # convert points coordinates to homogeneous coordinates if needed
    points     = helper.convert_to_homogeneous(points)
    points_img = helper.convert_to_homogeneous(points_img)

    # apply transformation on points and points images
    T  = helper.normalize_points(points)
    Tp = helper.normalize_points(points_img)

    new_points     = helper.apply_transformation(points, T)
    new_points_img = helper.apply_transformation(points_img, Tp)

    # get matrix map
    Pp = dlt_algorithm(new_points, new_points_img)
    
    # result matrix equals (Tp ^ -1) @ Pp @ T
    result = np.dot(np.dot(linalg.inv(Tp), Pp), T)
    result = np.array(result, np.float32)
    # print out result
    
    if __debug__:
        print("Normalized DLT algorithm result =")
        print(np.around(result, 4))
        print("")

    return result
