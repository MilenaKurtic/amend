import algorithms as alg
import numpy      as np

def main():
    points = np.array([
        [-3, -1], # A 
        [ 3, -1], # B
        [ 1,  1], # C
        [-1,  1]  # D
    ])

    points_img = np.array([
        [ -2, -1], # A'
        [  2, -1], # B'
        [  2,  1], # C'
        [ -2,  1]  # D'
    ])

    print("Naive algorithm:")
    print(np.around(alg.naive_algorithm(points, points_img), 5))
    print("")

    points = np.array([
        [-3, -1, 1], # A
        [ 3, -1, 1], # B
        [ 1,  1, 1], # C
        [-1,  1, 1], # D
        [ 1,  2, 3], # E
        [-8, -2, 1]  # F
    ])

    points_img = np.array([
        [ -2, -1, 1], # A'
        [  2, -1, 1], # B'
        [  2,  1, 1], # C'
        [ -2,  1, 1], # D'
        [  2,  1, 4], # E'
        [-16, -5, 4]  # F'
    ])
    
    print("DLT algorithm:")
    print(np.around(alg.dlt_algorithm(points, points_img), 5))
    print("")

    print("Normalized DLT algorithm:")
    print(np.around(alg.dlt_normalized_algorithm(points, points_img), 5))
    print("")

if __name__ == "__main__":
    main()
