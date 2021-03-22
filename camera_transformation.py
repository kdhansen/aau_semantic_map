import numpy as np
from numpy.linalg import inv
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

def project_pixel_to_ground(p):
    ### Transformation from body to camera-image frame ###
    # Camera is raised 1 m along the z axis
    translation_AB = np.identity(4)
    translation_AB[0:3, 3] = [0, 0, -1]

    # In this experiment, the camera is pointing 45 deg down towards the ground
    rot_CA = np.identity(4)
    rot_CA[0:3, 0:3] = R.from_rotvec(-np.pi/4 * np.array([0, 1, 0])).as_matrix()

    # The image plane is oriented to point z along the line of sight, x right
    # and y down.
    transform_IC = np.array([[0, -1,  0, 0],
                            [0,  0, -1, 0],
                            [1,  0,  0, 0],
                            [0,  0,  0, 1]])

    transform_IB = transform_IC.dot(rot_CA).dot(translation_AB)

    # Body center
    body_center_in_B = np.array([0, 0, 0, 1])
    body_center_in_I = transform_IB.dot(body_center_in_B)
    # print("Body center in B: {}\nBody center in I: {}".format(body_center_in_B, body_center_in_I))

    ### The camera is a 640x480 ###
    camera_matrix = np.array([[257.56,    0.0, 323.62],
                            [   0.0, 257.01, 245.47],
                            [   0.0,    0.0,    1.0]])

    pixel_in_sensor = np.array([p[0], p[1], 1])
    pixel_in_image = inv(camera_matrix).dot(pixel_in_sensor)
    # print("Pixel in I :{}".format(pixel_in_image))

    ### The normal to the surface ###
    plane_normal_in_B = np.array([0, 0, 1, 0])
    plane_normal_in_I = transform_IB.dot(plane_normal_in_B)
    # print("Plane normal in B: {}\nPlane normal in I: {}".format(plane_normal_in_B, plane_normal_in_I))

    point_in_I = np.ones(4)
    point_in_I[0:3] = ( pixel_in_image
                    * (body_center_in_I[0:3].dot(plane_normal_in_I[0:3]))
                    / ((pixel_in_image.dot(plane_normal_in_I[0:3]))) )
    point_in_B = inv(transform_IB).dot(point_in_I)

    # print("Point in I: {}\nPoint in B: {}".format(point_in_I, point_in_B))

    return tuple(point_in_B[0:2])

if __name__ == "__main__":
    pixels = [(0,0), (0, 480), (640, 480), (640, 0)]
    ground_points = []
    for p in pixels:
        ground_points.append(project_pixel_to_ground(p))
    fig, ax = plt.subplots(1,1)
    ax.plot(*zip(*ground_points))
    ax.plot(*project_pixel_to_ground((320, 240)), '*')
    ax.axis("equal")
    plt.show()