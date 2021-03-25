import numpy as np
from numpy.linalg import inv
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt


def project_pixel_to_ground(pixel, camera_matrix , transform_LI, ground_normal):
    """Project a point in the image plane to ground in the localization plane"""
    
    pixel_in_sensor = np.array([pixel[0], pixel[1], 1])
    pixel_I = np.ones(4)
    pixel_I[0:3] = inv(camera_matrix).dot(pixel_in_sensor)
    pixel_L = transform_LI.dot(pixel_I)[0:3]

    focal_point_I = np.array([0,0,0,1])
    focal_point_L = transform_LI.dot(focal_point_I)[0:3]

    distance_scaling = - ( focal_point_L.dot(ground_normal)
                          / (pixel_L-focal_point_L).dot(ground_normal) )

    point_L = focal_point_L + distance_scaling * (pixel_L - focal_point_L)

    return tuple(point_L)

def pixel_to_body(pixel, camera_matrix , transform_LI):
    pixel_in_sensor = np.array([pixel[0], pixel[1], 1])
    pixel_I = np.ones(4)
    pixel_I[0:3] = inv(camera_matrix).dot(pixel_in_sensor)
    pixel_L = transform_LI.dot(pixel_I)
    return tuple(pixel_L[0:3])

def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

if __name__ == "__main__":
    ### Transformation from localization frame to camera-image frame ###
    # Frame L is localization plane-frame, C is camera frame, I is image plane, 
    # which is oriented z along line of sight, x right and y down.
    transform_LC = np.identity(4)
    # Camera is raised 4 m along the z axis
    transform_LC[0:3, 3] = [0, 0, 4]
    # In this experiment, the camera is pointing 1 radians down towards the ground
    rot_BC = R.from_rotvec(1. * np.array([0, 1, 0]))
    transform_LC[0:3, 0:3] = rot_BC.as_matrix()

    # The image plane is oriented to point z along the line of sight, x right
    # and y down seen from the C frame.
    transform_CI = np.array([[ 0,  0,  1, 0],
                             [-1,  0,  0, 0],
                             [ 0, -1,  0, 0],
                             [ 0,  0,  0, 1]])

    transform_LI = transform_LC.dot(transform_CI)

    # The camera is a 640x480, data taken from Spot CameraInfo
    camera_matrix = np.array([[257.56,    0.0, 323.62],
                              [0.0,    257.01, 245.47],
                              [0.0,       0.0,    1.0]])

    pixels = [(0, 0), (0, 480), (640, 480), (640, 0)]
    camera_frame_B = []
    ground_points = []
    for p in pixels:
        camera_frame_B.append(pixel_to_body(p, camera_matrix, transform_LI))
        ground_points.append(project_pixel_to_ground(p, camera_matrix, transform_LI, [0, 0, 1]))
    fig = plt.figure()
    ax =  fig.add_subplot(111, projection='3d')
    ax.plot(*zip(*camera_frame_B))
    ax.plot(*transform_LC[0:3, 3], '*')
    ax.plot(*zip(*ground_points))
    ax.plot(*project_pixel_to_ground((320, 240),camera_matrix, transform_LI, [0, 0, 1]), '*')

    set_axes_equal(ax)

    plt.show()
