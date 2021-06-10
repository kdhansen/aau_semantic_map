from semanticmap import HomographicProjection
import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

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

    x_range = abs(x_limits[1] - x_limits[0])*1.5
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])*1.5
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])*1.5
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

if __name__ == "__main__":
    """Show a camera frame as a projected bounding box on a plane

    Several frames are used:
        W - World
        C - Camera
        P - Plane (The one we are projecting to.)
    """
    # Transformation from world frame to camera-image frame. Our camera is
    # raised 4 m along the z axis In this experiment, the camera is pointing 1.5
    # radians down towards the ground
    camera_transform = np.identity(4)
    camera_transform[0:3, 3] = [0, 0, 4]
    rot_C = R.from_rotvec(1.3 * np.array([0, 1, 0]))
    camera_transform[0:3, 0:3] = rot_C.as_matrix()

    # The camera is a 640x480
    camera_matrix = np.array([[250.,   0., 320.],
                              [  0., 250., 240.],
                              [  0.,   0.,   1.]])

    # Plane tranform. The plane is raised 1 m along the z axis and rotated
    # slightly around x.
    plane_transform = np.identity(4)
    plane_transform[0:3, 3] = [0, 0, 1]
    rot_P = R.from_rotvec(0.1 * np.array([1, 0, 0]))
    plane_transform[0:3, 0:3] = rot_P.as_matrix()

    # Constructing homographies for both the plane and the ground.
    H_ground = HomographicProjection(camera_transform, np.identity(4), camera_matrix)
    H_plane = HomographicProjection(camera_transform, plane_transform, camera_matrix)

    # Outline of the frame
    pixels = [(0, 0), (0, 480), (640, 480), (640, 0)]
    camera_frame_box = []
    ground_points = []
    plane_points = []
    for p in pixels:
        camera_frame_box.append(H_plane.pixel_in_image_plane(p))
        ground_points.append(H_ground.project_to_plane(p))
        plane_points.append(H_plane.project_to_plane(p))

    # Plot it all
    fig = plt.figure()
    ax =  fig.add_subplot(111, projection='3d')
    ax.plot(*zip(*camera_frame_box))
    ax.plot(*camera_transform[0:3, 3], '*')
    ax.plot(*zip(*ground_points))
    ax.plot(*zip(*plane_points))

    test_point = [200,200]
    tp_C = H_plane.pixel_in_image_plane(test_point)
    tp_P = H_plane.project_to_plane(test_point)
    ax.plot(*tp_C, '*')
    ax.plot(*tp_P, '*')

    set_axes_equal(ax)

    plt.show()
