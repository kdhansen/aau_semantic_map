import numpy as np
from numpy import pi
from numpy.linalg import inv, multi_dot
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

def homography_projection(pixel, camera_transform, plane_transform, camera_intrinsics):
    """Compute the homographic projection of a pixel from a camera to a plane.

    Given the world transformations of a camera and a projection plane, this
    computes the homography that projects a pixel in the image plane to a point
    on the projection plane.
    
    The camera world transformation is a 4x4 SE(3) transformation from the world
    frame to the camera frame, before applying camera intrinsics.

    The plane world transform is also an SE(3). If the pixel is projected to the
    world frame, this transform is just the identity matrix, e.g.:
        np.identity(4).
    
    All three matrices must be passed as Numpy arrays.
    
    The camera intrinsics are dependent on the camera hardware, usually
    available from the camera metadata or from calibration.

    The returned point is given in world coordinates.

    See more in the paper: "Robot Cigarette-Litter Identification and Removal"

    Example:
        The camera is 1.5 meters above ground and pointed 45 degrees downwards,
        the pixel (323,205) is projected to the world plane. The used camera is
        a 640x480 px with a focal length of 257 px. 
        
        cam_transform = np.identity(4)
        cam_transform[0:3, 3] = [0, 0, 1.5]
        cam_transform[0:3, 0:3] = np.array([[  cos(45),  0,  sin(45)],
                                            [        0,  0,        0],
                                            [ -sin(45), -1,  cos(45)]])
        intrinsics = np.array([[257,   0, 320],
                               [  0, 257, 240],
                               [  0,   0,   1]])
        point_in_plane = homography_projection([323,205], cam_transform,
                                               np.identity(4), intrinsics)

    Note: The rotation matrix here breaks the syntax a bit. Both numpy.cos and
          math.cos use radians. You could use scipy.spatial.transform.Rotation
          to generate the matrix.
    """
    # Transformations
    # Just recasting to T_xx notation. T_WC is the transformation of the camera
    # frame in the world frame, and T_WP is the transformation of the projecting
    # plane in the world frame.
    T_WC = camera_transform
    T_WP = plane_transform

    # The image plane, I, is oriented to point z along the line of sight,
    # x right and y down seen from the camera frame, C.
    T_CI = np.array([[ 0,  0,  1, 0],
                     [-1,  0,  0, 0],
                     [ 0, -1,  0, 0],
                     [ 0,  0,  0, 1]])

    # The camera intrinsics is the transformation from a point in the image
    # plane to a point in the sensor plane. This is opposite the direction we
    # want, so we invert it.
    T_IS = inv(camera_intrinsics)

    # Convert pixel and camera focal point to the projection plane frame. The
    # pixel is given in the sensor frame (homogeneous 2D) (r_S), so this is
    # first transformed into the homogeneous 3D image plane frame (r_I). Then
    # both the pixel and focal point is converted through the frames
    # camera->world->projection plane (r_I->r_P and f_I->f_P). The focal point,
    # f_I, is is the origin in the image frame.
    T_PI = multi_dot([inv(T_WP), T_WC, T_CI])
    
    f_I = np.array([0., 0., 0., 1.])
    f_P = T_PI.dot(f_I)

    r_S = np.array([pixel[0], pixel[1], 1])
    r_I = np.ones(4)
    r_I[0:3] = T_IS.dot(r_S)
    r_P = T_PI.dot(r_I)

    # The distance scalar use only the z-components of r and f. 
    z_fP = f_P[2]
    z_rP = r_P[2]
    d = z_fP / (z_fP - z_rP)

    # Scaling the pixel in P to project it onto the P plane.
    p_P = f_P + d*(r_P-f_P)

    # Returning the point in world frame as a regular list (not homogeneous) of three coordinates.
    p_W = T_WP.dot(p_P)
    print("p_P:", p_P)
    print("T_WP: ", T_WP)
    print("p_W:", p_W)
    return list(p_W[0:3])
    

def pixel_in_image_plane(pixel, camera_matrix, camera_transform):
    pixel_in_sensor = np.array([pixel[0], pixel[1], 1])
    pixel_I = np.ones(4)
    pixel_I[0:3] = inv(camera_matrix).dot(pixel_in_sensor)
    T_CI = np.array([[ 0,  0,  1, 0],
                     [-1,  0,  0, 0],
                     [ 0, -1,  0, 0],
                     [ 0,  0,  0, 1]])
    pixel_W = camera_transform.dot(T_CI.dot(pixel_I))
    return tuple(pixel_W[0:3])

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
    ### Transformation from localization frame to camera-image frame ###
    # Frame P is localization plane-frame, C is camera frame, I is image plane, 
    # which is oriented z along line of sight, x right and y down.
    camera_transform = np.identity(4)
    # Camera is raised 4 m along the z axis
    camera_transform[0:3, 3] = [0, 0, 4]
    # In this experiment, the camera is pointing 1.5 radians down towards the ground
    rot_C = R.from_rotvec(1.5 * np.array([0, 1, 0]))
    camera_transform[0:3, 0:3] = rot_C.as_matrix()

    # The camera is a 640x480
    camera_matrix = np.array([[250.,   0., 320.],
                              [  0., 250., 240.],
                              [  0.,   0.,   1.]])

    # plane tranform
    # plane is raised 1 m along the z axis and rotated slightly around x.
    plane_transform = np.identity(4)
    plane_transform[0:3, 3] = [0, 0, 1]
    rot_P = R.from_rotvec(0.2 * np.array([1, 0, 0]))
    plane_transform[0:3, 0:3] = rot_P.as_matrix()


    # outline of the frame
    pixels = [(0, 0), (0, 480), (640, 480), (640, 0)]
    camera_frame_box = []
    ground_points = []
    plane_points = []
    for p in pixels:
        camera_frame_box.append(pixel_in_image_plane(p, camera_matrix , camera_transform))
        ground_points.append(homography_projection(p, camera_transform, np.identity(4), camera_matrix))
        plane_points.append(homography_projection(p, camera_transform, plane_transform, camera_matrix))

    fig = plt.figure()
    ax =  fig.add_subplot(111, projection='3d')
    ax.plot(*zip(*camera_frame_box))
    ax.plot(*camera_transform[0:3, 3], '*')
    ax.plot(*zip(*ground_points))
    ax.plot(*zip(*plane_points))

    test_point = [200,200]
    tp_C = pixel_in_image_plane(test_point, camera_matrix , camera_transform)
    tp_P = homography_projection(test_point, camera_transform, plane_transform, camera_matrix)
    ax.plot(*tp_C, '*')
    ax.plot(*tp_P, '*')

    set_axes_equal(ax)

    plt.show()
