from numba import jit
import numpy as np
import matplotlib.pyplot as plt
visualization = True


def show_spheres(scale, points, rgb, label=None):
    """

    :param scale: int
    :param points: tuple (x, y, z)
    :param rgb:
    :return:
    """
    if label is not None:
        print('')
        print(label)

    points = np.stack([points[0].reshape(-1), points[1].reshape(-1), points[2].reshape(-1)], axis=1)

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # axis scale setting
    ax.set_xlim3d(-1 * scale, scale)
    ax.set_ylim3d(-1 * scale, scale)
    ax.set_zlim3d(-0.8 * scale, 0.8 * scale)

    x, y, z = 0, 0, 2
    ax.plot([0, scale * x], [0, scale * y], [0, scale * z])

    # label
    # ax.grid(False)
    ax.plot([0, scale * x], [0, scale * y], [0, scale * z])

    # if label is not None:
    #     x, y, z = label
    #
    #     # label
    #     ax.grid(False)
    #     ax.plot([0, scale * x], [0, scale * y], [0, scale * z])
    #
    #     # how rotate they are
    #     phi2 = np.arctan2(y, x) * 180 / np.pi
    #     theta = np.arccos(z) * 180 / np.pi
    #
    #     if phi2 < 0:
    #         phi2 = 360 + phi2

    r = rgb[0].reshape(-1)
    g = rgb[1].reshape(-1)
    b = rgb[2].reshape(-1)
    # rgb 0~1 scale
    r = (r - np.min(r)) / (np.max(r) - np.min(r))
    g = (g - np.min(g)) / (np.max(g) - np.min(g))
    b = (b - np.min(b)) / (np.max(b) - np.min(b))
    rgb = np.stack([r, g, b], axis=1)

    ax.scatter(points[:, 0], points[:, 1], points[:, 2], facecolors=rgb, alpha=1, depthshade=False,
               edgecolors=None,
               )  # data coloring
    plt.legend(loc=2)

    # Photos viewed at 90 degrees
    ax.view_init(0, 0)

    # Photos from above
    # ax.view_init(-1 * theta + 90, phi2)

    plt.draw()
    plt.show()


def rotate_map_given_R(R, height, width):
    # Inputs:
    #       phi,theta in degrees , height and width of an image
    # output:
    #       rotation map for x and y coordinate
    # goal:
    #       calculating rotation map for corresponding image dimension and phi,theta value.
    #       (1,0,0)(rho,phi,theta) on sphere goes to (1,phi,theta)

    def pos_conversion(x, y, z):
        # given postech protocol
        # return my protocol

        return z, -x, y

    def inv_conversion(x, y, z):
        # given my conversion
        # convert it to postech system.

        return -y, z, x

    # if not original_file.is_file():
    # step1
    spherePoints = flat_to_sphere(height, width)
    # R = calculate_Rmatrix_from_phi_theta(phi,theta)
    R_inv = np.linalg.inv(R)

    #step2
    spherePointsRotated = rotate_sphere_given_phi_theta(R_inv, spherePoints)

    #Create two mapping variable
    #step3
    [map_x, map_y] = sphere_to_flat(spherePointsRotated, height, width)

    # dst(y,x) = src(map_x(y,x),map_y(y,x))
    return [map_x, map_y]


@jit(nopython=True, cache=True)
def flat_to_sphere(height, width):
    # Input:
    #      height and width of image
    # Output:
    #      return (height,width,3) numpy ndarray. (y,x) of array has (x,y,z) value which is on sphere.
    # Goal:
    #      return sphere points
    # Create matrix that contains x,y,z coordinates

    sphere = np.zeros((height, width, 3))
    x_to_theta = np.zeros(width)
    y_to_phi = np.zeros(height)

    theta_slope = 2*np.pi/(width-1)
    phi_slope = np.pi/(height-1)

    # linear map from [y,x] to [phi,theta]
    for x in range(0, width):
        x_to_theta[x] = np.rad2deg(np.multiply(x, theta_slope))

    for y in range(0, height):
        y_to_phi[y] = np.rad2deg(np.multiply(y, phi_slope))

    # For every pixel coordinates, create a matrix that contains the
    # corresponding (x,y,z) coordinates
    for y_f in range(0, height):
        for x_f in range(0, width):
            theta = x_to_theta[x_f]
            phi = y_to_phi[y_f]

            phi = np.deg2rad(phi)
            theta = np.deg2rad(theta)
            x_s = np.sin(phi) * np.cos(theta)
            y_s = np.sin(phi) * np.sin(theta)
            z_s = np.cos(phi)
            sphere[y_f, x_f, 0] = x_s
            sphere[y_f, x_f, 1] = y_s
            sphere[y_f, x_f, 2] = z_s

    return sphere


@jit(nopython=True, cache=True)
def rotate_sphere_given_phi_theta(R, spherePoints):
    # Input:
    #       phi,theta in degrees and spherePoints(x,y,z of on sphere dimension (height,width,3) )
    # Output:
    #       spherePointsRotated of which dimension is (h,w,3) and contains (x',y',z' )
    #  (x',y',z')=R*(x,y,z) where R maps (0,0,1) to (vx,vy,vz) defined by theta,phi (i.e. R*(0,0,1)=(vx,vy,vz))
    # Goal:
    #      apply R to every point on sphere

    h, w, c = spherePoints.shape
    spherePointsRotated = np.zeros((h, w, c),dtype=np.float64)

    for y in range(0, h):
        for x in range(0, w):
            pointOnSphere = spherePoints[y, x, :]
            pointOnSphereRotated = np.dot(R, pointOnSphere)
            spherePointsRotated[y, x, :] = pointOnSphereRotated
            # spherePointsRotated[y, x, :] = np.dot(R, pointOnSphere)

    return spherePointsRotated


@jit(nopython=True, cache=True)
def calculate_Rmatrix_from_phi_theta(phi, theta):
    """
    A = [0,0,1] B = [x,y,z] ( = phi,theta) the goal is to find rotation matrix R where R*A == B
    please refer to this website https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d
    v = a cross b ,s = ||v|| (sine of angle), c = a dot b (cosine of angle)
    :param phi: z 축 각도
    :param theta: xy 축 각도
    :return: rotation matrix that moves [0,0,1] to ([x,y,z] that is equivalent to (phi,theta))
    """

    epsilon = 1e-7
    A = np.array([0, 0, 1], dtype=np.float64)  # original up-vector
    # B = spherical_to_cartesian(phi,theta)  # target vector

    phi = np.deg2rad(phi)
    theta = np.deg2rad(theta)
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    B = np.array([x, y, z], dtype=np.float64)

    desiredResult = B
    # dot(R,A) == B
    # If A == B then return identity(3)
    if A[0] - B[0] < epsilon \
            and A[0] - B[0] > -epsilon \
            and A[1] - B[1] < epsilon \
            and A[1] - B[1] > -epsilon \
            and A[2] - B[2] < epsilon \
            and A[2] - B[2] > -epsilon:
        # print('Identity matrix is returned')
        return np.identity(3)

    # v = np.cross(A, B)
    # In the numba, numpy.cross is not supported
    cross_1 = np.multiply(A[1],B[2])-np.multiply(A[2],B[1])
    cross_2 = np.multiply(A[2],B[0])-np.multiply(A[0],B[2])
    cross_3 = np.multiply(A[0],B[1])-np.multiply(A[1],B[0])
    v = np.array([cross_1,cross_2,cross_3])

    c = np.dot(A, B)
    skewSymmetric = skewSymmetricCrossProduct(v)

    if -epsilon < c + 1 and c + 1 < epsilon:
        R = -np.identity(3)
    else:
        R = np.identity(3) + skewSymmetric + np.dot(skewSymmetric, skewSymmetric) * (
                    1 / (1 + c))  # what if 1+c is 0?
    return R


@jit(nopython=True, cache=True)
def skewSymmetricCrossProduct(v):
    # Input:
    #   a vector in R^3
    # Output:
    #   [ 0 -v3 v2 ; v3 0 -v1; -v2 v1 0]
    v1 = v[0]
    v2 = v[1]
    v3 = v[2]

    skewSymmetricMatrix = np.array([[0, -v3, v2], [v3, 0, -v1], [-v2, v1, 0]], dtype=np.float64)

    return skewSymmetricMatrix


@jit(nopython=True, cache=True)
def sphere_to_flat(spherePointsRotated, height, width):
    # Input:
    #       y,x coordinate on 2d flat image,numpy nd array of dimension (height,width,3). ndarray(y,x) has x,y,z value on sphere ,height and width of an image
    # Output:
    #       x,y coordinate of 2d flat image
    # Goal:
    #       calculate destination x,y coordinate given information x,y(2d flat) <-> x,y,z(sphere)
    map_y = np.zeros((height, width), dtype=np.float32)
    map_x = np.zeros((height, width), dtype=np.float32)

    factor_phi = (height-1)/np.pi
    factor_theta = (width-1)/(2*np.pi)

    # Get multiplied(by inverted rotation matrix) x,y,z coordinates
    for image_y in range(0, height):
        for image_x in range(0, width):
            pointOnRotatedSphere_x = spherePointsRotated[image_y, image_x, 0]
            pointOnRotatedSphere_y = spherePointsRotated[image_y, image_x, 1]
            pointOnRotatedSphere_z = spherePointsRotated[image_y, image_x, 2]

            x_2 = np.power(pointOnRotatedSphere_x, 2)
            y_2 = np.power(pointOnRotatedSphere_y, 2)
            z_2 = np.power(pointOnRotatedSphere_z, 2)

            theta = float(np.arctan2(pointOnRotatedSphere_y, pointOnRotatedSphere_x))
            # atan2 returns value of which range is [-pi,pi], range of theta is [0,2pi] so if theta is negative value,actual value is theta+2pi
            if theta < 0:
                theta = theta + np.multiply(2, np.pi)

            rho = x_2 + y_2 + z_2
            rho = np.sqrt(rho)
            phi = np.arccos(pointOnRotatedSphere_z / rho)

            map_y[image_y, image_x] = phi*factor_phi
            map_x[image_y, image_x] = theta*factor_theta

    return [map_x, map_y]
