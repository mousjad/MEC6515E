import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R
import trimesh

# Load data

p_face_A = np.loadtxt('Resultats_pts_Face_A.asc', usecols=(4, 5, 6))
p_4_trou = np.loadtxt('Resultats_pts_4 trous.asc', usecols=(4, 5, 6))
p_Marbre = np.loadtxt('Resultats_pts_Marbre.asc', usecols=(4, 5, 6))
p_plan_B = np.loadtxt('Resultats_pts_planB.asc', usecols=(4, 5, 6))
p_plan_C = np.loadtxt('Resultats_pts_planC.asc', usecols=(4, 5, 6))
p_plan_D = np.loadtxt('Resultats_pts_planD.asc', usecols=(4, 5, 6))
p_trou_E = np.loadtxt('Resultats_pts_Trou_E.asc', usecols=(4, 5, 6))


def perpendicularityABC(p_plane_a, p_plane_b, p_plane_c, tolerance):
    k = least_square(p_plane_a)[:, 2]
    j = least_square(p_plane_b)[:, 2]
    i = np.cross(k, j)

    barycenter_c = np.zeros([1, 3])

    barycenter_c[0, 0] = sum(p_plane_c[:, 0]) / p_plane_c.shape[0]
    barycenter_c[0, 1] = sum(p_plane_c[:, 1]) / p_plane_c.shape[0]
    barycenter_c[0, 2] = sum(p_plane_c[:, 2]) / p_plane_c.shape[0]

    b_plane_c = p_plane_c - barycenter_c

    dev = np.dot(i, b_plane_c.T)
    tol_max = np.ones(dev.shape) * tolerance / 2
    tol_min = np.ones(dev.shape) * -tolerance / 2
    dev_max = np.ones(dev.shape) * np.max(dev)
    dev_min = np.ones(dev.shape) * np.min(dev)

    plt.plot(dev, 'rx', label='Points on the plane')
    plt.plot(tol_max, 'r-', label='Conformity Zone')
    plt.plot(tol_min, 'r-')
    plt.plot(dev_max, 'g-', label='Deviation Zone')
    plt.plot(dev_min, 'g-')
    plt.title('Deviation of perpendicularity')
    plt.legend(loc='upper right')
    plt.ylabel('Deviation (inch)')
    plt.xlabel('Point number')

    if any(abs(dev) > tolerance / 2):
        print('The specified plane is not perpendicular enough')
        print('The max deviation is: ', np.max(dev)-np.min(dev))
    else:
        print('The specified plane respects the tolerance')
        print('The max deviation is: ',np.max(dev)-np.min(dev))

    plt.show()

    return dev


def angularityABC(p_plane_a, p_plane_b, p_plane_c, angle, tolerance):
    k = least_square(p_plane_a)[:, 2]
    j = least_square(p_plane_b)[:, 2]
    i = np.cross(k, j)

    barycenter_c = np.zeros([1, 3])

    barycenter_c[0, 0] = sum(p_plane_c[:, 0]) / p_plane_c.shape[0]
    barycenter_c[0, 1] = sum(p_plane_c[:, 1]) / p_plane_c.shape[0]
    barycenter_c[0, 2] = sum(p_plane_c[:, 2]) / p_plane_c.shape[0]

    b_plane_c = p_plane_c - barycenter_c

    norm = rot_axis((angle - 90) * np.pi / 180, j).dot(i)
    dev = np.dot(norm, b_plane_c.T)
    tol_max = np.ones(dev.shape) * tolerance / 2
    tol_min = np.ones(dev.shape) * -tolerance / 2
    dev_max = np.ones(dev.shape) * np.max(dev)
    dev_min = np.ones(dev.shape) * np.min(dev)

    plt.plot(dev, 'rx', label='Points on the plane')
    plt.plot(tol_max, 'r-', label='Conformity Zone')
    plt.plot(tol_min, 'r-')
    plt.plot(dev_max, 'g-', label='Deviation Zone')
    plt.plot(dev_min, 'g-')
    plt.title('Deviation of angularity')
    plt.legend(loc='upper right')
    plt.ylabel('Deviation (inch)')
    plt.xlabel('Point number')

    if any(abs(dev) > tolerance / 2):
        print('The specified plane doesn''t respect the tolerance')
        print('The max deviation is: ', np.max(dev)-np.min(dev))
    else:
        print('The specified plane respects the tolerance')
        print('The max deviation is: ', np.max(dev)-np.min(dev))

    plt.show()

    return dev


def rot_axis(theta, u):
    return np.array([
        [np.cos(theta) + u[0] ** 2 * (1 - np.cos(theta)), u[0] * u[1] * (1 - np.cos(theta)) - u[2] * np.sin(theta),
         u[0] * u[2] * (1 - np.cos(theta)) + u[1] * np.sin(theta)],
        [u[1] * u[0] * (1 - np.cos(theta)) + u[2] * np.sin(theta), np.cos(theta) + u[1] ** 2 * (1 - np.cos(theta)),
         u[1] * u[2] * (1 - np.cos(theta)) - u[0] * np.sin(theta)],
        [u[2] * u[0] * (1 - np.cos(theta)) - u[1] * np.sin(theta),
         u[2] * u[1] * (1 - np.cos(theta)) + u[0] * np.sin(theta), np.cos(theta) + u[2] ** 2 * (1 - np.cos(theta))]
    ]).T


def least_square(p_plane):
    P_o_0 = np.zeros([1, 3])

    P_o_0[0, 0] = sum(p_plane[:, 0]) / p_plane.shape[0]
    P_o_0[0, 1] = sum(p_plane[:, 1]) / p_plane.shape[0]
    P_o_0[0, 2] = sum(p_plane[:, 2]) / p_plane.shape[0]
    b_plane = p_plane - P_o_0
    delta = P_o_0

    i_0 = np.ones((1, 3))
    j_0 = np.ones((1, 3))
    k_0 = np.ones((1, 3))

    while np.any(np.isnan(i_0)) or np.any(np.isnan(j_0)) or np.any(np.isnan(k_0)) or np.all(i_0 == 1):
        a = 0  # np.random.random_integers(0, p_plane.shape[0] - 1)
        b = 1  # np.random.random_integers(0, p_plane.shape[0] - 1)
        c = p_plane.shape[0] - 1  # np.random.random_integers(0, p_plane.shape[0] - 1)

        if np.linalg.norm(p_plane[b, :] - p_plane[a, :]) != 0 and np.linalg.norm(
                np.cross(i_0, (p_plane[c, :] - p_plane[a, :]))) != 0 \
                and not np.isnan(np.linalg.norm(p_plane[b, :] - p_plane[a, :])) and not np.isnan(
            np.linalg.norm(np.cross(i_0, (p_plane[c, :] - p_plane[a, :])))):
            i_0 = (p_plane[b, :] - p_plane[a, :]) / np.linalg.norm(p_plane[b, :] - p_plane[a, :])
            k_0 = (np.cross(i_0, (p_plane[c, :] - p_plane[a, :]))) / np.linalg.norm(
                np.cross(i_0, (p_plane[c, :] - p_plane[a, :])))
            j_0 = np.cross(k_0, i_0)

    axis = np.row_stack((i_0, j_0, k_0))
    P_1 = axis.dot(p_plane.T - delta.T).T
    rms = 1000
    rms_prec = rms + 1
    while rms - rms_prec <= -1e-10:
        X = P_1[:, 0]
        Y = P_1[:, 1]
        Z = P_1[:, 2]

        var = np.column_stack((np.ones((P_1.shape[0])), Y, -X))
        tau = np.matmul(np.linalg.pinv(var), Z)

        rot = R.from_euler('xy', (tau[1], tau[2])).as_matrix().T
        delta += axis.dot(np.array([[0, 0, tau[0]]]).T).T

        P_1 = rot.dot(axis.dot(p_plane.T - delta.T)).T
        axis = rot.dot(axis)
        u = axis.T
        P = u[:, 2].dot(b_plane.T)
        rms_prec = rms
        rms = np.sqrt(np.mean(P ** 2))

    return axis.T


# perpendicularityABC(p_Marbre, p_plan_B, p_plan_C, 0.004)
# angularityABC(p_Marbre, p_plan_B, p_plan_C, -90, 0.004)  # should be the same as previous line
# angularityABC(p_Marbre, p_plan_C, p_plan_D, 45, 0.005)

syn_plan_A = np.array(trimesh.load('A_plane.stl').vertices)
syn_plan_B = np.array(trimesh.load('B_plane.stl').vertices)
syn_plan_C = np.array(trimesh.load('C_plane_perf.stl').vertices)
syn_plan_C_dev = np.array(trimesh.load('C_plane_dev.stl').vertices)
syn_plan_D = np.array(trimesh.load('D_plane_perf.stl').vertices)
syn_plan_D_dev = np.array(trimesh.load('D_plane_dev.stl').vertices)

angularityABC(syn_plan_A, syn_plan_B, syn_plan_D, -45, 0.005)
print('Expected deviation is 0 inch')
angularityABC(syn_plan_A, syn_plan_B, syn_plan_D_dev, -45, 0.005)
print('Expected deviation is 0.7 inch')

perpendicularityABC(syn_plan_A, syn_plan_B, syn_plan_C, 0.005)
print('Expected deviation is 0 inch')
perpendicularityABC(syn_plan_A, syn_plan_B, syn_plan_C_dev, 0.005)
print('Expected deviation is 0.7 inch')