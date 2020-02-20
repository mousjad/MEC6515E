import matplotlib.pyplot as plt
import numpy as np

# Load data

p_face_A = np.loadtxt('Resultats_pts_Face_A.asc', usecols=(4, 5, 6))
p_4_trou = np.loadtxt('Resultats_pts_4 trous.asc', usecols=(4, 5, 6))
p_Marbre = np.loadtxt('Resultats_pts_Marbre.asc', usecols=(4, 5, 6))
p_plan_B = np.loadtxt('Resultats_pts_planB.asc', usecols=(4, 5, 6))
p_plan_C = np.loadtxt('Resultats_pts_planC.asc', usecols=(4, 5, 6))
p_plan_D = np.loadtxt('Resultats_pts_planD.asc', usecols=(4, 5, 6))
p_trou_E = np.loadtxt('Resultats_pts_Trou_E.asc', usecols=(4, 5, 6))

p_face_A_SYN = np.loadtxt('Resultats_pts_Face_A_SYN.asc', usecols=(4, 5, 6))
p_plan_B_SYN = np.loadtxt('Resultats_pts_planB_SYN.asc', usecols=(4, 5, 6))


def planarity(p_plane, tolerance):
    barycenter_plane = np.zeros([1, 3])

    barycenter_plane[0, 0] = sum(p_plane[:, 0]) / p_plane.shape[0]
    barycenter_plane[0, 1] = sum(p_plane[:, 1]) / p_plane.shape[0]
    barycenter_plane[0, 2] = sum(p_plane[:, 2]) / p_plane.shape[0]

    b_plane = p_plane - barycenter_plane
    b_barycenter_plane = barycenter_plane - barycenter_plane

    u, s, vh = np.linalg.svd(np.transpose(b_plane), full_matrices=True)

    dev = np.dot(b_plane, u[:, 2])
    dev_max = np.ones(dev.shape) * tolerance / 2
    dev_min = np.ones(dev.shape) * -tolerance / 2

    plt.plot(dev, 'rx', label='Point on the plane')
    plt.plot(dev_max, label='Tolerance zone')
    plt.plot(dev_min, label='Tolerance zone')
    plt.title('Deviation of planeity')
    plt.legend(loc='upper right')
    plt.ylabel('Deviation')
    plt.xlabel('Point number')

    if any(abs(dev) > tolerance / 2):
        print('The specified plane is not planar enough')
        print('The max deviation is: ', np.max(abs(dev)))
    else:
        print('The specified respects the tolerance')
        print('The max deviation is: ', np.max(abs(dev)))

    plt.show()
    return dev


def perpendicularityAB(p_plane_a, p_plane_perpendicular, tolerance):
    barycenter_plane_A = np.zeros([1, 3])

    barycenter_plane_A[0, 0] = sum(p_plane_a[:, 0]) / p_plane_a.shape[0]
    barycenter_plane_A[0, 1] = sum(p_plane_a[:, 1]) / p_plane_a.shape[0]
    barycenter_plane_A[0, 2] = sum(p_plane_a[:, 2]) / p_plane_a.shape[0]

    b_plane_A = p_plane_a - barycenter_plane_A
    b_barycenter_plane_A = barycenter_plane_A - barycenter_plane_A

    u, s, vh = np.linalg.svd(np.transpose(b_plane_A), full_matrices=True)

    b_plane_perpendicular = p_plane_perpendicular - barycenter_plane_A

    proj = b_plane_perpendicular - np.outer(
        np.reshape(b_plane_perpendicular.dot(u[:, 2]), [b_plane_perpendicular.shape[0], 1]), u[:, 2])

    barycenter_proj = np.zeros([1, 3])

    barycenter_proj[0, 0] = sum(proj[:, 0]) / proj.shape[0]
    barycenter_proj[0, 1] = sum(proj[:, 1]) / proj.shape[0]
    barycenter_proj[0, 2] = sum(proj[:, 2]) / proj.shape[0]

    b_proj = proj - barycenter_proj

    u_2, s_2, vh_2 = np.linalg.svd(np.transpose(b_proj), full_matrices=True)

    dev_2 = np.dot(b_proj, u_2[:, 1])
    dev_max = np.ones(dev_2.shape) * tolerance / 2
    dev_min = np.ones(dev_2.shape) * -tolerance / 2

    plt.plot(dev_2, 'rx', label='Point on the plane')
    plt.plot(dev_max, label='Tolerance zone')
    plt.plot(dev_min, label='Tolerance zone')
    plt.title('Deviation of perpendicularity')
    plt.legend(loc='upper right')
    plt.ylabel('Deviation')
    plt.xlabel('Point number')

    if any(abs(dev_2) > tolerance / 2):
        print('The specified plane is not perpendicular enough')
        print('The max deviation is: ', np.max(abs(dev_2)))
    else:
        print('The specified respects the tolerance')
        print('The max deviation is: ', np.max(abs(dev_2)))

    plt.show()

    return dev_2

#
# # Planarity
# # Real face A
#
# planarity(p_face_A, 0.002)
#
# # Synthetic face
#
# nx = np.linspace(0, 4, 5)
# ny = np.linspace(0, 2, 5)
# zv = 0.0002*(np.random.random(25)-0.5)
# # nz[np.random.random_integers(0,4)] = 0.001
# xv, yv = np.meshgrid(nx, ny)
# xyz = np.column_stack([xv.flatten(), yv.flatten(), zv.flatten()])
#
# planarity(p_face_A_SYN, 0.002)
#
# # Perpendicularity
# # Real Marbre and face b
#
# perpendicularityAB(p_Marbre, p_plan_B, 0.004)
#
# # Sythetic perpendicularity
#
# nx = np.linspace(0, 4, 5)
# ny = np.linspace(0, 2, 5)
# zv = 0.0004*(np.random.random(25)-0.5)
# # nz[np.random.random_integers(0,25)] = 0.001
# xv, yv = np.meshgrid(nx, ny)
# xyz = np.column_stack([xv.flatten(), yv.flatten(), zv.flatten()])
#
# R_y = np.array([
#     [0, 0, 1],
#     [0, 1, 0],
#     [-1, 0, 0]
# ])
#
# xyz_perpendicular = xyz @ R_y
#
# perpendicularityAB(p_face_A_SYN, p_plan_B_SYN, 0.004)
