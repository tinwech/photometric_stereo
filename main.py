import cv2
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from ast import literal_eval as make_tuple
from sklearn.preprocessing import normalize
import scipy
import time

image_row = 120
image_col = 120

# visualizing the mask (size : "image width" * "image height")
def mask_visualization(M):
    mask = np.copy(np.reshape(M, (image_row, image_col)))
    plt.figure()
    plt.imshow(mask, cmap='gray')
    plt.title('Mask')

# visualizing the unit normal vector in RGB color space
# N is the normal map which contains the "unit normal vector" of all pixels (size : "image width" * "image height" * 3)
def normal_visualization(N):
    # converting the array shape to (w*h) * 3 , every row is a normal vetor of one pixel
    N_map = np.copy(np.reshape(N, (image_row, image_col, 3)))
    # Rescale to [0,1] float number
    N_map = (N_map + 1.0) / 2.0
    plt.figure()
    plt.imshow(N_map)
    plt.title('Normal map')

# visualizing the depth on 2D image
# D is the depth map which contains "only the z value" of all pixels (size : "image width" * "image height")
def depth_visualization(D):
    D_map = np.copy(np.reshape(D, (image_row, image_col)))
    # D = np.uint8(D)
    plt.figure()
    plt.imshow(D_map)
    plt.colorbar(label='Distance to Camera')
    plt.title('Depth map')
    plt.xlabel('X Pixel')
    plt.ylabel('Y Pixel')

# convert depth map to point cloud and save it to ply file
# Z is the depth map which contains "only the z value" of all pixels (size : "image width" * "image height")
def save_ply(Z, filepath):
    Z_map = np.reshape(Z, (image_row, image_col)).copy()
    data = np.zeros((image_row * image_col, 3), dtype=np.float32)
    # let all point float on a base plane 
    baseline_val = np.min(Z_map)
    Z_map[np.where(Z_map == 0)] = baseline_val
    for i in range(image_row):
        for j in range(image_col):
            idx = i * image_col + j
            data[idx][0] = j
            data[idx][1] = i
            data[idx][2] = Z_map[image_row - 1 - i][j]
    # output to ply file
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(data)
    o3d.io.write_point_cloud(filepath, pcd, write_ascii=True)

# show the result of saved ply file
def show_ply(filepath):
    pcd = o3d.io.read_point_cloud(filepath)
    o3d.visualization.draw_geometries([pcd])

# read the .bmp file
def read_bmp(filepath):
    global image_row
    global image_col
    image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    image_row, image_col = image.shape
    return image

def get_data(name):
    images = []
    lights = []
    with open(f'./test/{name}/LightSource.txt', 'r') as f:
        for line in f.readlines():
            fname, pos = line.split(': ')
            images.append(read_bmp(f'./test/{name}/{fname}.bmp'))
            lights.append(np.array(make_tuple(pos)))

    _, mask = cv2.threshold(images[0], 0, 255, cv2.THRESH_BINARY)
    return np.array(images), np.array(lights), mask

def get_normal(images, lights):
    # reshape images (m, w, h) to (m, w * h)
    I = np.reshape(images, (images.shape[0], -1))
    # normalize the light source vector
    L = normalize(lights, axis=1)
    # solve KdN by pseudo-inverse
    KdN = np.linalg.pinv(L) @ I
    # normalize KdN to get N
    N = normalize(KdN.T, axis=1)
    return N

def get_surface(N, mask):
    N = np.reshape(N, (image_row, image_col, 3))

    # dealing with extreme normal
    N = cv2.GaussianBlur(N, (3, 3), 1)
                
    px, py = np.where(mask != 0)
    S = len(px)
    M = scipy.sparse.lil_matrix((2 * S, S))
    V = np.zeros(2 * S)
    pix2id = np.zeros(mask.shape, dtype='int')
    for idx, (x, y) in enumerate(zip(px, py)):
        pix2id[x][y] = idx

    for idx, (x, y) in enumerate(zip(px, py)):
        nx = N[x][y][0]
        ny = N[x][y][1]
        nz = N[x][y][2]

        if mask[x][y + 1]:
            row = idx * 2
            M[row, pix2id[x][y]] = -1
            M[row, pix2id[x][y + 1]] = 1
            if nz != 0:
                V[row] = -nx / nz

        if mask[x + 1][y]:
            row = idx * 2 + 1
            M[row, pix2id[x][y]] = 1
            M[row, pix2id[x + 1][y]] = -1
            if nz != 0:
                V[row] = -ny / nz

    MtM = M.T @ M
    MtV = M.T @ V
    z = scipy.sparse.linalg.spsolve(MtM, MtV)

    depth = np.zeros((image_row, image_col))
    for idx, (x, y) in enumerate(zip(px, py)):
        depth[x][y] = z[idx]

    return depth

if __name__ == '__main__':

    for obj in ['bunny', 'star', 'venus']:
        start = time.time()
        images, lights, mask = get_data(obj)
        N = get_normal(images, lights)
        Z = get_surface(N, mask)
        print(f"{obj}, elapsed time: {time.time() - start}")

        mask_visualization(mask)
        normal_visualization(N)
        depth_visualization(Z)
        plt.show()

        save_ply(Z, f'{obj}.ply')
        show_ply(f'{obj}.ply')
