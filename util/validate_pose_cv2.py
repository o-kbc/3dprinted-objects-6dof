import os
import cv2
import numpy as np

from vispy import gloo
from plyfile import PlyData
from scipy.spatial.distance import pdist


F = os.path.dirname(os.path.abspath(__file__))


def draw_triangle(image, vertices, color=(0, 255, 0)):
    image_copy = image.copy()
    points = vertices.reshape((-1,1,2))
    cv2.polylines(image_copy, [points], True, color)
    return image_copy


def draw_mesh(image, vertices2d, indices):
    img = image.copy()
    for index in indices:
        triangle = np.array([vertices2d[index[0]], vertices2d[index[1]], vertices2d[index[2]]], dtype='int32')
        img = draw_triangle(img, triangle)
    return img


def draw_axys(image, rvec, tvec, intrisics, distortions):
    img = image.copy()
    axis = np.array([(0, 0, .03), (.05, 0, .03), (0, .05, .03), (0, 0, .08)])
    points2d, _ = cv2.projectPoints(axis, rvec, tvec, intrisics, distortions)
    points2d = [(int(x), int(y)) for x, y in (p[0] for p in points2d)]
    cv2.line(img, points2d[0], points2d[1], (255, 0, 0), 5)
    cv2.line(img, points2d[0], points2d[2], (0, 255, 0), 5)
    cv2.line(img, points2d[0], points2d[3], (0, 0, 255), 5)
    return img


def draw_corners(image, points, color=(0, 255, 0), front= (255, 255, 0), 
                 back= (0, 255, 255), size=5, text=False, use_same_color=False):
    image_cp = image.copy()
    
    p = []

    if use_same_color:
        front, back = color, color

    for i, point in enumerate(points):
        if len(point) == 1:
            x, y = int(point[0][0]), int(point[0][1])
            if i != 0 or len(points) == 8:
                p.append([x,y])
        else:
            x, y = int(point[0]), int(point[1])
            if i != 0 or len(points) == 8:
                p.append([x,y])

        if i == 0:
            current_color = color
        elif i < 5:
            current_color = back
        else:
            current_color = front
        cv2.circle(image_cp, (x, y), size, current_color, -1)
        if text:
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(image_cp , str(i), (x, y - 5), font, 1, color, 3, cv2.LINE_AA)

    cv2.line(image_cp, (p[0][0], p[0][1]), (p[1][0], p[1][1]), back, 4)
    cv2.line(image_cp, (p[0][0], p[0][1]), (p[2][0], p[2][1]), back, 4)
    cv2.line(image_cp, (p[3][0], p[3][1]), (p[1][0], p[1][1]), back, 4)
    cv2.line(image_cp, (p[3][0], p[3][1]), (p[2][0], p[2][1]), back, 4)

    cv2.line(image_cp, (p[0][0], p[0][1]), (p[4][0], p[4][1]), color, 4)
    cv2.line(image_cp, (p[5][0], p[5][1]), (p[1][0], p[1][1]), color, 4)
    cv2.line(image_cp, (p[6][0], p[6][1]), (p[2][0], p[2][1]), color, 4)
    cv2.line(image_cp, (p[7][0], p[7][1]), (p[3][0], p[3][1]), color, 4)

    cv2.line(image_cp, (p[5][0], p[5][1]), (p[4][0], p[4][1]), front, 4)
    cv2.line(image_cp, (p[6][0], p[6][1]), (p[4][0], p[4][1]), front, 4)
    cv2.line(image_cp, (p[7][0], p[7][1]), (p[5][0], p[5][1]), front, 4)
    cv2.line(image_cp, (p[7][0], p[7][1]), (p[6][0], p[6][1]), front, 4)

    return image_cp


class Model3D(object):
	'''
	Title: SSD-6D: Making RGB-Based 3D Detection and 6D Pose Estimation Great Again
	Author: Wadim Kehl; Fabian Manhardt;
	Availability: https://github.com/SergioRAgostinho/ssd6d/blob/master/rendering/model.py
	'''
    def __init__(self, file_to_load=None):
        self.vertices = None
        self.centroid = None
        self.indices = None
        self.colors = None
        self.texcoord = None
        self.texture = None
        self.collated = None
        self.vertex_buffer = None
        self.index_buffer = None
        self.bb = None
        self.bb_vbuffer = None
        self.bb_ibuffer = None
        self.diameter = None

        if file_to_load:
            self.load(file_to_load)

        self._compute_bbox()


    def _compute_bbox(self):
        self.bb = []

        minx, maxx = min(self.vertices[:, 0]), max(self.vertices[:, 0])
        miny, maxy = min(self.vertices[:, 1]), max(self.vertices[:, 1])
        minz, maxz = min(self.vertices[:, 2]), max(self.vertices[:, 2])

        self.bb.append([minx, miny, minz])
        self.bb.append([minx, miny, maxz])
        self.bb.append([minx, maxy, minz])
        self.bb.append([minx, maxy, maxz])
        self.bb.append([maxx, miny, minz])
        self.bb.append([maxx, miny, maxz])
        self.bb.append([maxx, maxy, minz])
        self.bb.append([maxx, maxy, maxz])
        self.bb = np.asarray(self.bb, dtype=np.float32)

        self.diameter = max(pdist(self.bb, 'euclidean'))

        colors  = [[1, 0, 0],[1, 1, 0], [0, 1, 0], [0, 1, 1], [0, 0, 1], [0, 1, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]]
        indices = [0, 1, 0, 2, 3, 1, 3, 2, 4, 5, 4, 6, 7, 5, 7, 6, 0, 4, 1, 5, 2, 6, 3, 7]

        vertices_type = [('a_position', np.float32, 3), ('a_color', np.float32, 3)]
        collated = np.asarray(list(zip(self.bb, colors)), vertices_type)

        self.bb_vbuffer = gloo.VertexBuffer(collated)
        self.bb_ibuffer = gloo.IndexBuffer(indices)


    def load(self, path, demean=False, scale=1.0):
        
        data = PlyData.read(path)
        self.vertices = np.zeros((data['vertex'].count, 3))
        self.vertices[:, 0] = np.array(data['vertex']['x'])
        self.vertices[:, 1] = np.array(data['vertex']['y'])
        self.vertices[:, 2] = np.array(data['vertex']['z'])
        self.vertices *= scale
        self.centroid = np.mean(self.vertices, 0)

        if demean:
            self.centroid = np.zeros((1, 3), np.float32)
            self.vertices -= self.centroid

        self._compute_bbox()

        self.indices = np.asarray(list(data['face']['vertex_indices']), np.uint32)

        filename = path.split('/')[-1]
        abs_path = path[:path.find(filename)]
        tex_to_load = None
        if os.path.exists(abs_path + filename[:-4] + '.jpg'):
            tex_to_load = abs_path + filename[:-4] + '.jpg'
        elif os.path.exists(abs_path + filename[:-4] + '.png'):
            tex_to_load = abs_path + filename[:-4] + '.png'

        if tex_to_load is not None:
            image = cv2.flip(cv2.imread(tex_to_load, cv2.IMREAD_UNCHANGED), 0)
            self.texture = gloo.Texture2D(image)

            if 'texcoord' in str(data):
                self.texcoord = np.asarray(list(data['face']['texcoord']))
                assert self.indices.shape[0] == self.texcoord.shape[0]
                temp = np.zeros((data['vertex'].count, 2))
                temp[self.indices.flatten()] = self.texcoord.reshape((-1, 2))
                self.texcoord = temp

            elif 'texture_u' in str(data):
                self.texcoord = np.zeros((data['vertex'].count, 2))
                self.texcoord[:, 0] = np.array(data['vertex']['texture_u'])
                self.texcoord[:, 1] = np.array(data['vertex']['texture_v'])

        if self.texcoord is not None:
            vertices_type = [('a_position', np.float32, 3), ('a_texcoord', np.float32, 2)]
            self.collated = np.asarray(list(zip(self.vertices, self.texcoord)), vertices_type)

        else:
            self.colors = 0.5*np.ones((data['vertex'].count, 3))
            if 'blue' in str(data):
                self.colors[:, 0] = np.array(data['vertex']['blue'])
                self.colors[:, 1] = np.array(data['vertex']['green'])
                self.colors[:, 2] = np.array(data['vertex']['red'])
                self.colors /= 255.0
            else:
                pass
            vertices_type = [('a_position', np.float32, 3), ('a_color', np.float32, 3)]
            self.collated = np.asarray(list(zip(self.vertices, self.colors)), vertices_type)

        self.vertex_buffer = gloo.VertexBuffer(self.collated)
        self.index_buffer = gloo.IndexBuffer(self.indices.flatten())
        
        
def project_mesh(cadModel, bounds, cameraMatrix, frame, coefficients=np.zeros((1,5)), pose=None):
    
    cad = Model3D(cadModel)
    
    print('CAD bounding box: {}'.format(cad.bb))
    print('CAD diameter: {}'.format(cad.diameter))
    print('Projected corners: {}'.format(bounds))
    print('Camera Matrix: {}'.format(cameraMatrix))
    print('Distortions Coefficients: {}'.format(coefficients))
    
    _, rvec, tvec = cv2.solvePnP(cad.bb, bounds, cameraMatrix, coefficients)
    
    print('Computed Pose: r:{} t:{}'.format(rvec.flatten(), tvec.flatten()))
    axys_image1 = draw_axys(frame, rvec, tvec, cameraMatrix, coefficients)
    computedVertices1, _ = cv2.projectPoints(cad.vertices, rvec, tvec, cameraMatrix, coefficients)
    printed1 = draw_mesh(axys_image1, computedVertices1, cad.indices)
    cv2.imshow('Projected Mesh CV2', printed1)
    
    printed2 = draw_corners(frame, bounds)
    cv2.imshow('Bounds', printed2)
    
    if pose is not None:
        print('Annotated Pose: r:{} t:{}'.format(pose[0].flatten(), pose[1].flatten()))
        axys_image0 = draw_axys(frame, pose[0], pose[1], cameraMatrix, coefficients)
        computedVertices0, _ = cv2.projectPoints(cad.vertices, pose[0], pose[1], cameraMatrix, coefficients)
        printed0 = draw_mesh(axys_image0, computedVertices0, cad.indices)
        cv2.imshow('Annotated Values on Markerfield', printed0)
    
    k = cv2.waitKey(0)
    if k == 27:
        cv2.destroyAllWindows()
        exit(1)
    


if __name__ == '__main__':
    
    
    CADPATH = F + '\\..\\CAD\\car01\\car01.ply'
    DATASET = F + '\\..\\data\\car01\\APPLE_IPHONE_X\\SIMPLE'
    
    CAMERAVALUES = F + '\\..\\cameras\\APPLE_IPHONE_X\\camera_matrix.txt'
    COEFFICIENTS = F + '\\..\\cameras\\APPLE_IPHONE_X\\coefficients.txt'
    
    IMGFOLDER     = 'frames'
    BMASKFOLDER   = 'binary_mask'
    POSESFOLDER   = 'poses'
    CORNERSFOLDER = 'corners'
    
    images  = os.listdir(os.path.join(DATASET, IMGFOLDER))
    
    resize_image = (640, 480)
    
    for name in images:
        imagePath = os.path.join(DATASET, IMGFOLDER, name) 
        image = cv2.imread(imagePath)
        H, W, C = image.shape
        image = cv2.resize(image, resize_image)

        cfile = imagePath.replace(IMGFOLDER, CORNERSFOLDER).replace('jpg', 'txt')
        values = np.loadtxt(cfile).flatten()
        values = values[2:].reshape((8,2))
        values[:, 0] = values[:, 0] * image.shape[1]
        values[:, 1] = values[:, 1] * image.shape[0]
        #values = np.array(values, dtype=np.int32)
    
        camera = np.loadtxt(CAMERAVALUES).reshape((3,3))
        camera[0,:] = camera[0,:] * (image.shape[1]/W)
        camera[1,:] = camera[1,:] * (image.shape[0]/H)
    
        coefficients = np.loadtxt(COEFFICIENTS).reshape((1,-1))
    
        pose = np.loadtxt(imagePath.replace(IMGFOLDER, POSESFOLDER).replace('jpg', 'txt'))
    
        project_mesh(CADPATH, values, camera, image, coefficients=coefficients, pose=pose)
    