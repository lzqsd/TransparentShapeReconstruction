from __future__ import print_function
import numpy as np
from PIL import Image
import cv2
import os.path as osp
import torch
from torch.autograd import Variable
import h5py
import trimesh as trm
import open3d as o3d
from skimage import measure

def writeErrToScreen(errorName, errorArr, epoch, j):
    print( ('[%d/%d] {0}:' % (epoch, j) ).format(errorName), end=' ')
    for n in range(0, len(errorArr) ):
        print('%.6f' % errorArr[n].data.item(), end = ' ')
    print('.')

def writeCoefToScreen(coefName, coef, epoch, j):
    print( ('[%d/%d] {0}:' % (epoch, j) ).format(coefName), end=' ')
    coefNp = coef.cpu().data.numpy()
    for n in range(0, len(coefNp) ):
        print('%.6f' % coefNp[n], end = ' ')
    print('.')

def writeNpErrToScreen(errorName, errorArr, epoch, j):
    print( ('[%d/%d] {0}:' % (epoch, j) ).format(errorName), end=' ')
    for n in range(0, len(errorArr) ):
        print('%.6f' % errorArr[n], end = ' ')
    print('.')

def writeErrToFile(errorName, errorArr, fileOut, epoch, j):
    fileOut.write( ('[%d/%d] {0}:'% (epoch, j) ).format(errorName) )
    for n in range(0, len(errorArr) ):
        fileOut.write('%.6f ' % errorArr[n].data.item() )
    fileOut.write('.\n')

def writeCoefToFile(coefName, coef, fileOut, epoch, j):
    fileOut.write( ('[%d/%d] {0}: ' % (epoch, j) ).format(coefName) )
    coefNp = coef.cpu().data.numpy()
    for n in range(0, len(coefNp) ):
        fileOut.write('%.6f ' % coefNp[n] )
    fileOut.write('.\n')

def writeNpErrToFile(errorName, errorArr, fileOut, epoch, j):
    fileOut.write( ('[%d/%d] {0}:' % (epoch, j) ).format(errorName) )
    for n in range(0, len(errorArr) ):
        fileOut.write('%.6f ' % errorArr[n] )
    fileOut.write('.\n')

def turnErrorIntoNumpy(errorArr):
    errorNp = []
    for n in range(0, len(errorArr) ):
        errorNp.append(errorArr[n].data.item() )
    return np.array(errorNp)[np.newaxis, :]

def visualizeGtEnvmap(dst, names, nrows, ncols, gridRows, gridCols, edge=5):
    imgNum = len(names)
    assert(gridRows * gridCols >= imgNum)

    nRows = gridRows * nrows + edge * (gridRows + 1)
    nCols = gridCols * ncols + edge * (gridCols + 1)
    imArr = np.zeros([nRows, nCols, 3], dtype = np.float32)
    for rId in range(0, gridRows):
        for cId in range(0, gridCols):
            if rId * gridCols + cId >= imgNum:
                break
            n = rId * gridCols + cId
            sr = edge * (rId+1) + rId * nrows
            sc = edge * (cId+1) + cId * ncols

            name = names[n]
            root = '/'.join(name.split('/')[0:-1] )
            fileName = name.split('/')[-1]
            envFile = osp.join(root, fileName.split('_')[1] + '.txt')
            with open(envFile, 'r') as f:
                envName = f.readlines()[0]
                envName = envName.strip()
            im = cv2.imread(envName, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)[:, :, ::-1]
            im = cv2.resize(im, (ncols, nrows), interpolation=cv2.INTER_AREA)
            imArr[sr : sr + nrows, sc : sc + ncols, :] = im
    imArr = np.clip(imArr, 0, 1)
    imArr = (255 * imArr).astype(np.uint8)
    imArr = Image.fromarray(imArr)
    imArr.save(dst)

def computeConfMap(imBatch, segBatch, coef, gpuId):
    im = 0.5 * (imBatch + 1)

    coef0, coef1 = torch.split(coef, 1)
    coef0 = coef0.view(1, 1, 1, 1)
    coef1 = coef1.view(1, 1, 1, 1)
    minIm, _ = torch.min(im, dim=1)
    w0 = (1 - torch.exp( -(1-minIm) / 0.02) ).unsqueeze(1)
    w1 = Variable(0 * torch.FloatTensor(segBatch.size() ).cuda(gpuId) ) + 1
    return coef0 * w0 + coef1 * w1


def writeImageToFile(imgBatch, nameBatch, isGama = False):
    batchSize = imgBatch.size(0)
    for n in range(0, batchSize):
        img = imgBatch[n, :, :, :].data.cpu().numpy()
        img = np.clip(img, 0, 1)
        if isGama:
            img = np.power(img, 1.0/2.2)
        img = (255 *img.transpose([1, 2, 0] ) ).astype(np.uint8)
        if img.shape[2] == 1:
            img = np.concatenate([img, img, img], axis=2)
        img = Image.fromarray(img )
        img.save(nameBatch[n] )


def writeImageBlockToFile(imgBatch, nameBatch, isGama = False ):
    batchSize = imgBatch.size(0 )
    chNum = imgBatch.size(3 )
    for n in range(0, batchSize ):
        img = imgBatch[n, :].data.cpu().numpy()
        name = nameBatch[n]

        bheight, bwidth = img.shape[0], img.shape[1]
        imHeight, imWidth = img.shape[3], img.shape[4]

        img = img.transpose([0, 3, 1, 4, 2])
        img = img.reshape([bheight * imHeight, bwidth * imWidth, chNum ] )
        img = (255 * np.clip(np.clip(img, 0, 1), 0, 1) ).astype(np.uint8 )
        if chNum == 3:
            cv2.imwrite(nameBatch[n], img[:, :, ::-1] )
        else:
            cv2.imwrite(nameBatch[n], img.squeeze() )


def writeDataToFile(imgBatch, nameBatch ):
    batchSize = imgBatch.size(0)
    for n in range(0, batchSize):
        img = imgBatch[n, :].data.cpu().numpy()
        name = nameBatch[n]

        hf = h5py.File(name, 'w')
        hf.create_dataset('data', data=img, compression='lzf' )
        hf.close()

def writeDepthToFile(depthBatch, nameBatch):
    batchSize = depthBatch.size(0)
    for n in range(0, batchSize):
        depth = depthBatch[n, :, :, :].data.cpu().numpy().squeeze()
        np.save(nameBatch[n], depth)

def writeEnvToFile(SHBatch, nameBatch):
    batchSize = SHBatch.size(0)
    for n in range(0, batchSize):
        SH = SHBatch[n, :, :].data.cpu().numpy()
        np.save(nameBatch[n], SH)

def writeAlbedoNameToFile(fileName, albedoNameBatch):
    with open(fileName, 'w') as fileOut:
        for n in range(0, len(albedoNameBatch) ):
            fileOut.write('%s\n' % albedoNameBatch[n] )

def normalErrToColorMap(normalErr, mask, colormap):
    errMap = torch.acos(torch.clamp((-1) * normalErr, -1, 1)) * mask
    errMap = torch.clamp(errMap * 180 / np.pi, 0, colormap.shape[0]-1).type(torch.long)
    errMap = colormap[errMap, :].squeeze(1).permute(0, 3, 1, 2).type(torch.float) * mask
    return errMap

def errToColorMap(err, mask, colormap, scale = 1, maxValue = 1 ):
    errMap = torch.clamp(err * scale, 0, maxValue ) / maxValue
    errMap = (errMap * colormap.shape[0]-1).type(torch.long)
    errMap = colormap[errMap, :].squeeze(1).permute(0, 3, 1, 2).type(torch.float) * mask
    return errMap


def writePointCloud(fileName, visualHull, colormap ):
    colormap = colormap.cpu().data.numpy()
    visualHull = visualHull.cpu().data.numpy().squeeze()
    resolution = visualHull.shape[0]
    y, x, z = np.meshgrid(np.linspace(-1.1, 1.1, resolution),
            np.linspace(-1.1, 1.1, resolution),
            np.linspace(-1.1, 1.1, resolution) )
    x = x[:, :, :, np.newaxis ]
    y = y[:, :, :, np.newaxis ]
    z = z[:, :, :, np.newaxis ]
    coord = np.concatenate([x, y, z], axis=3 ).astype(np.float32 )
    visualHull = visualHull[:, :, :, np.newaxis]
    vInd = visualHull > 0
    error = visualHull[vInd ].reshape(-1 )
    error = -np.log(error )
    points = coord[np.concatenate([vInd, vInd, vInd ], axis=3)].reshape(-1, 3)
    error = np.clip(error / np.mean(error ) * 0.5, 0, 1) * (colormap.shape[0] - 1)
    colors = colormap[error.astype(np.uint32), :].reshape(-1, 3)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points )
    pcd.colors = o3d.utility.Vector3dVector(colors )
    o3d.io.write_point_cloud(fileName, pcd)

    return

def writeVisualHull(fileName, visualHull ):
    visualHull = visualHull.cpu().data.numpy().squeeze()
    VH = visualHull.copy()
    VH[visualHull == 0] = 1
    VH[visualHull > 0 ] = -1
    verts, faces, normals, _ = measure.marching_cubes_lewiner(VH, 0)

    resolution = visualHull.shape[0]
    axisLen = float(resolution - 1) / 2.0
    verts = (verts - axisLen) / axisLen * 1.1
    mesh = trm.Trimesh(vertices = verts, vertex_normals = normals, faces = faces )
    mesh.export(fileName )
    return

def writePointWithComputedNormal(fileName, points, normals):
    points = points.data.cpu().numpy()
    normals = normals.data.cpu().numpy()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points )
    pcd.normals = o3d.utility.Vector3dVector(normals )
    o3d.geometry.PointCloud.estimate_normals(pcd, search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30 ) )
    pcd.colors = o3d.utility.Vector3dVector(np.ones(points.shape)*0.7 )
    o3d.io.write_point_cloud(fileName, pcd )


def writePointWithPredictedNormal(fileName, points, normals):
    points = points.data.cpu().numpy()
    normals = normals.data.cpu().numpy()

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points )
    pcd.normals = o3d.utility.Vector3dVector(normals )
    pcd.colors = o3d.utility.Vector3dVector(0.5 * (normals + 1) )
    o3d.io.write_point_cloud(fileName, pcd )

def writePointWithPredictedNormalSeparate(fileName, points, normals, viewIds, camNum = 10):
    points = points.data.cpu().numpy()
    normals = normals.data.cpu().numpy()
    viewIds = viewIds.data.cpu().numpy()
    for n in range(0, camNum + 1):
        fileName_n = fileName.replace('.ply', '_%d.ply' % n )
        points_n = points[viewIds == n, :]
        normals_n = normals[viewIds == n, :]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_n )
        pcd.normals = o3d.utility.Vector3dVector(normals_n )
        pcd.colors = o3d.utility.Vector3dVector(0.5 * (normals_n + 1) )
        o3d.io.write_point_cloud(fileName_n, pcd )

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points )
    pcd.normals = o3d.utility.Vector3dVector(normals )
    pcd.colors = o3d.utility.Vector3dVector(0.5 * (normals + 1) )
    o3d.io.write_point_cloud(fileName, pcd )
