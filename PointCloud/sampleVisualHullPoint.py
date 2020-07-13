import trimesh as trm
import numpy as np
import glob
import os.path as osp
import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataRoot', default='../../Data/Shapes/' )
parser.add_argument('--mode', default='test', help='whether to run on the training or testing set' )
parser.add_argument('--rs', type=int, default=0, help='the start id of the shape' )
parser.add_argument('--re', type=int, default=10, help='the end id of the shape' )
parser.add_argument('--camNum', type=int )
parser.add_argument('--isForceOutput', action='store_true', help='if we should overwrite previous point cloud')
opt = parser.parse_args()
print(opt )

dataRoot = osp.join(opt.dataRoot, opt.mode )
nVHviews = opt.camNum
nameVH = 'visualHullSubd_{}.ply'.format(nVHviews )
nameGt = 'poissonSubd.ply'

nSample = 20000
print(osp.join(dataRoot, 'Shape__*' ) )
shapeList = glob.glob(osp.join(dataRoot, 'Shape__*' ) )
print(len(shapeList ) )
shapeList = sorted(shapeList )[opt.rs : min(opt.re, len(shapeList ) ) ]

cnt = 0
for shape in shapeList:
    cnt += 1
    print('Processing %d/%d' % (cnt, len(shapeList) ), shape )
    pointsVHName = osp.join( shape, nameVH.replace('.ply', '_pts.npy') )
    normalsVHName = osp.join( shape, nameVH.replace('.ply', '_ptsNormals.npy') )
    pointsGtName = osp.join( shape, nameGt.replace('.ply', '_%d_pts.npy' % nVHviews ) )
    normalsGtName = osp.join( shape, nameGt.replace('.ply', '_%d_ptsNormals.npy' % nVHviews ) )

    if osp.isfile(pointsVHName ) and osp.isfile(normalsVHName ) and osp.isfile(pointsGtName ) and osp.isfile(normalsGtName ):
        print('Warning: the generated files already exists')
        if opt.isForceOutput:
            print("Will be overwritten.")
        else:
            print("Will be skipped.")
            continue

    meshVH = trm.load(osp.join(shape, nameVH ) )
    meshVH_faceNormals = meshVH.face_normals
    pointsVH, facesIdVH = trm.sample.sample_surface_even(meshVH, count=nSample)
    normalsVH = meshVH_faceNormals[facesIdVH, :]
    np.save(pointsVHName, pointsVH.astype(np.float32 ) )
    np.save(normalsVHName, normalsVH.astype(np.float32 ) )

    meshGt = trm.load(osp.join(shape, nameGt ) )
    meshGt_faceNormals = meshGt.face_normals
    pointsGt, _, facesIdGt = trm.proximity.closest_point(meshGt, pointsVH )
    normalsGt = meshGt_faceNormals[facesIdGt, :]
    np.save(pointsGtName, pointsGt.astype(np.float32 ) )
    np.save(normalsGtName, normalsGt.astype(np.float32 ) )
