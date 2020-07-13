import trimesh as trm
import numpy as np
import glob
import os.path as osp
import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataRoot', default='../../Data/Shapes/' )
parser.add_argument('--mode', default='test', help='whether to run on the training or testing set')
parser.add_argument('--shapeStart', type=int, default=0, help='the start id of the shape' )
parser.add_argument('--shapeEnd', type=int, default=3000, help='the end id of the shape' )
parser.add_argument('--isForceOutput', action='store_true', help='if we should overwrite previous point cloud')
opt = parser.parse_args()
print(opt )

dataRoot = osp.join(opt.dataRoot, opt.mode )
nameGt = 'poissonSubd.ply'

nSample = 20000
print(osp.join(dataRoot, 'Shape__*' ) )
shapeList = glob.glob(osp.join(dataRoot, 'Shape__*' ) )
print(len(shapeList ) )
shapeList = sorted(shapeList )[opt.shapeStart : min(opt.shapeEnd, len(shapeList) ) ]

for shape in shapeList:
    print('Processing... ', shape )
    pointsGtName = osp.join( shape, nameGt.replace('.ply', '_UniformPts.npy') )
    normalsGtName = osp.join( shape, nameGt.replace('.ply', '_UniformPtsNormals.npy') )

    if osp.isfile(pointsGtName ) or osp.isfile(normalsGtName ):
        print('Warning: some of the generated files already exists')
        if opt.isForceOutput:
            print("Will be overwritten.")
        else:
            print("Will be skipped.")
            continue

    meshGt = trm.load(osp.join(shape, nameGt ) )
    meshGt_faceNormals = meshGt.face_normals
    pointsGt, facesIdGt = trm.sample.sample_surface_even(meshGt, count=nSample)
    normalsGt = meshGt_faceNormals[facesIdGt, :]
    np.save(pointsGtName, pointsGt.astype(np.float32 ) )
    np.save(normalsGtName, normalsGt.astype(np.float32 ) )

