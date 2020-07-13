import torch
import numpy as np
from torch.autograd import Variable
import torch.optim as optim
import argparse
import random
import os
import models
import torchvision.utils as vutils
import utils
import dataLoader
import torch.nn as nn
from torch.utils.data import DataLoader
import math
import cv2
import torch.nn.functional as F
import os.path as osp
from model.pointnet import PointNetRefinePoint
from chamfer_distance import ChamferDistance
import open3d as o3d
import trimesh as trm

parser = argparse.ArgumentParser()
# The locationi of training set
parser.add_argument('--dataRoot', default='../../Data/Images%d/test/', help='path to images' )
parser.add_argument('--shapeRoot', default='../../Data/Shapes/test/', help='path to images' )
parser.add_argument('--experiment', default=None, help='the path to store samples and models' )
parser.add_argument('--testRoot', default=None, help='the path to output the code')
# The basic training setting
parser.add_argument('--nepoch', type=int, default=10, help='the number of epochs for training' )
parser.add_argument('--batchSize', type=int, default=10, help='input batch size' )
parser.add_argument('--imageHeight', type=int, default=192, help='the height / width of the input image to network' )
parser.add_argument('--imageWidth', type=int, default=256, help='the height / width of the input image to network' )
parser.add_argument('--envHeight', type=int, default=256, help='the height / width of the input envmap to network' )
parser.add_argument('--envWidth', type=int, default=512, help='the height / width of the input envmap to network' )
# The parameters
parser.add_argument('--camNum', type=int, default=10, help='the number of views to create the visual hull' )
parser.add_argument('--sampleNum', type=int, default=1, help='the sample num for the cost volume' )
parser.add_argument('--shapeStart', type=int, default=0, help='the start id of the shape' )
parser.add_argument('--shapeEnd', type=int, default=3000, help='the end id of the shape' )
# The rendering parameters
parser.add_argument('--eta1', type=float, default=1.0003, help='the index of refraction of air' )
parser.add_argument('--eta2', type=float, default=1.4723, help='the index of refraction of glass' )
parser.add_argument('--fov', type=float, default=63.4, help='the field of view of camera' )
# Weight of Loss
parser.add_argument('--normalWeight', type=float, default=5.0, help='the weight of normal' )
parser.add_argument('--pointWeight', type=float, default=200.0, help='the weight of point' )
# The gpu setting
parser.add_argument('--cuda', action='store_true', help='enables cuda' )
parser.add_argument('--deviceIds', type=int, nargs='+', default=[0], help='the gpus used for training network' )
# The view selection mode
parser.add_argument('--viewMode', type=int, default=0, help='the view selection Mode: 0-renderError, 1-nearest, 2-average')
# The loss function
parser.add_argument('--lossMode', type=int, default=2, help='the loss function: 0-view, 1-nearest, 3-chamfer')
# Output the baseline with mapping only
parser.add_argument('--isBaseLine', action='store_true', help='whether to output the baseline with only the normal mapping')
# whether to use rendering error
parser.add_argument('--isNoRenderError', action='store_true', help='whether to use rendering error or not')

opt = parser.parse_args()
print(opt )

opt.dataRoot = opt.dataRoot % opt.camNum
opt.gpuId = opt.deviceIds[0]
nw = opt.normalWeight
pw = opt.pointWeight

if opt.experiment is None:
    opt.experiment = "check%d_point_nw%.2f_pw%.2f" % (opt.camNum, nw, pw)

opt.experiment += '_view_'
if opt.viewMode == 0:
    opt.experiment += 'renderError'
elif opt.viewMode == 1:
    opt.experiment += 'nearest'
elif opt.viewMode == 2:
    opt.experiment += 'average'
else:
    print('Wrong: unrecognizable view selection mode')
    assert(False )

if opt.isNoRenderError:
    opt.experiment += '_norendering'

opt.experiment += '_loss_'
if opt.lossMode == 0:
    opt.experiment += 'view'
elif opt.lossMode == 1:
    opt.experiment += 'nearest'
elif opt.lossMode == 2:
    opt.experiment += 'chamfer'
else:
    print('Wrong: unrecognizable loss function mode')
    assert(False )

opt.testRoot = opt.experiment.replace('check', 'testMesh')
if opt.isBaseLine:
    opt.testRoot += '_baseLine'

os.system('mkdir {0}'.format( opt.testRoot ) )
os.system('cp *.py %s' % opt.testRoot )

opt.seed = 0
print("Random Seed: ", opt.seed )
random.seed(opt.seed )
torch.manual_seed(opt.seed )

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda" )

# Other modules
brdfDataset = dataLoader.BatchLoader(
        opt.dataRoot, shapeRoot = opt.shapeRoot,
        imHeight = opt.imageHeight, imWidth = opt.imageWidth,
        envHeight = opt.envHeight, envWidth = opt.envWidth,
        isRandom = True, phase='TEST', rseed = 1,
        isLoadOptim = False, isLoadEnvmap = False,
        isLoadCam = False, isLoadVH = False, isLoadPoints = True,
        shapeRs = opt.shapeStart, shapeRe = opt.shapeEnd,
        camNum = opt.camNum )
brdfLoader = DataLoader( brdfDataset, batch_size = 1, num_workers = 16, shuffle = True )

chamferDist = ChamferDistance()
if opt.cuda:
    chamferDist = chamferDist.cuda()

j = 0
pointErrsNpList = np.ones([1, 2], dtype=np.float32 )
metroErrsNpList = np.ones([1, 2], dtype=np.float32 )
normalErrsNpList = np.ones([1, 2], dtype=np.float32 )
meanAngleErrsNpList = np.ones([1, 2], dtype=np.float32 )
medianAngleErrsNpList = np.ones([1, 2], dtype=np.float32 )

epoch = opt.nepoch
validCount = 0
testingLog = open('{0}/testingLog_{1}.txt'.format(opt.testRoot, epoch ), 'w' )
for i, dataBatch in enumerate(brdfLoader ):
    j += 1

    # Load visual hull point
    pointVH_cpu = dataBatch['pointVH'][0]
    pointVHBatch = Variable(pointVH_cpu ).cuda()
    normalPointVH_cpu = dataBatch['normalPointVH'][0]
    normalPointVHBatch = Variable(normalPointVH_cpu ).cuda()

    # Load ground-truth points
    pointGt_cpu = dataBatch['pointGt'][0]
    pointGtBatch = Variable(pointGt_cpu ).cuda()
    normalPointGt_cpu = dataBatch['normalPointGt'][0]
    normalPointGtBatch = Variable(normalPointGt_cpu ).cuda()

    name = dataBatch['name'][0][0]
    shapeRoot = '/'.join(name.split('/')[0:-1] )

    gtNormal = normalPointGtBatch
    gtPoint = pointGtBatch

    # Predict New Points and New Normal
    ########################################
    pointPreds = []
    normalPreds = []
    pointPreds.append(pointVHBatch )
    normalPreds.append(normalPointVHBatch )

    meshName = osp.join(shapeRoot, 'reconstruct_%d_view_' % opt.camNum )
    if opt.viewMode == 0:
        meshName += 'renderError'
    elif opt.viewMode == 1:
        meshName += 'nearest'
    elif opt.viewMode == 2:
        meshName += 'average'

    if opt.isNoRenderError:
        meshName += '_norendering'

    if opt.isBaseLine:
        meshName += '_baseLine.ply'
    else:
        meshName += '_loss_'
        if opt.lossMode == 0:
            meshName += 'view'
        elif opt.lossMode == 1:
            meshName += 'nearest'
        elif opt.lossMode == 2:
            meshName += 'chamfer'
        meshName += '.ply'

    if not osp.isfile(meshName ):
        print("Warning: mesh %s does not exist.")
        j -= 1
        continue
    else:
        validCount += 1
        mesh = trm.load(meshName )
        mesh_faceNormals = mesh.face_normals
        points, facesId = trm.sample.sample_surface_even(mesh, count=20000)
        normals = mesh_faceNormals[facesId, :]
        pointPredBatch = Variable(torch.from_numpy(points.astype(np.float32 ) ) ).cuda()
        normalPredBatch = Variable(torch.from_numpy(normals.astype(np.float32 ) ) ).cuda()
        pointPreds.append(pointPredBatch )
        normalPreds.append(normalPredBatch )

    pointErrs = []
    metroErrs = []
    normalErrs = []
    meanAngleErrs = []
    medianAngleErrs = []

    assert(len(pointPreds) == len(normalPreds ) )
    for m in range(0, len(pointPreds) ):
        dist1, id1, dist2, id2 = chamferDist(gtPoint.unsqueeze(0),
                pointPreds[m].unsqueeze(0) )
        pointErrs.append(0.5 * (torch.mean(dist1) + torch.mean(dist2) )  )
        metroErrs.append(max(torch.max(dist1 ), torch.max(dist2 ) ) )

        id1, id2 = id1.long().squeeze(0).detach(), id2.long().squeeze(0).detach()

        gtToPredNormal = torch.index_select(normalPreds[m], dim=0, index = id1 )
        predToGtNormal = torch.index_select(gtNormal, dim=0, index=id2 )

        normalErr_gtToPred = torch.mean(torch.pow(gtToPredNormal - gtNormal, 2) )
        normalErr_predToGt = torch.mean(torch.pow(predToGtNormal - normalPreds[m], 2) )
        normalErrs.append(0.5 * (normalErr_gtToPred + normalErr_predToGt ) )

        meanAngle_gtToPred = 180.0/np.pi * torch.mean(torch.acos(torch.clamp(
            torch.sum(gtToPredNormal * gtNormal, dim=1 ), -1, 1) ) )
        meanAngle_predToGt = 180.0/np.pi * torch.mean(torch.acos(torch.clamp(
            torch.sum(normalPreds[m] * predToGtNormal, dim=1 ), -1, 1) ) )
        meanAngleErrs.append(0.5 * (meanAngle_gtToPred + meanAngle_predToGt ) )

        medianAngle_gtToPred = 180.0/np.pi * torch.median(torch.acos(torch.clamp(
            torch.sum( gtToPredNormal * gtNormal, dim=1), -1, 1) ) )
        medianAngle_predToGt = 180.0/np.pi * torch.median(torch.acos(torch.clamp(
            torch.sum( normalPreds[m] * predToGtNormal, dim=1), -1, 1) ) )
        medianAngleErrs.append(0.5 * (medianAngle_gtToPred + medianAngle_predToGt ) )

    utils.writeErrToScreen('point', pointErrs, epoch, j )
    utils.writeErrToScreen('metro', metroErrs, epoch, j )
    utils.writeErrToScreen('normal', normalErrs, epoch, j )
    utils.writeErrToScreen('meanAngle', meanAngleErrs, epoch, j )
    utils.writeErrToScreen('medianAngle', medianAngleErrs, epoch, j )

    utils.writeErrToFile('point', pointErrs, testingLog, epoch, j )
    utils.writeErrToFile('metro', metroErrs, testingLog, epoch, j )
    utils.writeErrToFile('normal', normalErrs, testingLog, epoch, j )
    utils.writeErrToFile('meanAngle', meanAngleErrs, testingLog, epoch, j )
    utils.writeErrToFile('medianAngle', medianAngleErrs, testingLog, epoch, j )

    pointErrsNpList = np.concatenate([pointErrsNpList, utils.turnErrorIntoNumpy(pointErrs ) ], axis=0 )
    metroErrsNpList = np.concatenate([metroErrsNpList, utils.turnErrorIntoNumpy(metroErrs ) ], axis=0 )
    normalErrsNpList = np.concatenate([normalErrsNpList, utils.turnErrorIntoNumpy(normalErrs ) ], axis=0 )
    meanAngleErrsNpList = np.concatenate([meanAngleErrsNpList, utils.turnErrorIntoNumpy(meanAngleErrs ) ], axis=0 )
    medianAngleErrsNpList = np.concatenate([medianAngleErrsNpList, utils.turnErrorIntoNumpy(medianAngleErrs ) ], axis=0 )

    utils.writeNpErrToScreen('pointAccu', np.mean(pointErrsNpList[1:j+1, :], axis=0), epoch, j )
    utils.writeNpErrToScreen('metroAccu', np.mean(metroErrsNpList[1:j+1, :], axis=0), epoch, j )
    utils.writeNpErrToScreen('normalAccu', np.mean(normalErrsNpList[1:j+1, :], axis=0), epoch, j )
    utils.writeNpErrToScreen('meanAngleAccu', np.mean(meanAngleErrsNpList[1:j+1, :], axis=0), epoch, j )
    utils.writeNpErrToScreen('medianAngleAccu', np.mean(medianAngleErrsNpList[1:j+1, :], axis=0), epoch, j )

    utils.writeNpErrToFile('pointAccu', np.mean(pointErrsNpList[1:j+1, :], axis=0), testingLog, epoch, j )
    utils.writeNpErrToFile('metroAccu', np.mean(metroErrsNpList[1:j+1, :], axis=0), testingLog, epoch, j )
    utils.writeNpErrToFile('normalAccu', np.mean(normalErrsNpList[1:j+1, :], axis=0), testingLog, epoch, j )
    utils.writeNpErrToFile('meanAngleAccu', np.mean(meanAngleErrsNpList[1:j+1, :], axis=0), testingLog, epoch, j )
    utils.writeNpErrToFile('medianAngleAccu', np.mean(medianAngleErrsNpList[1:j+1, :], axis=0), testingLog, epoch, j )



testingLog.close()

# Save the error record
np.save('{0}/pointError_{1}.npy'.format(opt.testRoot, epoch), pointErrsNpList )
np.save('{0}/metroError_{1}.npy'.format(opt.testRoot, epoch), metroErrsNpList )
np.save('{0}/normalError_{1}.npy'.format(opt.testRoot, epoch), normalErrsNpList )
np.save('{0}/meanAngleError_{1}.npy'.format(opt.testRoot, epoch), meanAngleErrsNpList)
np.save('{0}/medianAngleError_{1}.npy'.format(opt.testRoot, epoch), medianAngleErrsNpList )
print('Valid Mesh Num: %s' % validCount )
