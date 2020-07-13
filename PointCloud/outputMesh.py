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

parser = argparse.ArgumentParser()
# The locationi of training set
parser.add_argument('--dataRoot', default='../../Data/Images%d/test/', help='path to images' )
parser.add_argument('--shapeRoot', default='../../Data/Shapes/test/', help='path to images' )
parser.add_argument('--experiment', default=None, help='the path to store samples and models' )
parser.add_argument('--outputRoot', default=None, help='the path to output the code')
# The basic training setting
parser.add_argument('--nepoch', type=int, default=10, help='the number of epochs for training' )
parser.add_argument('--imageHeight', type=int, default=192, help='the height / width of the input image to network' )
parser.add_argument('--imageWidth', type=int, default=256, help='the height / width of the input image to network' )
parser.add_argument('--envHeight', type=int, default=256, help='the height / width of the input envmap to network' )
parser.add_argument('--envWidth', type=int, default=512, help='the height / width of the input envmap to network' )
# The parameters
parser.add_argument('--camNum', type=int, default=10, help='the number of views to create the visual hull' )
parser.add_argument('--sampleNum', type=int, default=1, help='the sample num for the cost volume' )
parser.add_argument('--shapeStart', type=int, default=0, help='the start id of the shape' )
parser.add_argument('--shapeEnd', type=int, default=600, help='the end id of the shape' )
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

opt.batchSize = opt.camNum

opt.outputRoot = opt.experiment.replace('check', 'output')
if opt.isBaseLine:
    opt.outputRoot += '_baseLine'

os.system('mkdir {0}'.format( opt.outputRoot ) )
os.system('cp *.py %s' % opt.outputRoot )

opt.seed = 0
print("Random Seed: ", opt.seed )
random.seed(opt.seed )
torch.manual_seed(opt.seed )

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda" )

# Other modules
renderer = models.renderer(eta1 = opt.eta1, eta2 = opt.eta2,
        isCuda = opt.cuda, gpuId = opt.gpuId,
        batchSize = opt.batchSize,
        fov = opt.fov,
        imWidth=opt.imageWidth, imHeight = opt.imageHeight,
        envWidth = opt.envWidth, envHeight = opt.envHeight )

brdfDataset = dataLoader.BatchLoader(
        opt.dataRoot, shapeRoot = opt.shapeRoot,
        imHeight = opt.imageHeight, imWidth = opt.imageWidth,
        envHeight = opt.envHeight, envWidth = opt.envWidth,
        isRandom = True, phase='TEST', rseed = 1,
        isLoadOptim = True, isLoadEnvmap = True,
        isLoadCam = True, isLoadVH = True, isLoadPoints = True,
        shapeRs = opt.shapeStart, shapeRe = opt.shapeEnd,
        camNum = opt.camNum )
brdfLoader = DataLoader( brdfDataset, batch_size = 1, num_workers = 16, shuffle = True )

sampler = models.groundtruthSampler(
        camNum = opt.camNum,
        fov = opt.fov,
        imHeight = opt.imageHeight,
        imWidth = opt.imageWidth )

# Define the model and optimizer
lr_scale = 1
pointNet = PointNetRefinePoint()
pointNet.load_state_dict(torch.load('%s/pointNet_%d.pth' % (opt.experiment, opt.nepoch-1) ) )
chamferDist = ChamferDistance()
if opt.cuda:
    pointNet = pointNet.cuda()
    chamferDist = chamferDist.cuda()

j = 0
pointErrsNpList = np.ones([1, 2], dtype=np.float32 )
normalErrsNpList = np.ones([1, 2], dtype=np.float32 )
meanAngleErrsNpList = np.ones([1, 2], dtype=np.float32 )
medianAngleErrsNpList = np.ones([1, 2], dtype=np.float32 )

epoch = opt.nepoch
testingLog = open('{0}/testingLog_{1}.txt'.format(opt.outputRoot, epoch ), 'w' )
for i, dataBatch in enumerate(brdfLoader ):
    j += 1

    # Load ground-truth from cpu to gpu
    seg1_cpu = dataBatch['seg1'].squeeze(0 )
    seg1Batch = Variable((seg1_cpu ) ).cuda()

    normal1_cpu = dataBatch['normal1'].squeeze(0 )
    normal1Batch = Variable(normal1_cpu ).cuda()

    # Load the image from cpu to gpu
    im_cpu = dataBatch['im'].squeeze(0 )
    imBatch = Variable(im_cpu ).cuda()

    # Load environment map
    envmap_cpu = dataBatch['env'].squeeze(0 )
    envBatch = Variable(envmap_cpu ).cuda()

    # Load camera parameters
    origin_cpu = dataBatch['origin'].squeeze(0 )
    originBatch = Variable(origin_cpu ).cuda()

    lookat_cpu = dataBatch['lookat'].squeeze(0 )
    lookatBatch = Variable(lookat_cpu ).cuda()

    up_cpu = dataBatch['up'].squeeze(0 )
    upBatch = Variable(up_cpu ).cuda()

    # Load optimized normal
    normal1Opt_cpu = dataBatch['normalOpt' ].squeeze(0)
    normal1OptBatch = Variable(normal1Opt_cpu ).cuda()
    normal2Opt_cpu = dataBatch['normal2Opt' ].squeeze(0)
    normal2OptBatch = Variable(normal2Opt_cpu ).cuda()

    # Load depth
    depth1_cpu = dataBatch['depth1'].squeeze(0 )
    depth1Batch = Variable(depth1_cpu ).cuda()

    # Load VH depth
    depth1VH_cpu = dataBatch['depth1VH'].squeeze(0 )[:, 2:3, :, :]
    depth1VHBatch = Variable(depth1VH_cpu ).cuda()

    seg1VH_cpu = dataBatch['seg1VH'].squeeze(0 )
    seg1VHBatch = Variable(seg1VH_cpu ).cuda()

    seg1IntBatch = seg1VHBatch * seg1Batch

    # Load visual hull point
    pointVH_cpu = dataBatch['pointVH'][0]
    pointVHBatch = Variable(pointVH_cpu ).cuda()
    normalPointVH_cpu = dataBatch['normalPointVH'][0]
    normalPointVHBatch = Variable(normalPointVH_cpu ).cuda()

    # Load nearest points
    point_cpu = dataBatch['point'][0]
    pointBatch = Variable(point_cpu ).cuda()
    normalPoint_cpu = dataBatch['normalPoint'][0]
    normalPointBatch = Variable(normalPoint_cpu ).cuda()

    # Load ground-truth points
    pointGt_cpu = dataBatch['pointGt'][0]
    pointGtBatch = Variable(pointGt_cpu ).cuda()
    normalPointGt_cpu = dataBatch['normalPointGt'][0]
    normalPointGtBatch = Variable(normalPointGt_cpu ).cuda()

    name = dataBatch['name'][0][0]
    shapeRoot = '/'.join(name.split('/')[0:-1] )

    # Sample the point from visual hull
    refraction, reflection, maskTr = renderer.forward(
            originBatch, lookatBatch, upBatch,
            envBatch,
            normal1OptBatch, normal2OptBatch )
    renderedImg = torch.clamp(refraction + reflection, 0, 1 )
    error = torch.sum(torch.pow(renderedImg - imBatch, 2), dim=1 ).unsqueeze(1 ) * seg1IntBatch
    maskTr = (1 - maskTr) * seg1IntBatch

    if opt.viewMode == 0:
        feature, gtNormal, gtPoint, viewIds = sampler.sampleBestView(
                originBatch, lookatBatch, upBatch,
                pointVHBatch, normalPointVHBatch,
                pointBatch, normalPointBatch,
                maskTr, normal1OptBatch, error, depth1VHBatch, seg1IntBatch,
                depth1Batch, normal1Batch )
    elif opt.viewMode == 1:
        feature, gtNormal, gtPoint, viewIds = sampler.sampleNearestView(
                originBatch, lookatBatch, upBatch,
                pointVHBatch, normalPointVHBatch,
                pointBatch, normalPointBatch,
                maskTr, normal1OptBatch, error, depth1VHBatch, seg1IntBatch,
                depth1Batch, normal1Batch )
    elif opt.viewMode == 2:
        feature, viewIds = sampler.sampleNearestViewAverage(
                originBatch, lookatBatch, upBatch,
                pointVHBatch, normalPointVHBatch,
                pointBatch, normalPointBatch,
                maskTr, normal1OptBatch, error, depth1VHBatch, seg1IntBatch,
                depth1Batch, normal1Batch )
        gtNormal = normalPointGtBatch
        gtPoint = pointGtBatch

    if opt.lossMode == 1:
        gtNormal = normalPointBatch
        gtPoint = pointBatch
    elif opt.lossMode == 2:
        gtNormal = normalPointGtBatch
        gtPoint = pointGtBatch

    normalInitial = feature[:, 0:3].clone()

    # Predict New Points and New Normal
    ########################################
    pointPreds = []
    normalPreds = []
    pointPreds.append(pointVHBatch )
    normalPreds.append(feature[:, 0:3] )

    pointPredDelta, normalPred = pointNet(
            pointVHBatch.unsqueeze(0).permute([0, 2, 1]),
            normalInitial.unsqueeze(0).permute([0, 2, 1]),
            feature.unsqueeze(0).permute([0, 2, 1])  )
    pointPredDelta = pointPredDelta.squeeze(0 )
    normalPred = normalPred.squeeze(0 )

    pointPred = pointVHBatch + pointPredDelta
    pointPreds.append(pointPred )
    normalPreds.append(normalPred )

    pointErrs = []
    normalErrs = []
    meanAngleErrs = []
    medianAngleErrs = []

    if opt.lossMode == 0 or opt.lossMode == 1:
        for m in range(0, len(pointPreds ) ):
            pointErrs.append(torch.mean(torch.pow(pointPreds[m] - gtPoint, 2) ) )

        for m in range(0, len(normalPreds ) ):
            normalErrs.append(torch.mean(torch.pow(normalPreds[m] - gtNormal, 2) ) )
            meanAngleErrs.append(180.0/np.pi * torch.mean(torch.acos(
                torch.clamp( torch.sum(normalPreds[m] * gtNormal, dim=1 ), -1, 1) ) ) )
            medianAngleErrs.append(180.0/np.pi * torch.median(torch.acos(
                torch.clamp( torch.sum( normalPreds[m] * gtNormal, dim=1), -1, 1) ) ) )
    elif opt.lossMode == 2:
        assert(len(pointPreds) == len(normalPreds ) )
        for m in range(0, len(pointPreds) ):
            dist1, id1, dist2, id2 = chamferDist(gtPoint.unsqueeze(0),
                    pointPreds[m].unsqueeze(0) )
            pointErrs.append(0.5 * (torch.mean(dist1) + torch.mean(dist2) )  )
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
    utils.writeErrToScreen('normal', normalErrs, epoch, j )
    utils.writeErrToScreen('meanAngle', meanAngleErrs, epoch, j )
    utils.writeErrToScreen('medianAngle', medianAngleErrs, epoch, j )

    utils.writeErrToFile('point', pointErrs, testingLog, epoch, j )
    utils.writeErrToFile('normal', normalErrs, testingLog, epoch, j )
    utils.writeErrToFile('meanAngle', meanAngleErrs, testingLog, epoch, j )
    utils.writeErrToFile('medianAngle', medianAngleErrs, testingLog, epoch, j )

    pointErrsNpList = np.concatenate([pointErrsNpList, utils.turnErrorIntoNumpy(pointErrs ) ], axis=0 )
    normalErrsNpList = np.concatenate([normalErrsNpList, utils.turnErrorIntoNumpy(normalErrs ) ], axis=0 )
    meanAngleErrsNpList = np.concatenate([meanAngleErrsNpList, utils.turnErrorIntoNumpy(meanAngleErrs ) ], axis=0 )
    medianAngleErrsNpList = np.concatenate([medianAngleErrsNpList, utils.turnErrorIntoNumpy(medianAngleErrs ) ], axis=0 )

    utils.writeNpErrToScreen('pointAccu', np.mean(pointErrsNpList[1:j+1, :], axis=0), epoch, j )
    utils.writeNpErrToScreen('normalAccu', np.mean(normalErrsNpList[1:j+1, :], axis=0), epoch, j )
    utils.writeNpErrToScreen('meanAngleAccu', np.mean(meanAngleErrsNpList[1:j+1, :], axis=0), epoch, j )
    utils.writeNpErrToScreen('medianAngleAccu', np.mean(medianAngleErrsNpList[1:j+1, :], axis=0), epoch, j )

    utils.writeNpErrToFile('pointAccu', np.mean(pointErrsNpList[1:j+1, :], axis=0), testingLog, epoch, j )
    utils.writeNpErrToFile('normalAccu', np.mean(normalErrsNpList[1:j+1, :], axis=0), testingLog, epoch, j )
    utils.writeNpErrToFile('meanAngleAccu', np.mean(meanAngleErrsNpList[1:j+1, :], axis=0), testingLog, epoch, j )
    utils.writeNpErrToFile('medianAngleAccu', np.mean(medianAngleErrsNpList[1:j+1, :], axis=0), testingLog, epoch, j )


    shapeName = osp.join(opt.outputRoot, 'pointCloud_%d_view_' % opt.camNum )
    outputName = osp.join(shapeRoot, 'reconstruct_%d_view_' % opt.camNum )
    if opt.viewMode == 0:
        shapeName += 'renderError'
        outputName += 'renderError'
    elif opt.viewMode == 1:
        shapeName += 'nearest'
        outputName += 'nearest'
    elif opt.viewMode == 2:
        shapeName += 'average'
        outputName += 'average'

    if opt.isNoRenderError:
        shapeName += '_norendering'
        outputName += '_norendering'

    if opt.isBaseLine:
        shapeName += '_baseLine.ply'
        outputName += '_baseLine.ply'
        pointArr = pointPreds[0].data.cpu().numpy()
        normalArr = normalPreds[0].data.cpu().numpy()
    else:
        shapeName += '_loss_'
        outputName += '_loss_'
        if opt.lossMode == 0:
            shapeName += 'view'
            outputName += 'view'
        elif opt.lossMode == 1:
            shapeName += 'nearest'
            outputName += 'nearest'
        elif opt.lossMode == 2:
            shapeName += 'chamfer'
            outputName += 'chamfer'
        shapeName += '.ply'
        outputName += '.ply'
        pointArr = pointPreds[1].data.cpu().numpy()
        normalArr = normalPreds[1].data.cpu().numpy()

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pointArr )
    pcd.normals = o3d.utility.Vector3dVector(normalArr )
    pcd.colors = o3d.utility.Vector3dVector(0.5 * (normalArr + 1) )
    o3d.io.write_point_cloud(shapeName, pcd)

    colmapTrim = 3
    colmapDepth = 8
    colmapNumThreads = 8
    cmd = 'colmap poisson_mesher --input_path %s ' \
            + '--PoissonMeshing.trim %d ' \
            + '--PoissonMeshing.depth %d ' \
            + '--PoissonMeshing.num_threads %d ' \
            + '--output_path %s'
    cmd = cmd % (shapeName, colmapTrim, colmapDepth, colmapNumThreads, outputName )
    print(cmd )
    os.system(cmd )

    # Smooth the mesh
    cmd = 'xvfb-run -a -s "-screen 0 800x600x24" meshlabserver -i %s -o %s -om vn -s remesh.mlx' % \
            (outputName,  osp.join(shapeRoot, outputName.replace('.ply', '_Subd.ply') ) )
    print(cmd )
    os.system(cmd )

    if j == 1 or j % 2000 == 0:
        vutils.save_image( (torch.clamp(imBatch, 0, 1)**(1.0/2.2) *seg1IntBatch.expand_as(imBatch) ).data,
                '{0}/{1}_im.png'.format(opt.outputRoot, j), nrow=5 )
        vutils.save_image( ( 0.5 * (normal1OptBatch + 1 ) * seg1IntBatch ).data,
                '{0}/{1}_normal1Opt.png'.format(opt.outputRoot, j), nrow=5 )
        vutils.save_image( (0.5 * (normal2OptBatch + 1) * seg1IntBatch ).data,
                '{0}/{1}_normal2Opt.png'.format(opt.outputRoot, j), nrow=5 )
        vutils.save_image( (seg1IntBatch ).data,
                '{0}/{1}_seg1Int.png'.format(opt.outputRoot, j), nrow=5 )
        utils.writePointWithComputedNormal(osp.join(opt.outputRoot, '%d_gtPt.ply' % j), gtPoint, gtNormal )
        utils.writePointWithPredictedNormal(osp.join(opt.outputRoot, '%d_gtNormal.ply' % j), gtPoint, gtNormal )

        for m in range(0, len(pointPreds ) ):
            utils.writePointWithComputedNormal(osp.join(opt.outputRoot, '%d_predPt_%d.ply' % (j, m) ), pointPreds[m], normalPreds[m] )
            utils.writePointWithPredictedNormal(osp.join(opt.outputRoot, '%d_predNormal_%d.ply' % (j, m) ), pointPreds[m], normalPreds[m] )

testingLog.close()

# Save the error record
np.save('{0}/pointError_{1}.npy'.format(opt.outputRoot, epoch), pointErrsNpList )
np.save('{0}/normalError_{1}.npy'.format(opt.outputRoot, epoch), normalErrsNpList )
