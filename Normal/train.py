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

parser = argparse.ArgumentParser()
# The locationi of training set
parser.add_argument('--dataRoot', default='../../Data/Images%d/train/', help='path to images' )
parser.add_argument('--shapeRoot', default='../../Data/Shapes/train/', help='path to images' )
parser.add_argument('--experiment', default=None, help='the path to store samples and models'  )
# The basic training setting
parser.add_argument('--nepoch', type=int, default=10, help='the number of epochs for training' )
parser.add_argument('--batchSize', type=int, default=10, help='input batch size' )
parser.add_argument('--imageHeight', type=int, default=192, help='the height / width of the input image to network' )
parser.add_argument('--imageWidth', type=int, default=256, help='the height / width of the input image to network' )
parser.add_argument('--envHeight', type=int, default=256, help='the height / width of the input envmap to network' )
parser.add_argument('--envWidth', type=int, default=512, help='the height / width of the input envmap to network' )
# The parameters
parser.add_argument('--camNum', type=int, default=10, help='the number of views to create the visual hull' )
parser.add_argument('--sampleNum', type=int, default=1, help='the sample num for the cost volume')
parser.add_argument('--shapeStart', type=int, default=0, help='the start id of the shape')
parser.add_argument('--shapeEnd', type=int, default=3000, help='the end id of the shape')
parser.add_argument('--isAddCostVolume', action='store_true', help='whether to use cost volume or not' )
parser.add_argument('--poolingMode', type=int, default=2, help='0: maxpooling, 1: average pooling 2: learnable pooling' )
parser.add_argument('--isNoErrMap', action='store_true', help = 'whether to remove the error map in the input')
# The rendering parameters
parser.add_argument('--eta1', type=float, default=1.0003, help='the index of refraction of air' )
parser.add_argument('--eta2', type=float, default=1.4723, help='the index of refraction of glass' )
parser.add_argument('--fov', type=float, default=63.4, help='the field of view of camera' )
# The loss parameters
parser.add_argument('--normalWeight', type=float, default=1.0, help='the weight for normal' )
# The gpu setting
parser.add_argument('--cuda', action='store_true', help='enables cuda' )
parser.add_argument('--deviceIds', type=int, nargs='+', default=[0], help='the gpus used for training network' )

opt = parser.parse_args()
print(opt)

opt.gpuId = opt.deviceIds[0]
opt.dataRoot = opt.dataRoot % opt.camNum

nw = opt.normalWeight

####################################
# Initialize the angles
if opt.camNum == 5:
    thetas = [0, 25, 25, 25]
    phis = [0, 0, 120, 240]
elif opt.camNum == 10:
    thetas = [0, 15, 15, 15]
    phis = [0, 0, 120, 240]
elif opt.camNum == 20:
    thetas = [0, 10, 10, 10]
    phis = [0, 0, 120, 240]

thetaJitters = [0, 0, 0, 0]
phiJitters = [0, 0, 0, 0]

thetas = np.array(thetas ).astype(np.float32 ) / 180.0 * np.pi
phis = np.array(phis ).astype(np.float32 ) / 180.0 * np.pi
thetaJitters = np.array(thetaJitters ).astype(np.float32 ) / 180.0 * np.pi
phiJitters = np.array(phiJitters ).astype(np.float32 ) / 180.0 * np.pi
angleNum = thetas.shape[0]
####################################

if opt.experiment is None:
    opt.experiment = "check%d_normal_nw%.2f" % (opt.camNum, nw )
    if opt.isNoErrMap:
        opt.experiment += '_noerr'
    if opt.isAddCostVolume:
        if opt.poolingMode == 0:
            opt.experiment +=  '_volume_sp%d_an%d_maxpool' % (opt.sampleNum, angleNum )
        elif opt.poolingMode == 1:
            opt.experiment += '_volume_sp%d_an%d_avgpool' % (opt.sampleNum, angleNum )
        elif opt.poolingMode == 2:
            opt.experiment += '_volume_sp%d_an%d_weigtedSum' % (opt.sampleNum, angleNum )
        else:
            print("Wrong: unrecognizable pooling mode.")
            assert(False)

os.system('mkdir {0}'.format( opt.experiment ) )
os.system('cp *.py %s' % opt.experiment )

opt.seed = 0
print("Random Seed: ", opt.seed )
random.seed(opt.seed )
torch.manual_seed(opt.seed )

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda" )

####################################
# Initialize Network
encoder = nn.DataParallel(models.encoder(isAddCostVolume = opt.isAddCostVolume ), device_ids = opt.deviceIds )
decoder = nn.DataParallel(models.decoder(), device_ids = opt.deviceIds )
normalFeature = nn.DataParallel(models.normalFeature(), device_ids = opt.deviceIds )
normalPool = Variable(torch.ones([1, angleNum * angleNum, 1, 1, 1], dtype=torch.float32 ) )

##############  ######################
# Send things into GPU
if opt.cuda:
    encoder = encoder.cuda()
    decoder = decoder.cuda()
    normalFeature = normalFeature.cuda()
    normalPool = normalPool.cuda()
####################################

# Other modules
renderer = models.renderer(eta1 = opt.eta1, eta2 = opt.eta2,
        isCuda = opt.cuda, gpuId = opt.gpuId,
        batchSize = opt.batchSize,
        fov = opt.fov,
        imWidth=opt.imageWidth, imHeight = opt.imageHeight,
        envWidth = opt.envWidth, envHeight = opt.envHeight )

####################################
# Initial Optimizer
opNormalFeature = optim.Adam(normalFeature.parameters(), lr=1e-4, betas=(0.5, 0.999) )
normalPool.requires_grad = True
opNormalPool = optim.Adam([normalPool ], lr=1e-4, betas=(0.5, 0.999) )
opEncoder = optim.Adam(encoder.parameters(), lr=1e-4, betas=(0.5, 0.999) )
opDecoder = optim.Adam(decoder.parameters(), lr=1e-4, betas=(0.5, 0.999) )
#####################################


####################################
if opt.isAddCostVolume:
    buildCostVolume = models.buildCostVolume(
            thetas = thetas, phis = phis,
            thetaJitters = thetaJitters,
            phiJitters = phiJitters,
            eta1 = opt.eta1, eta2 = opt.eta2,
            batchSize = opt.batchSize,
            fov = opt.fov,
            imWidth = opt.imageWidth, imHeight = opt.imageHeight,
            envWidth = opt.envWidth, envHeight = opt.envHeight,
            sampleNum = opt.sampleNum )
else:
    buildCostVolume = None

brdfDataset = dataLoader.BatchLoader(
        opt.dataRoot, shapeRoot = opt.shapeRoot,
        imHeight = opt.imageHeight, imWidth = opt.imageWidth,
        envHeight = opt.envHeight, envWidth = opt.envWidth,
        isRandom = True, phase='TRAIN', rseed = 1,
        isLoadVH = True, isLoadEnvmap = True, isLoadCam = True,
        shapeRs = opt.shapeStart, shapeRe = opt.shapeEnd,
        camNum = opt.camNum )
brdfLoader = DataLoader( brdfDataset, batch_size = 1, num_workers = 4, shuffle = True )

j = 0

normal1ErrsNpList = np.ones( [1, 2], dtype = np.float32 )
meanAngle1ErrsNpList = np.ones([1, 2], dtype = np.float32 )
medianAngle1ErrsNpList = np.ones([1, 2], dtype = np.float32 )
normal2ErrsNpList = np.ones( [1, 2], dtype = np.float32 )
meanAngle2ErrsNpList = np.ones([1, 2], dtype = np.float32 )
medianAngle2ErrsNpList = np.ones([1, 2], dtype = np.float32 )
renderedErrsNpList = np.ones([1, 2], dtype=np.float32 )

for epoch in list(range(0, opt.nepoch ) ):
    trainingLog = open('{0}/trainingLog_{1}.txt'.format(opt.experiment, epoch ), 'w' )
    for i, dataBatch in enumerate(brdfLoader):
        j += 1

        # Load ground-truth from cpu to gpu
        normal1_cpu = dataBatch['normal1'].squeeze(0)
        normal1Batch = Variable(normal1_cpu ).cuda()

        seg1_cpu = dataBatch['seg1'].squeeze(0 )
        seg1Batch = Variable((seg1_cpu ) ).cuda()

        normal2_cpu = dataBatch['normal2'].squeeze(0 )
        normal2Batch = Variable(normal2_cpu ).cuda()

        seg2_cpu = dataBatch['seg2'].squeeze(0 )
        seg2Batch = Variable(seg2_cpu ).cuda()

        # Load the image from cpu to gpu
        im_cpu = dataBatch['im'].squeeze(0 )
        imBatch = Variable(im_cpu ).cuda()

        imBg_cpu = dataBatch['imE'].squeeze(0 )
        imBgBatch = Variable(imBg_cpu ).cuda()

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

        # Load visual hull data
        normal1VH_cpu = dataBatch['normal1VH'].squeeze(0)
        normal1VHBatch = Variable(normal1VH_cpu ).cuda()

        seg1VH_cpu = dataBatch['seg1VH'].squeeze(0 )
        seg1VHBatch = Variable((seg1VH_cpu ) ).cuda()

        normal2VH_cpu = dataBatch['normal2VH'].squeeze(0 )
        normal2VHBatch = Variable(normal2VH_cpu ).cuda()

        seg2VH_cpu = dataBatch['seg2VH'].squeeze(0 )
        seg2VHBatch = Variable(seg2VH_cpu ).cuda()

        # Clear the gradient in optimizer
        opEncoder.zero_grad()
        opDecoder.zero_grad()
        opNormalFeature.zero_grad()
        opNormalPool.zero_grad()

        batchSize = normal1Batch.size(0 )
        ########################################################
        normal1Preds = []
        normal2Preds = []
        renderedImgs = []

        refraction, reflection, maskVH = renderer.forward(
                originBatch, lookatBatch, upBatch,
                envBatch,
                normal1VHBatch, normal2VHBatch )
        renderedImg = torch.clamp(refraction + reflection, 0, 1)

        normal1Preds.append(normal1VHBatch )
        normal2Preds.append(normal2VHBatch )
        renderedImgs.append(renderedImg )
        errorVH = torch.sum(torch.pow(renderedImg - imBatch, 2.0) * seg1Batch, dim=1).unsqueeze(1)

        if opt.isNoErrMap:
            inputBatch = torch.cat([imBatch, imBgBatch, seg1Batch,
                normal1VHBatch, normal2VHBatch, errorVH * 0, maskVH * 0], dim=1 )
        else:
            inputBatch = torch.cat([imBatch, imBgBatch, seg1Batch,
                normal1VHBatch, normal2VHBatch, errorVH, maskVH], dim=1 )
        if opt.isAddCostVolume:
            costVolume = buildCostVolume.forward(imBatch,
                    originBatch, lookatBatch, upBatch,
                    envBatch, normal1VHBatch, normal2VHBatch,
                    seg1Batch )
            volume = normalFeature(costVolume ).unsqueeze(1 )
            volume = volume.view([batchSize, angleNum * angleNum, 64, int(opt.imageHeight/2), int(opt.imageWidth/2)] )
            if opt.poolingMode == 0:
                volume = volume.transpose(1, 4).transpose(2, 3)
                volume = volume.reshape([-1, 64, angleNum, angleNum ] )
                volume = F.max_pool2d(volume, kernel_size = angleNum )
                volume = volume.reshape([batchSize, int(opt.imageWidth/4), int(opt.imageHeight/4), 64] )
                volume = volume.transpose(1, 3)
            elif opt.poolingMode == 1:
                volume = volume.transpose(1, 4 ).transpose(2, 3 )
                volume = volume.reshape([-1, 64, angleNum, angleNum ] )
                volume = F.avg_pool2d(volume, kernel_size = angleNum )
                volume = volume.reshape([batchSize, int(opt.imageWidth/4), int(opt.imageHeight/4), 64] )
                volume = volume.transpose(1, 3 )
            elif opt.poolingMode == 2:
                weight = F.softmax(normalPool, dim=1 )
                volume = torch.sum(volume * normalPool, dim=1 )
            else:
                assert(False )
            x = encoder(inputBatch, volume )
        else:
            x = encoder(inputBatch )

        normal1Pred, normal2Pred = decoder(x )
        refraction, reflection, _ = renderer.forward(
                originBatch, lookatBatch, upBatch,
                envBatch,
                normal1Pred, normal2Pred )
        renderedImg = torch.clamp(refraction + reflection, 0, 1)

        normal1Preds.append(normal1Pred )
        normal2Preds.append(normal2Pred )
        renderedImgs.append(renderedImg )
        ########################################################

        # Compute the error
        normal1Errs = []
        meanAngle1Errs = []
        medianAngle1Errs = []

        normal2Errs = []
        meanAngle2Errs = []
        medianAngle2Errs = []

        renderedErrs = []

        pixel1Num = max( (torch.sum(seg1Batch ).cpu().data).item(), 1)
        pixel2Num = max( (torch.sum(seg2Batch).cpu().data ).item(), 1)
        for m in range(0, len(normal1Preds) ):
            normal1Errs.append( torch.sum( (normal1Preds[m] - normal1Batch)
                    * (normal1Preds[m] - normal1Batch) * seg1Batch.expand_as(imBatch) ) / pixel1Num / 3.0 )
            meanAngle1Errs.append(180.0/np.pi * torch.sum(torch.acos(
                torch.clamp( torch.sum(normal1Preds[m] * normal1Batch, dim=1 ), -1, 1) )* seg1Batch.squeeze(1) ) / pixel1Num )
            if pixel1Num != 1:
                medianAngle1Errs.append(180.0/np.pi * torch.median(torch.acos(
                    torch.clamp(torch.sum( normal1Preds[m] * normal1Batch, dim=1), -1, 1)[seg1Batch.squeeze(1) != 0] ) ) )
            else:
                medianAngle1Errs.append(0*torch.sum(normal1Preds[m]) )

        for m in range(0, len(normal2Preds) ):
            normal2Errs.append( torch.sum( (normal2Preds[m] - normal2Batch)
                    * (normal2Preds[m] - normal2Batch) * seg1Batch.expand_as(imBatch) ) / pixel1Num / 3.0 )
            meanAngle2Errs.append(180.0/np.pi * torch.sum(torch.acos(
                torch.clamp( torch.sum(normal2Preds[m] * normal2Batch, dim=1 ), -1, 1) )* seg1Batch.squeeze(1) ) / pixel1Num )
            if pixel1Num != 1:
                medianAngle2Errs.append(180.0/np.pi * torch.median(torch.acos(
                    torch.clamp(torch.sum( normal2Preds[m] * normal2Batch, dim=1), -1, 1)[seg1Batch.squeeze(1) != 0] ) ) )
            else:
                medianAngle2Errs.append(0*torch.sum(normal2Preds[m] ) )

        for m in range(0, len(renderedImgs ) ):
            renderedErrs.append( torch.sum( (renderedImgs[m] - imBgBatch)
                * (renderedImgs[m] - imBgBatch ) * seg2Batch.expand_as(imBatch ) ) / pixel2Num / 3.0 )

        totalErr = nw * (normal1Errs[-1] + normal2Errs[-1])
        totalErr.backward()

        # Update the network parameter
        opDecoder.step()
        opEncoder.step()
        opNormalFeature.step()
        opNormalPool.step()

        # Output training error
        utils.writeErrToScreen('normal1', normal1Errs, epoch, j)
        utils.writeErrToScreen('medianAngle1', medianAngle1Errs, epoch, j)
        utils.writeErrToScreen('meanAngle1', meanAngle1Errs, epoch, j)
        utils.writeErrToScreen('normal2', normal2Errs, epoch, j)
        utils.writeErrToScreen('medianAngle2', medianAngle2Errs, epoch, j)
        utils.writeErrToScreen('meanAngle2', meanAngle2Errs, epoch, j)
        utils.writeErrToScreen('rendered', renderedErrs, epoch, j)

        utils.writeErrToFile('normal1', normal1Errs, trainingLog, epoch, j)
        utils.writeErrToFile('medianAngle1', medianAngle1Errs, trainingLog, epoch, j)
        utils.writeErrToFile('meanAngle1', meanAngle1Errs, trainingLog, epoch, j)
        utils.writeErrToFile('normal2', normal2Errs, trainingLog, epoch, j)
        utils.writeErrToFile('medianAngle2', medianAngle2Errs, trainingLog, epoch, j)
        utils.writeErrToFile('meanAngle2', meanAngle2Errs, trainingLog, epoch, j)
        utils.writeErrToFile('rendered', renderedErrs, trainingLog, epoch, j)

        normal1ErrsNpList = np.concatenate( [normal1ErrsNpList, utils.turnErrorIntoNumpy(normal1Errs)], axis=0 )
        medianAngle1ErrsNpList = np.concatenate( [medianAngle1ErrsNpList, utils.turnErrorIntoNumpy(medianAngle1Errs)], axis=0 )
        meanAngle1ErrsNpList = np.concatenate( [meanAngle1ErrsNpList, utils.turnErrorIntoNumpy(meanAngle1Errs)], axis=0 )
        normal2ErrsNpList = np.concatenate( [normal2ErrsNpList, utils.turnErrorIntoNumpy(normal2Errs)], axis=0 )
        medianAngle2ErrsNpList = np.concatenate( [medianAngle2ErrsNpList, utils.turnErrorIntoNumpy(medianAngle2Errs)], axis=0 )
        meanAngle2ErrsNpList = np.concatenate( [meanAngle2ErrsNpList, utils.turnErrorIntoNumpy(meanAngle2Errs)], axis=0 )
        renderedErrsNpList = np.concatenate( [renderedErrsNpList, utils.turnErrorIntoNumpy(renderedErrs ) ], axis=0 )

        if j < 1000:
            utils.writeNpErrToScreen('normal1Accu', np.mean(normal1ErrsNpList[1:j+1, :], axis=0), epoch, j)
            utils.writeNpErrToScreen('medianAngle1Accu', np.mean(medianAngle1ErrsNpList[1:j+1, :], axis=0), epoch, j)
            utils.writeNpErrToScreen('meanAngle1Accu', np.mean(meanAngle1ErrsNpList[1:j+1, :], axis=0), epoch, j)
            utils.writeNpErrToScreen('normal2Accu', np.mean(normal2ErrsNpList[1:j+1, :], axis=0), epoch, j)
            utils.writeNpErrToScreen('medianAngle2Accu', np.mean(medianAngle2ErrsNpList[1:j+1, :], axis=0), epoch, j)
            utils.writeNpErrToScreen('meanAngle2Accu', np.mean(meanAngle2ErrsNpList[1:j+1, :], axis=0), epoch, j)
            utils.writeNpErrToScreen('renderedAccu', np.mean(renderedErrsNpList[1:j+1, :], axis=0), epoch, j)

            utils.writeNpErrToFile('normal1Accu', np.mean(normal1ErrsNpList[1:j+1, :], axis=0), trainingLog, epoch, j)
            utils.writeNpErrToFile('medianAngle1Accu', np.mean(medianAngle1ErrsNpList[1:j+1, :], axis=0), trainingLog, epoch, j)
            utils.writeNpErrToFile('meanAngle1Accu', np.mean(meanAngle1ErrsNpList[1:j+1, :], axis=0), trainingLog, epoch, j)
            utils.writeNpErrToFile('normal2Accu', np.mean(normal2ErrsNpList[1:j+1, :], axis=0), trainingLog, epoch, j)
            utils.writeNpErrToFile('medianAngle2Accu', np.mean(medianAngle2ErrsNpList[1:j+1, :], axis=0), trainingLog, epoch, j)
            utils.writeNpErrToFile('meanAngle2Accu', np.mean(meanAngle2ErrsNpList[1:j+1, :], axis=0), trainingLog, epoch, j)
            utils.writeNpErrToFile('renderedAccu', np.mean(renderedErrsNpList[1:j+1, :], axis=0), trainingLog, epoch, j)
        else:
            utils.writeNpErrToScreen('normal1Accu', np.mean(normal1ErrsNpList[j-999:j+1, :], axis=0), epoch, j )
            utils.writeNpErrToScreen('medianAngle1Accu', np.mean(medianAngle1ErrsNpList[j-999:j+1, :], axis=0), epoch, j )
            utils.writeNpErrToScreen('meanAngle1Accu', np.mean(meanAngle1ErrsNpList[j-999:j+1, :], axis=0), epoch, j )
            utils.writeNpErrToScreen('normal2Accu', np.mean(normal2ErrsNpList[j-999:j+1, :], axis=0), epoch, j )
            utils.writeNpErrToScreen('medianAngle2Accu', np.mean(medianAngle2ErrsNpList[j-999:j+1, :], axis=0), epoch, j )
            utils.writeNpErrToScreen('meanAngle2Accu', np.mean(meanAngle2ErrsNpList[j-999:j+1, :], axis=0), epoch, j )
            utils.writeNpErrToScreen('renderedAccu', np.mean(renderedErrsNpList[j-999:j+1, :], axis=0), epoch, j )

            utils.writeNpErrToFile('normal1Accu', np.mean(normal1ErrsNpList[j-999:j+1, :], axis=0), trainingLog, epoch, j )
            utils.writeNpErrToFile('medianAngle1Accu', np.mean(medianAngle1ErrsNpList[j-999:j+1, :], axis=0), trainingLog, epoch, j )
            utils.writeNpErrToFile('meanAngle1Accu', np.mean(meanAngle1ErrsNpList[j-999:j+1, :], axis=0), trainingLog, epoch, j )
            utils.writeNpErrToFile('normal2Accu', np.mean(normal2ErrsNpList[j-999:j+1, :], axis=0), trainingLog, epoch, j )
            utils.writeNpErrToFile('medianAngle2Accu', np.mean(medianAngle2ErrsNpList[j-999:j+1, :], axis=0), trainingLog, epoch, j )
            utils.writeNpErrToFile('meanAngle2Accu', np.mean(meanAngle2ErrsNpList[j-999:j+1, :], axis=0), trainingLog, epoch, j )
            utils.writeNpErrToFile('renderedAccu', np.mean(renderedErrsNpList[j-999:j+1, :], axis=0), trainingLog, epoch, j )


        if j == 1 or j == 1000 or j% 2000 == 0:
            vutils.save_image( (0.5*(normal1Batch + 1)*seg1Batch.expand_as(imBatch) ).data,
                    '{0}/{1}_normal1Gt.png'.format(opt.experiment, j), nrow=5 )
            vutils.save_image( (0.5*(normal2Batch + 1)*seg1Batch.expand_as(imBatch) ).data,
                    '{0}/{1}_normal2Gt.png'.format(opt.experiment, j), nrow=5 )

            vutils.save_image( (torch.clamp(imBatch, 0, 1)**(1.0/2.2) *seg2Batch.expand_as(imBatch) ).data,
                    '{0}/{1}_im.png'.format(opt.experiment, j), nrow=5 )
            vutils.save_image( ( torch.clamp(imBgBatch, 0, 1)**(1.0/2.2) ).data,
                    '{0}/{1}_imBg.png'.format(opt.experiment, j), nrow=5 )

            vutils.save_image( seg2Batch,
                    '{0}/{1}_mask2.png'.format(opt.experiment, j), nrow=5 )

            # Save the predicted results
            for n in range(0, len(normal1Preds) ):
                vutils.save_image( ( 0.5*(normal1Preds[n] + 1)*seg1Batch.expand_as(imBatch) ).data,
                        '{0}/{1}_normal1Pred_{2}.png'.format(opt.experiment, j, n), nrow=5 )
            for n in range(0, len(normal2Preds) ):
                vutils.save_image( ( 0.5*(normal2Preds[n] + 1)*seg1Batch.expand_as(imBatch) ).data,
                        '{0}/{1}_normal2Pred_{2}.png'.format(opt.experiment, j, n), nrow=5 )
            for n in range(0, len(renderedImgs ) ):
                vutils.save_image( (torch.clamp(renderedImgs[n], 0, 1)**(1.0/2.2) *seg2Batch.expand_as(imBatch) ) .data,
                        '{0}/{1}_renederedImg_{2}.png'.format(opt.experiment, j, n), nrow=5 )

    trainingLog.close()

    # Update the training rate
    if (epoch + 1) % 2 == 0:
        for param_group in opEncoder.param_groups:
            param_group['lr'] /= 2
        for param_group in opDecoder.param_groups:
            param_group['lr'] /= 2
        for param_group in opNormalFeature.param_groups:
            param_group['lr'] /= 2
        for param_group in opNormalPool.param_groups:
            param_group['lr'] /= 2

    # Save the error record
    np.save('{0}/normal1Error_{1}.npy'.format(opt.experiment, epoch), normal1ErrsNpList )
    np.save('{0}/medianAngle1Error_{1}.npy'.format(opt.experiment, epoch), medianAngle1ErrsNpList )
    np.save('{0}/meanAngle1Error_{1}.npy'.format(opt.experiment, epoch), meanAngle1ErrsNpList )
    np.save('{0}/normal2Error_{1}.npy'.format(opt.experiment, epoch), normal2ErrsNpList )
    np.save('{0}/medianAngle2Error_{1}.npy'.format(opt.experiment, epoch), medianAngle2ErrsNpList )
    np.save('{0}/meanAngle2Error_{1}.npy'.format(opt.experiment, epoch), meanAngle2ErrsNpList )
    np.save('{0}/renderError_{1}.npy'.format(opt.experiment, epoch), renderedErrsNpList )

    # save the models
    torch.save(encoder.module.state_dict(), '{0}/encoder_{1}.pth'.format(opt.experiment, epoch) )
    torch.save(decoder.module.state_dict(), '{0}/decoder_{1}.pth'.format(opt.experiment, epoch) )
    torch.save(normalFeature.module.state_dict(), '{0}/normalFeature_{1}.pth'.format(opt.experiment, epoch ) )
    torch.save(normalPool.data.cpu(), '{0}/normalPool_{1}.pth'.format(opt.experiment, epoch ) )
