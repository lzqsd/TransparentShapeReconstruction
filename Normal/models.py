import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


class normalFeature(nn.Module ):
    def __init__(self):
        super(normalFeature, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=8, out_channels=64, kernel_size=4, stride=2, padding=1, bias=True )
        self.gn1 = nn.GroupNorm(4, 64 )
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True )
        self.gn2 = nn.GroupNorm(4, 64 )

    def forward(self, x):
        x1 = F.relu(self.gn1(self.conv1(x) ), True )
        x2 = F.relu(self.gn2(self.conv2(x1) ), True )
        return x2

class encoder(nn.Module ):
    def __init__(self, isAddCostVolume = False ):
        super(encoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=15, out_channels=64, kernel_size=4, stride=2, padding=1, bias=True )
        self.gn1 = nn.GroupNorm(4, 64 )
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True )
        self.gn2 = nn.GroupNorm(4, 64 )
        if isAddCostVolume == True:
            self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1, bias=True )
        else:
            self.conv3 = nn.Conv2d(in_channels=64, out_channels=256, kernel_size=4, stride=2, padding=1, bias=True )
        self.gn3 = nn.GroupNorm(16, 256 )
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=True )
        self.gn4 = nn.GroupNorm(16, 256 )
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1, bias=True )
        self.gn5 = nn.GroupNorm(32, 512 )
        self.conv6 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, bias=True )
        self.gn6 = nn.GroupNorm(32, 512 )

    def forward(self, x, volume = None ):
        x1 = F.relu(self.gn1(self.conv1(x) ), True )
        x2 = F.relu(self.gn2(self.conv2(x1) ), True )

        if not volume is None:
            if volume.size(2) != x2.size(2) or volume.size(3) != x2.size(3):
                volume = F.interpolate(volume, [x2.size(2), x2.size(3)], mode='bilinear')
            x2 = torch.cat([volume, x2], dim=1 )
        x3 = F.relu(self.gn3(self.conv3(x2 ) ), True )
        x4 = F.relu(self.gn4(self.conv4(x3) ), True )
        x5 = F.relu(self.gn5(self.conv5(x4) ), True )
        x = F.relu(self.gn6(self.conv6(x5) ), True )

        return x


class decoder(nn.Module ):
    def __init__(self ):
        super(decoder, self ).__init__()
        # branch for normal prediction
        self.dconv0 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, bias=True )
        self.dgn0 = nn.GroupNorm(32, 512 )
        self.dconv1 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=1, bias=True )
        self.dgn1 = nn.GroupNorm(16, 256 )
        self.dconv2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=True )
        self.dgn2 = nn.GroupNorm(16, 256 )
        self.dconv3 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1, bias=True )
        self.dgn3 = nn.GroupNorm(8, 128 )
        self.dconv4 = nn.Conv2d(in_channels=128,  out_channels=128, kernel_size=3, stride=1, padding=1, bias=True )
        self.dgn4 = nn.GroupNorm(8, 128 )
        self.dconv5 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True )
        self.dgn5 = nn.GroupNorm(4, 64 )

        self.convFinal = nn.Conv2d(in_channels=64, out_channels = 6, kernel_size = 3, stride=1, padding=1, bias=True )

    def forward(self, x):
        x1 = F.relu( self.dgn0(self.dconv0(x ) ), True)
        x2 = F.relu( self.dgn1(self.dconv1(x1) ), True)
        x3 = F.relu( self.dgn2(self.dconv2(F.interpolate(x2, scale_factor=2, mode='bilinear') ) ), True)
        x4 = F.relu( self.dgn3(self.dconv3(x3) ), True)
        x5 = F.relu( self.dgn4(self.dconv4(F.interpolate(x4, scale_factor=2, mode='bilinear') ) ), True)
        x6 = F.relu( self.dgn5(self.dconv5(x5) ), True)

        x_orig  = torch.tanh( self.convFinal(F.interpolate(x6, scale_factor=2, mode='bilinear') ) )

        normal1, normal2 = torch.split(x_orig, [3, 3], dim=1 )
        normal1 = normal1 / torch.clamp(torch.sqrt(torch.sum(normal1 * normal1, dim=1 ).unsqueeze(1) ), min=1e-6)
        normal2 = normal2 / torch.clamp(torch.sqrt(torch.sum(normal2 * normal2, dim=1 ).unsqueeze(1) ), min=1e-6)

        return normal1, normal2


class renderer():
    def __init__( self, eta1 = 1.0003, eta2 = 1.4723,
            isCuda = True, gpuId = 0,
            batchSize = 12,
            fov = 63.4, imWidth = 480, imHeight = 360,
            envWidth = 512, envHeight = 256 ):
        # Set the parameters
        self.eta1 = eta1
        self.eta2 = eta2
        self.batchSize = batchSize
        self.fov = fov / 180 * np.pi
        self.imHeight = imHeight
        self.imWidth = imWidth
        self.envHeight = envHeight
        self.envWidth = envWidth
        self.isCuda = isCuda
        self.gpuId = gpuId

        # Compute view direction
        x, y = np.meshgrid(np.linspace(-1, 1, imWidth),
                np.linspace(-1, 1, imHeight ) )
        tan_fovx = np.tan(self.fov / 2.0 )
        tan_fovy = tan_fovx / float(imWidth) * float(imHeight )
        x, y = x * tan_fovx, -y * tan_fovy
        z = -np.ones([imHeight, imWidth], dtype=np.float32 )
        x, y, z = x[np.newaxis, :, :], y[np.newaxis, :, :], z[np.newaxis, :, :]
        v = np.concatenate([x, y, z], axis=0).astype(np.float32 )
        v = v / np.maximum(np.sqrt(np.sum(v * v, axis=0) )[np.newaxis, :], 1e-6 )
        v = v[np.newaxis, :, :, :]
        self.v = Variable(torch.from_numpy(v ) )

        # Compute the offset
        offset = np.arange(0, batchSize).reshape([batchSize, 1, 1, 1])
        offset = (offset * envWidth * envHeight ).astype(np.int64 )
        self.offset = Variable(torch.from_numpy(offset ) )

    def refraction(self, l, normal, eta1, eta2 ):
        # l n x 3 x imHeight x imWidth
        # normal n x 3 x imHeight x imWidth
        # eta1 float
        # eta2 float
        cos_theta = torch.sum(l * (-normal), dim=1 ).unsqueeze(1)
        i_p = l + normal * cos_theta
        t_p = eta1 / eta2 * i_p

        t_p_norm = torch.sum(t_p * t_p, dim=1)
        totalReflectMask = (t_p_norm.detach() > 0.999999).unsqueeze(1)

        t_i = torch.sqrt(1 - torch.clamp(t_p_norm, 0, 0.999999 ) ).unsqueeze(1).expand_as(normal ) * (-normal )
        t = t_i + t_p
        t = t / torch.sqrt( torch.clamp(torch.sum(t * t, dim=1 ), min=1e-10 ) ).unsqueeze(1)

        cos_theta_t = torch.sum(t * (-normal), dim=1 ).unsqueeze(1)

        e_i = (cos_theta_t * eta2 - cos_theta * eta1) / \
                torch.clamp(cos_theta_t * eta2 + cos_theta * eta1, min=1e-10 )
        e_p = (cos_theta_t * eta1 - cos_theta * eta2) / \
                torch.clamp(cos_theta_t * eta1 + cos_theta * eta2, min=1e-10 )

        attenuate = torch.clamp(0.5 * (e_i * e_i + e_p * e_p), 0, 1).detach()

        return t, attenuate, totalReflectMask

    def reflection(self, l, normal ):
        # l n x 3 x imHeight x imWidth
        # normal n x 3 x imHeight x imWidth
        # eta1 float
        # eta2 float
        cos_theta = torch.sum(l * (-normal), dim=1 ).unsqueeze(1)
        r_p = l + normal * cos_theta
        r_p_norm = torch.clamp(torch.sum(r_p * r_p, dim=1), 0, 0.999999 )
        r_i = torch.sqrt(1 - r_p_norm ).unsqueeze(1).expand_as(normal ) * normal
        r = r_p + r_i
        r = r / torch.sqrt(torch.clamp(torch.sum(r*r, dim=1), min=1e-10 ).unsqueeze(1) )

        return r

    def transformCoordinate(self, l, origin, lookat, up ):
        batchSize = origin.size(0 )
        assert(batchSize <= self.batchSize )

        # Rotate to world coordinate
        zAxis = origin - lookat
        yAxis = up
        xAxis = torch.cross(yAxis, zAxis, dim=1 )
        xAxis = xAxis / torch.sqrt(torch.clamp(torch.sum(xAxis * xAxis, dim=1).unsqueeze(1 ), min=1e-10 ) )
        yAxis = yAxis / torch.sqrt(torch.clamp(torch.sum(yAxis * yAxis, dim=1).unsqueeze(1 ), min=1e-10 ) )
        zAxis = zAxis / torch.sqrt(torch.clamp(torch.sum(zAxis * zAxis, dim=1).unsqueeze(1 ), min=1e-10 ) )

        xAxis = xAxis.view([batchSize, 3, 1, 1, 1])
        yAxis = yAxis.view([batchSize, 3, 1, 1, 1])
        zAxis = zAxis.view([batchSize, 3, 1, 1, 1])
        rotMat = torch.cat([xAxis, yAxis, zAxis], dim=2 )
        l = l.unsqueeze(1)

        l = torch.sum(rotMat.expand([batchSize, 3, 3, self.imHeight, self.imWidth ] ) * \
                l.expand([batchSize, 3, 3, self.imHeight, self.imWidth ] ), dim=2)
        l = l / torch.sqrt( torch.clamp(torch.sum(l*l, dim=1 ).unsqueeze(1), min=1e-10 ) )

        return l

    def sampleEnvLight(self, l, envmap ):
        batchSize = envmap.size(0 )
        assert(batchSize <= self.batchSize )
        channelNum = envmap.size(1 )

        l = torch.clamp(l, -0.999999, 0.999999)
        # Compute theta and phi
        x, y, z = torch.split(l, [1, 1, 1], dim=1 )
        theta = torch.acos(y )
        phi = torch.atan2( x, z )
        v = theta / np.pi * (self.envHeight-1)
        u = (-phi / np.pi / 2.0 + 0.5) * (self.envWidth-1)

        # Bilinear interpolation to get the new image
        offset = self.offset.detach()[0:batchSize, :]
        offset = offset.expand_as(u ).clone().cuda()

        u, v = torch.flatten(u), torch.flatten(v)
        u1 = torch.clamp(torch.floor(u).detach(), 0, self.envWidth-1)
        v1 = torch.clamp(torch.floor(v).detach(), 0, self.envHeight-1)
        u2 = torch.clamp(torch.ceil(u).detach(), 0, self.envWidth-1)
        v2 = torch.clamp(torch.ceil(v).detach(), 0, self.envHeight-1)

        w_r = (u - u1).unsqueeze(1)
        w_l = (1 - w_r )
        w_u = (v2 - v).unsqueeze(1)
        w_d = (1 - w_u )

        u1, v1 = u1.long(), v1.long()
        u2, v2 = u2.long(), v2.long()
        offset = torch.flatten(offset )
        size_0 = self.envWidth * self.envHeight * batchSize
        envmap = envmap.transpose(1, 2).transpose(2, 3).reshape([size_0, channelNum ])
        index = (v1 * self.envWidth + u2) + offset
        envmap_ru = torch.index_select(envmap, 0, index )
        index = (v2 * self.envWidth + u2) + offset
        envmap_rd = torch.index_select(envmap, 0, index )
        index = (v1 * self.envWidth + u1) + offset
        envmap_lu = torch.index_select(envmap, 0, index )
        index = (v2 * self.envWidth + u1) + offset
        envmap_ld = torch.index_select(envmap, 0, index )

        envmap_r = envmap_ru * w_u.expand_as(envmap_ru ) + \
                envmap_rd * w_d.expand_as(envmap_rd )
        envmap_l = envmap_lu * w_u.expand_as(envmap_lu ) + \
                envmap_ld * w_d.expand_as(envmap_ld )
        renderedImg = envmap_r * w_r.expand_as(envmap_r ) + \
                envmap_l * w_l.expand_as(envmap_l )

        # Post processing
        renderedImg = renderedImg.reshape([batchSize, self.imHeight, self.imWidth, 3])
        renderedImg = renderedImg.transpose(3, 2).transpose(2, 1)

        return renderedImg

    def getBackground(self, origin, lookat, up, envmap, normal1):
        l = self.v.expand_as(normal1 ).clone().cuda()
        l = self.transformCoordinate(l, origin, lookat, up)
        backImg = self.sampleEnvLight(l, envmap )
        return backImg

    def forward(self, origin, lookat, up, envmap, normal1, normal2 ):
        # origin n x 3
        # lookat n x 3
        # up n x 3
        # envmap n x 3 x envHeight x envWidth
        # normal1 n x 3 x imHeight x imWidth
        # normal2 n x 3 x imHeight x imWidth
        # view 3 x imHeight x imWidth
        l = self.v.expand_as(normal1 ).clone().cuda()
        l_t, attenuate1, mask1 = self.refraction(l, normal1, self.eta1, self.eta2 )
        l_t, attenuate2, mask2 = self.refraction(l_t, normal2, self.eta2, self.eta1 )

        l_r = self.reflection(l, normal1 )

        l_t = self.transformCoordinate(l_t, origin, lookat, up)
        l_r = self.transformCoordinate(l_r, origin, lookat, up)

        refractImg = self.sampleEnvLight(l_t, envmap )
        refractImg = refractImg * (1-attenuate1) * (1-attenuate2)

        reflectImg = self.sampleEnvLight(l_r, envmap )
        reflectImg = reflectImg * attenuate1

        mask = torch.clamp( (mask1 + mask2).float(), 0, 1)

        return refractImg, reflectImg, mask

class buildCostVolume():
    def __init__(self, thetas, phis, thetaJitters, phiJitters, isCuda = True,
            eta1 = 1.0003, eta2 = 1.4723,
            batchSize = 2,
            fov = 63.4, imWidth = 480, imHeight = 360,
            envWidth = 512, envHeight = 256, sampleNum = 50, lamb = 1.0):

        phis = phis.squeeze().astype(np.float32 )
        thetas = thetas.squeeze().astype(np.float32 )

        self.lamb = lamb

        thetas2, thetas1 = np.meshgrid(thetas, thetas )
        thetaJitters1, thetaJitters2 = np.meshgrid(thetaJitters, thetaJitters )

        phis2, phis1 = np.meshgrid(phis, phis )
        phiJitters1, phiJitters2 = np.meshgrid(phiJitters, phiJitters )

        thetas2, thetas1 = thetas2.reshape(1, -1, 1, 1, 1), thetas1.reshape(1, -1, 1, 1, 1)
        thetaJitters1 = thetaJitters1.reshape(1, -1, 1, 1, 1)
        thetaJitters2 = thetaJitters2.reshape(1, -1, 1, 1, 1)

        phis2, phis1 = phis2.reshape(1, -1, 1, 1, 1), phis1.reshape(1, -1, 1, 1, 1)
        phiJitters1 = phiJitters1.reshape(1, -1, 1, 1, 1)
        phiJitters2 = phiJitters2.reshape(1, -1, 1, 1, 1)

        self.thetas1 = Variable(torch.from_numpy(thetas1 ) )
        self.thetas2 = Variable(torch.from_numpy(thetas2 ) )
        self.thetaJitters1 = Variable(torch.from_numpy(thetaJitters1 ) )
        self.thetaJitters2 = Variable(torch.from_numpy(thetaJitters2 ) )

        self.phis1 = Variable(torch.from_numpy(phis1 ) )
        self.phis2 = Variable(torch.from_numpy(phis2 ) )
        self.phiJitters1 = Variable(torch.from_numpy(phiJitters1) )
        self.phiJitters2 = Variable(torch.from_numpy(phiJitters2) )

        self.inputNum = self.thetas1.size(1)
        self.renderer = renderer(eta1 = eta1, eta2 = eta2,
                isCuda = isCuda,
                batchSize = batchSize * self.inputNum,
                fov = fov, imWidth = imWidth, imHeight = imHeight,
                envWidth = envWidth, envHeight = envHeight )

        self.imWidth = imWidth
        self.imHeight = imHeight
        self.envWidth = envWidth
        self.envHeight = envHeight

        self.sampleNum = sampleNum

        up = np.array([0, 1, 0], dtype=np.float32 )
        up = up.reshape([1, 3, 1, 1])
        self.up = Variable(torch.from_numpy(up ) )

        self.isCuda = isCuda

    def forward(self, imBatch, origin, lookat, up, envmap, normal1, normal2, seg1Batch ):
        batchSize = normal1.size(0 )
        imBatch = imBatch.unsqueeze(1 ).expand([batchSize, self.inputNum, 3, self.imHeight, self.imWidth] )
        seg1Batch = seg1Batch.unsqueeze(1).expand([batchSize, self.inputNum, 1, self.imHeight, self.imWidth] )
        envmap = envmap.unsqueeze(1 ).expand([batchSize, self.inputNum, 3, self.envHeight, self.envWidth ] )
        origin = origin.unsqueeze(1 ).expand([batchSize, self.inputNum, 3] )
        lookat = lookat.unsqueeze(1 ).expand([batchSize, self.inputNum, 3] )
        up = up.unsqueeze(1 ).expand([batchSize, self.inputNum, 3] )

        imBatch = imBatch.reshape([batchSize * self.inputNum, 3, self.imHeight, self.imWidth] )
        seg1Batch = seg1Batch.reshape([batchSize * self.inputNum, 1, self.imHeight, self.imWidth ] )
        envmap = envmap.reshape([batchSize * self.inputNum, 3, self.envHeight, self.envWidth ] )
        origin = origin.reshape([batchSize * self.inputNum, 3] )
        lookat = lookat.reshape([batchSize * self.inputNum, 3] )
        up = up.reshape([batchSize * self.inputNum, 3] )

        normal1Best = Variable(torch.zeros([batchSize * self.inputNum, 3, self.imHeight, self.imWidth], dtype=torch.float32 ) ).cuda()
        normal2Best = Variable(torch.zeros([batchSize * self.inputNum, 3, self.imHeight, self.imWidth], dtype=torch.float32 ) ).cuda()

        for n in range(0, self.sampleNum ):
            upAxis = self.up.clone().cuda()
            yAxis1 = upAxis - torch.sum(upAxis * normal1, dim=1 ).unsqueeze(1) * normal1
            yAxis1 = yAxis1 / torch.sqrt( torch.clamp(torch.sum(yAxis1 * yAxis1, dim=1).unsqueeze(1), min=1e-10 ) )
            zAxis1 = normal1
            xAxis1 = torch.cross(yAxis1, zAxis1, dim=1 )
            xAxis1 = xAxis1 / torch.sqrt( torch.clamp(torch.sum(xAxis1 * xAxis1, dim=1).unsqueeze(1), min=1e-10 ) )

            yAxis2 = upAxis - torch.sum(upAxis * normal2, dim=1 ).unsqueeze(1) * normal2
            yAxis2 = yAxis2 / torch.sqrt( torch.clamp(torch.sum(yAxis2 * yAxis2, dim=1).unsqueeze(1), min=1e-10 ) )
            zAxis2 = normal2
            xAxis2 = torch.cross(yAxis2, zAxis2, dim=1 )
            xAxis2 = xAxis2 / torch.sqrt( torch.clamp(torch.sum(xAxis2 * xAxis2, dim=1).unsqueeze(1), min=1e-10 ) )

            xAxis1, xAxis2 = xAxis1.unsqueeze(1), xAxis2.unsqueeze(1)
            yAxis1, yAxis2 = yAxis1.unsqueeze(1), yAxis2.unsqueeze(1)
            zAxis1, zAxis2 = zAxis1.unsqueeze(1), zAxis2.unsqueeze(1)

            thetaOffsets1 = Variable(torch.from_numpy( (np.random.random(self.thetas1.size() ).astype(np.float32 ) ) ) )
            thetas1 = (self.thetas1 + self.thetaJitters1 * thetaOffsets1 ).cuda()
            phiOffsets1 = Variable(torch.from_numpy( (np.random.random(self.phis1.size() ).astype(np.float32 ) ) ) )
            phis1 = (self.phis1 + self.phiJitters1 * phiOffsets1 ).cuda()
            normal1Array = torch.sin(thetas1 ) * torch.cos(phis1 ) * xAxis1 \
                    + torch.sin(thetas1 ) * torch.sin(phis1 ) * yAxis1 \
                    + torch.cos(thetas1 )  * zAxis1

            thetaOffsets2 = Variable(torch.from_numpy( (np.random.random(self.thetas2.size() ).astype(np.float32 ) ) ) )
            thetas2 = (self.thetas2 + self.thetaJitters2 * thetaOffsets2 ).cuda()
            phiOffsets2 = Variable(torch.from_numpy( (np.random.random(self.phis2.size() ).astype(np.float32 ) ) ) )
            phis2 = (self.phis2 + self.phiJitters2 * phiOffsets2 ).cuda()
            normal2Array = torch.sin(thetas2 ) * torch.cos(phis2 ) * xAxis2 \
                    + torch.sin(thetas2 ) * torch.sin(phis2 ) * yAxis2 \
                    + torch.cos(thetas2 )  * zAxis2

            normal1Array = normal1Array.reshape([batchSize * self.inputNum, 3, self.imHeight, self.imWidth] )
            normal2Array = normal2Array.reshape([batchSize * self.inputNum, 3, self.imHeight, self.imWidth] )

            refractImg, reflectImg, mask = self.renderer.forward(origin, lookat, up, envmap, normal1Array, normal2Array )
            renderedImg = torch.clamp(refractImg + refractImg, 0, 1)

            cost = torch.norm(renderedImg - imBatch, dim=1 ).unsqueeze(1)
            normal1Best += normal1Array * torch.exp(-self.lamb * cost)
            normal2Best += normal2Array * torch.exp(-self.lamb * cost)

        normal1Best = normal1Best * seg1Batch
        normal2Best = normal2Best * seg1Batch

        normal1Best = normal1Best / torch.clamp(torch.sqrt(torch.sum(normal1Best * normal1Best, dim=1) ), min=1e-10 ).unsqueeze(1)
        normal2Best = normal2Best / torch.clamp(torch.sqrt(torch.sum(normal2Best * normal2Best, dim=1) ), min=1e-10 ).unsqueeze(1)

        refractImg, reflectImg, mask = self.renderer.forward(origin, lookat, up, envmap, normal1Best, normal2Best )
        renderedImg = torch.clamp(refractImg + refractImg, 0, 1)
        cost = torch.norm(renderedImg - imBatch, dim=1 ).unsqueeze(1)

        costVolume = torch.cat([cost, mask, normal1Best, normal2Best], dim=1 )

        return costVolume


class buildVisualHull():
    def __init__(self,fov = 63.4, volumeSize=32 ):
        minX, maxX = -1.1, 1.1
        minY, maxY = -1.1, 1.1
        minZ, maxZ = -1.1, 1.1

        y, x, z = np.meshgrid(
                np.linspace(minX, maxX, volumeSize ),
                np.linspace(minY, maxY, volumeSize ),
                np.linspace(minZ, maxZ, volumeSize ) )
        x = x[:, :, :, np.newaxis ]
        y = y[:, :, :, np.newaxis ]
        z = z[:, :, :, np.newaxis ]
        coord = np.concatenate([x, y, z], axis=3 ).astype(np.float32 )
        coord = coord[np.newaxis, :]
        self.coord = Variable(torch.from_numpy(coord ) )

        self.volumeSize = volumeSize
        self.fov = fov / 180.0 * np.pi


    def forward(self, origin, lookat, up, error, mask ):
        batchSize = origin.size(0 )
        imHeight = error.size(2)
        imWidth = error.size(3)
        pixelSize = 0.5 * imWidth / np.tan(self.fov / 2.0 )

        offset = np.arange(0, 10 ) * (imHeight * imWidth )
        offset = offset.reshape(10, 1, 1, 1).astype(np.float32 )
        offset = Variable(torch.from_numpy(offset ) ).cuda()

        yAxis = up
        zAxis = lookat - origin
        zAxis = zAxis / torch.sqrt(torch.sum(zAxis * zAxis, dim=1 ).unsqueeze(1) )
        xAxis = torch.cross(zAxis, yAxis, dim=1 )
        xAxis = xAxis / torch.sqrt(torch.sum(xAxis * xAxis, dim=1 ).unsqueeze(1) )

        origin = origin.reshape([batchSize, 1, 1, 1, 3] )
        xAxis = xAxis.reshape([batchSize, 1, 1, 1, 3] )
        yAxis = yAxis.reshape([batchSize, 1, 1, 1, 3] )
        zAxis = zAxis.reshape([batchSize, 1, 1, 1, 3] )

        coordCam = self.coord.cuda() - origin
        weight = torch.sqrt(torch.sum(coordCam * coordCam, dim=4) )
        weight = torch.exp(-weight )

        xCam = torch.sum(coordCam * xAxis, dim=4 )
        yCam = torch.sum(coordCam * yAxis, dim=4 )
        zCam = torch.sum(coordCam * zAxis, dim=4 )

        xCam = xCam / torch.clamp(zCam, min=1e-6 )
        yCam = yCam / torch.clamp(zCam, min=1e-6 )

        xId = xCam * pixelSize + imWidth / 2.0
        yId = -yCam * pixelSize + imHeight  / 2.0

        xId = torch.clamp(xId, 0, imWidth-1 )
        yId = torch.clamp(yId, 0, imHeight-1 )

        xId_d, xId_u = xId.floor(), xId.ceil()
        yId_d, yId_u = yId.floor(), yId.ceil()
        ind_dd = yId_d * imWidth + xId_d + offset
        ind_du = yId_d * imWidth + xId_u + offset
        ind_ud = yId_u * imWidth + xId_d + offset
        ind_uu = yId_u * imWidth + xId_u + offset

        wx_d = (xId - xId_d )
        wx_u = 1 - wx_d
        wy_d = (yId - yId_d )
        wy_u = 1 - wy_d

        error = error.reshape(-1)
        mask = mask.reshape(-1)

        volumeError_dd = torch.index_select(error, 0, ind_dd.reshape(-1).long() ).reshape(wx_d.size() )
        volumeError_du = torch.index_select(error, 0, ind_du.reshape(-1).long() ).reshape(wx_d.size() )
        volumeError_ud = torch.index_select(error, 0, ind_ud.reshape(-1).long() ).reshape(wx_d.size() )
        volumeError_uu = torch.index_select(error, 0, ind_uu.reshape(-1).long() ).reshape(wx_d.size() )
        volumeError = volumeError_dd * wy_d * wx_d \
                + volumeError_du * wy_d * wx_u \
                + volumeError_ud * wy_u * wx_d \
                + volumeError_uu * wy_u * wx_u
        volumeError = torch.sum(volumeError * weight, dim=0).unsqueeze(0).unsqueeze(0)

        volumeMask_dd = torch.index_select(mask, 0, ind_dd.reshape(-1).long() ).reshape(wx_d.size() )
        volumeMask_du = torch.index_select(mask, 0, ind_du.reshape(-1).long() ).reshape(wx_d.size() )
        volumeMask_ud = torch.index_select(mask, 0, ind_ud.reshape(-1).long() ).reshape(wx_d.size() )
        volumeMask_uu = torch.index_select(mask, 0, ind_uu.reshape(-1).long() ).reshape(wx_d.size() )
        volumeMask = volumeMask_dd * wy_d * wx_d \
                + volumeMask_du * wy_d * wx_u \
                + volumeMask_ud * wy_u * wx_d \
                + volumeMask_uu * wy_u * wx_u
        volumeMask = torch.prod(volumeMask, dim=0 ).unsqueeze(0).unsqueeze(0)

        volume = volumeMask * volumeError
        return volume
