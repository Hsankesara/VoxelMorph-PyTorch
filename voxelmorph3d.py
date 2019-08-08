import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
use_gpu = torch.cuda.is_available()

class UNet(nn.Module):
    def contracting_block(self, in_channels, out_channels, kernel_size=3):
        """
        This function creates one contracting block
        """
        block = torch.nn.Sequential(
            torch.nn.Conv3d(kernel_size=kernel_size, in_channels=in_channels, out_channels=out_channels, padding=1),
            torch.nn.BatchNorm3d(out_channels),
            torch.nn.ReLU(),
            torch.nn.Conv3d(kernel_size=kernel_size, in_channels=out_channels, out_channels=out_channels, padding=1),
            torch.nn.BatchNorm3d(out_channels),
            torch.nn.ReLU(),
        )
        return block

    def expansive_block(self, in_channels, mid_channel, out_channels, kernel_size=3):
        """
        This function creates one expansive block
        """
        block = torch.nn.Sequential(
            torch.nn.Conv3d(kernel_size=kernel_size, in_channels=in_channels, out_channels=mid_channel, padding=1),
            torch.nn.BatchNorm3d(mid_channel),
            torch.nn.ReLU(),
            torch.nn.Conv3d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=mid_channel, padding=1),
            torch.nn.BatchNorm3d(mid_channel),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose3d(in_channels=mid_channel, out_channels=out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            torch.nn.BatchNorm3d(out_channels),
            torch.nn.ReLU(),
        )
        return block

    def final_block(self, in_channels, mid_channel, out_channels, kernel_size=3):
        """
        This returns final block
        """
        block = torch.nn.Sequential(
                    torch.nn.Conv3d(kernel_size=kernel_size, in_channels=in_channels, out_channels=mid_channel, padding=1),
                    torch.nn.BatchNorm3d(mid_channel),
                    torch.nn.ReLU(),
                    torch.nn.Conv3d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=out_channels, padding=1),
                    torch.nn.BatchNorm3d(out_channels),
                    torch.nn.ReLU()
                )
        return block

    def __init__(self, in_channel, out_channel):
        super(UNet, self).__init__()
        #Encode
        self.conv_encode1 = self.contracting_block(in_channels=in_channel, out_channels=32)
        self.conv_maxpool1 = torch.nn.MaxPool3d(kernel_size=2)
        self.conv_encode2 = self.contracting_block(32, 64)
        self.conv_maxpool2 = torch.nn.MaxPool3d(kernel_size=2)
        self.conv_encode3 = self.contracting_block(64, 128)
        self.conv_maxpool3 = torch.nn.MaxPool3d(kernel_size=2)
        # Bottleneck
        mid_channel = 128
        self.bottleneck = torch.nn.Sequential(
                                torch.nn.Conv3d(kernel_size=3, in_channels=mid_channel, out_channels=mid_channel * 2, padding=1),
                                torch.nn.BatchNorm3d(mid_channel * 2),
                                torch.nn.ReLU(),
                                torch.nn.Conv3d(kernel_size=3, in_channels=mid_channel*2, out_channels=mid_channel, padding=1),
                                torch.nn.BatchNorm3d(mid_channel),
                                torch.nn.ReLU(),
                                torch.nn.ConvTranspose3d(in_channels=mid_channel, out_channels=mid_channel, kernel_size=3, stride=2, padding=1, output_padding=1),
                                torch.nn.BatchNorm3d(mid_channel),
                                torch.nn.ReLU(),
                            )
        # Decode
        self.conv_decode3 = self.expansive_block(256, 128, 64)
        self.conv_decode2 = self.expansive_block(128, 64, 32)
        self.final_layer = self.final_block(64, 32, out_channel)

    def crop_and_concat(self, upsampled, bypass, crop=False):
        """
        This layer crop the layer from contraction block and concat it with expansive block vector
        """
        if crop:
            c = (bypass.size()[2] - upsampled.size()[2]) // 2
            bypass = F.pad(bypass, (-c, -c, -c, -c))
        return torch.cat((upsampled, bypass), 1)

    def forward(self, x):
        # Encode
        encode_block1 = self.conv_encode1(x)
        encode_pool1 = self.conv_maxpool1(encode_block1)
        encode_block2 = self.conv_encode2(encode_pool1)
        encode_pool2 = self.conv_maxpool2(encode_block2)
        encode_block3 = self.conv_encode3(encode_pool2)
        encode_pool3 = self.conv_maxpool3(encode_block3)
        # Bottleneck
        bottleneck1 = self.bottleneck(encode_pool3)
        # Decode
        decode_block3 = self.crop_and_concat(bottleneck1, encode_block3)
        cat_layer2 = self.conv_decode3(decode_block3)
        decode_block2 = self.crop_and_concat(cat_layer2, encode_block2)
        cat_layer1 = self.conv_decode2(decode_block2)
        decode_block1 = self.crop_and_concat(cat_layer1, encode_block1)
        final_layer = self.final_layer(decode_block1)
        return  final_layer

class SpatialTransformation(nn.Module):
    def __init__(self):
        super(SpatialTransformation, self).__init__()

    def meshgrid(self, height, width, depth):
        x_t = torch.matmul(torch.ones([height, 1]), torch.transpose(torch.unsqueeze(torch.linspace(0.0, width -1.0, width), 1), 1, 0))
        y_t = torch.matmul(torch.unsqueeze(torch.linspace(0.0, height - 1.0, height), 1), torch.ones([1, width]))

        x_t = x_t.expand([height, width, depth])
        y_t = y_t.expand([height, width, depth])


        z_t = torch.linspace(0.0, depth -1.0, depth)
        z_t = torch.unsqueeze(torch.unsqueeze(z_t, 0), 0)
        z_t = z_t.expand([height, width, depth])

        return x_t, y_t, z_t

    def repeat(self, x, n_repeats):
        rep = torch.transpose(torch.unsqueeze(torch.ones(n_repeats), 1), 1, 0)
        rep = rep.long()
        print(rep.shape)
        print(torch.reshape(x, (-1, 1)).shape)
        x = torch.matmul(torch.reshape(x, (-1, 1)), rep)
        return torch.squeeze(torch.reshape(x, (-1, 1)))


    def interpolate(self, im, x, y, z):

        im = F.pad(im, (1,1,1,1,1,1,0,0))

        batch_size, height, width, depth = im.shape

        batch_size, out_height, out_width, out_depth = x.shape

        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        z = z.reshape(1, -1)

        x = x + 1
        y = y + 1
        z = z + 1

        max_x = width - 1
        max_y = height - 1
        max_z = depth - 1

        x0 = torch.floor(x).long()
        x1 = x0 + 1
        y0 = torch.floor(y).long()
        y1 = y0 + 1
        z0 = torch.floor(z).long()
        z1 = z0 + 1

        x0 = torch.clamp(x0, 0, max_x)
        x1 = torch.clamp(x1, 0, max_x)
        y0 = torch.clamp(y0, 0, max_y)
        y1 = torch.clamp(y1, 0, max_y)
        z0 = torch.clamp(z0, 0, max_z)
        z1 = torch.clamp(z1, 0, max_z)

        dim3 = depth
        dim2 = depth*width
        dim1 = depth*width*height
        base = self.repeat(torch.arange(0, batch_size)*dim1, out_height*out_width*out_depth)

        base_y0 = base + y0*dim2
        base_y1 = base + y1*dim2

        idx_a = base_y0 + x0*dim3 + z0
        idx_b = base_y1 + x0*dim3 + z0
        idx_c = base_y0 + x1*dim3 + z0
        idx_d = base_y1 + x1*dim3 + z0
        idx_e = base_y0 + x0*dim3 + z1
        idx_f = base_y1 + x0*dim3 + z1
        idx_g = base_y0 + x1*dim3 + z1
        idx_h = base_y1 + x1*dim3 + z1

        # use indices to lookup pixels in the flat image and restore
        # channels dim
        im_flat = torch.reshape(im, [-1, 1])
        im_flat = im_flat.float()

        Ia = torch.gather(im_flat, 0, idx_a.transpose(1,0))
        Ib = torch.gather(im_flat, 0, idx_b.transpose(1,0))
        Ic = torch.gather(im_flat, 0, idx_c.transpose(1,0))
        Id = torch.gather(im_flat, 0, idx_d.transpose(1,0))
        Ie = torch.gather(im_flat, 0, idx_e.transpose(1,0))
        If = torch.gather(im_flat, 0, idx_f.transpose(1,0))
        Ig = torch.gather(im_flat, 0, idx_g.transpose(1,0))
        Ih = torch.gather(im_flat, 0, idx_h.transpose(1,0))

        # and finally calculate interpolated values
        x1_f = x1.float()
        y1_f = y1.float()
        z1_f = z1.float()

        dx = x1_f - x
        dy = y1_f - y
        dz = z1_f - z

        wa = (dz * dx * dy).transpose(1,0)
        wb = (dz * dx * (1-dy)).transpose(1,0)
        wc = (dz * (1-dx) * dy).transpose(1,0)
        wd = (dz * (1-dx) * (1-dy)).transpose(1,0)
        we = ((1-dz) * dx * dy).transpose(1,0)
        wf = ((1-dz) * dx * (1-dy)).transpose(1,0)
        wg = ((1-dz) * (1-dx) * dy).transpose(1,0)
        wh = ((1-dz) * (1-dx) * (1-dy)).transpose(1,0)

        print(wa.shape, Ia.shape)
        print(torch.squeeze(torch.stack([wa*Ia, wb*Ib, wc*Ic, wd*Id, we*Ie, wf*If, wg*Ig, wh*Ih], dim=1)).shape)
        print(torch.sum(torch.squeeze(torch.stack([wa*Ia, wb*Ib, wc*Ic, wd*Id, we*Ie, wf*If, wg*Ig, wh*Ih], dim=1)), 1).shape)
        output = torch.sum(torch.squeeze(torch.stack([wa*Ia, wb*Ib, wc*Ic, wd*Id, we*Ie, wf*If, wg*Ig, wh*Ih], dim=1)), 1)
        print(output.shape, Ia.shape)
        output = torch.reshape(output, [-1, out_height, out_width, out_depth])
        return output

    def forward(self, moving_image, deformation_matrix):
        dx = deformation_matrix[:, :, :, :, 0]
        dy = deformation_matrix[:, :, :, :, 1]
        dz = deformation_matrix[:, :, :, :, 2]

        batch_size, height, width, depth = dx.shape

        x_mesh, y_mesh, z_mesh = self.meshgrid(height, width, depth)

        x_mesh = x_mesh.expand([batch_size, height, width, depth])
        y_mesh = y_mesh.expand([batch_size, height, width, depth])
        z_mesh = z_mesh.expand([batch_size, height, width, depth])
        x_new = dx + x_mesh
        y_new = dy + y_mesh
        z_new = dz + z_mesh

        return self.interpolate(moving_image, x_new, y_new, z_new)


class VoxelMorph(nn.Module):
    def __init__(self, use_gpu=False):
        super(VoxelMorph, self).__init__()
        self.unet = UNet(2, 3)
        if use_gpu:
            self.unet = self.unet.cuda()
        pass

    def forward(self, moving_image, fixed_image):
        x = torch.stack([moving_image, fixed_image], dim=1)
        self.deformation_matrix = self.unet(x)

        registered_image = self.spatial_transform(moving_image, self.deformation_matrix)
        return registered_image

    def spatial_transform(self, moving_image, deformation_matrix):
        #print(moving_image.size(), deformation_matrix.size())
        registered_image = torch.zeros_like(moving_image)
        print("Registered Image Before", registered_image.is_cuda)
        m,n,_,__ = moving_image.shape
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    pdash = (moving_image[:, i, j, k] +  deformation_matrix[:, :, i,  j, k].permute(1,0)).permute(1,0)
                    moving_neighbours = moving_image[:, max(0, i-1) : i+2, max(0,  j-1) : j+2, max(0, k-1) : k+2]
                    moving_deformation_matrices  = deformation_matrix[:, :, max(0, i-1) : i+2, max(0,  j-1) : j+2, max(0, k-1) : k+2]
                    m, x, y, z = moving_neighbours.shape
                    registered_image[:,  i, j, k] = torch.sum(torch.mul(moving_neighbours, torch.prod(1- torch.abs(pdash.expand(x,y,z,m, 3).permute(3, 4,0, 1, 2) -
                                                                                                                   moving_deformation_matrices), dim=1)), dim=[1,2,3])
        print("Registered After Before", registered_image.is_cuda)
        return registered_image


def get_neighbours(y, i, j, k, n):
    return y[:, max(0, i - n // 2 ) : i + n // 2 + 1, max(0,  j - n // 2) : j + n // 2 + 1, max(0, k - n // 2) : k + n // 2 + 1]

def removed_local_mean_intensity(fp, fn, n):
    #rint(fn)
    # print('Local Mean Intensity', torch.sum(fn, dim=[1,2,3]))
    return fp - (torch.sum(fn, dim=[1,2,3]) / (n**3))

def cross_correlation_loss(y, ytrue, n):
    ccloss = 0
    m,size,_,__ = y.shape
    for i in range(size):
        for j in range(size):
            for k in range(size):
                y_neighbours = get_neighbours(y, i, j, k, n)
                ytrue_neighbours = get_neighbours(ytrue, i, j, k, n)
                gdash = removed_local_mean_intensity(y[:,i,j,k], y_neighbours, n)
                fdash = removed_local_mean_intensity(ytrue[:,i,j,k], ytrue_neighbours, n)
                numerator = torch.pow(torch.sum(torch.mul(ytrue_neighbours - fdash, y_neighbours - gdash), dim=[1,2,3]), 2)
                denom = torch.sum(torch.pow(ytrue_neighbours - fdash, 2), dim=[1,2,3]) * torch.sum(torch.pow(y_neighbours - gdash, 2), dim=[1,2,3])
                if denom != 0:
                    ccloss += torch.sum((numerator / denom))
                else:
                    ccloss += 0
    return ccloss

def smooothing_loss(y):
    m,size,_,__ = y.shape
    grad = torch.zeros((3, m, size, size, size))
    if use_gpu:
        grad = grad.cuda()
    print("Grad Before", grad.is_cuda)
    grad[0, :, 0 : size - 1, :, :] = y[:, 1 : size, :, :] - y[:, 0 : size - 1, :, :]
    grad[1, :, :, 0 : size - 1, :] = y[:, :, 1 : size,:] - y[:, :, 0 : size - 1, :]
    grad[2, :, :, :, 0 : size - 1] = y[:, :, :, 1 : size] - y[:, :, :, 0 : size - 1]
    grad = grad.permute(1,0,2,3,4)
    print("Grad After", grad.is_cuda)
    loss_each_image = torch.sum(torch.norm(grad, p=2, dim=1), dim=[1,2,3])
    return torch.sum(loss_each_image)

def vox_morph_loss(y, ytrue, n=9, lamda=0.2):
    loss = - cross_correlation_loss(y, ytrue, n) + lamda * smooothing_loss(y)
    return loss
