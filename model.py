import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from module import *


class Generator32(nn.Module):
    r"""
    ResNet backbone generator for SNGAN.
    Attributes:
        nz (int): Noise dimension for upsampling.
        ngf (int): Variable controlling generator feature map sizes.
        bottom_width (int): Starting width for upsampling generator output to an image.
    """

    def __init__(self, nz=128, ngf=256, bottom_width=4):
        super().__init__()

        self.l1 = nn.Linear(nz, (bottom_width ** 2) * ngf)
        self.unfatten = nn.Unflatten(1, (ngf, bottom_width, bottom_width))
        self.block2 = GBlock(ngf, ngf, upsample=True)
        self.block3 = GBlock(ngf, ngf, upsample=True)
        self.block4 = GBlock(ngf, ngf, upsample=True)
        self.b5 = nn.BatchNorm2d(ngf)
        self.c5 = nn.Conv2d(ngf, 3, 3, 1, padding=1)
        self.activation = nn.ReLU(True)

        nn.init.xavier_uniform_(self.l1.weight.data, 1.0)
        nn.init.xavier_uniform_(self.c5.weight.data, 1.0)

    def forward(self, x):
        h = self.l1(x)
        h = self.unfatten(h)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.b5(h)
        h = self.activation(h)
        h = self.c5(h)
        y = torch.tanh(h)
        return y


class Discriminator32(nn.Module):
    r"""
    ResNet backbone discriminator for SNGAN.
    Attributes:
        ndf (int): Variable controlling discriminator feature map sizes.
    """

    def __init__(self, ndf=128):
        super().__init__()

        self.block1 = DBlockOptimized(3, ndf)
        self.block2 = DBlock(ndf, ndf, downsample=True)
        self.block3 = DBlock(ndf, ndf, downsample=False)
        self.block4 = DBlock(ndf, ndf, downsample=False)
        self.l5 = SNLinear(ndf, 1)
        self.activation = nn.ReLU(True)

        nn.init.xavier_uniform_(self.l5.weight.data, 1.0)

    def forward(self, x):
        h = x
        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.activation(h)
        h = torch.sum(h, dim=(2, 3))
        y = self.l5(h)
        return y


class Generator64(nn.Module):
    r"""
    ResNet backbone generator for SNGAN.
    Attributes:
        nz (int): Noise dimension for upsampling.
        ngf (int): Variable controlling generator feature map sizes.
        bottom_width (int): Starting width for upsampling generator output to an image.
    """

    def __init__(self, nz=128, ngf=1024, bottom_width=4):
        super().__init__()

        self.l1 = nn.Linear(nz, (bottom_width ** 2) * ngf)
        self.unfatten = nn.Unflatten(1, (ngf, bottom_width, bottom_width))
        self.block2 = GBlock(ngf, ngf >> 1, upsample=True)
        self.block3 = GBlock(ngf >> 1, ngf >> 2, upsample=True)
        self.block4 = GBlock(ngf >> 2, ngf >> 3, upsample=True)
        self.block5 = GBlock(ngf >> 3, ngf >> 4, upsample=True)
        self.b6 = nn.BatchNorm2d(ngf >> 4)
        self.c6 = nn.Conv2d(ngf >> 4, 3, 3, 1, padding=1)
        self.activation = nn.ReLU(True)

        nn.init.xavier_uniform_(self.l1.weight.data, 1.0)
        nn.init.xavier_uniform_(self.c6.weight.data, 1.0)

    def forward(self, x):
        h = self.l1(x)
        h = self.unfatten(h)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.block5(h)
        h = self.b6(h)
        h = self.activation(h)
        h = self.c6(h)
        y = torch.tanh(h)
        return y


class Discriminator64(nn.Module):
    r"""
    ResNet backbone discriminator for SNGAN.
    Attributes:
        ndf (int): Variable controlling discriminator feature map sizes.
    """

    def __init__(self, ndf=1024):
        super().__init__()

        self.block1 = DBlockOptimized(3, ndf >> 4)
        self.block2 = DBlock(ndf >> 4, ndf >> 3, downsample=True)
        self.block3 = DBlock(ndf >> 3, ndf >> 2, downsample=True)
        self.block4 = DBlock(ndf >> 2, ndf >> 1, downsample=True)
        self.block5 = DBlock(ndf >> 1, ndf, downsample=True)
        self.l6 = SNLinear(ndf, 1)
        self.activation = nn.ReLU(True)

        nn.init.xavier_uniform_(self.l6.weight.data, 1.0)

    def forward(self, x):
        h = x
        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.block5(h)
        h = self.activation(h)
        h = torch.sum(h, dim=(2, 3))
        y = self.l6(h)
        return y

# class cGenerator64_old(nn.Module):
#     def __init__(self):
#         super(cGenerator64_old, self).__init__()
        
#         self.enc_layer1=ConvBlock(4,8)
#         self.enc_layer2=ConvBlock(8,16)
#         self.enc_layer3=ConvBlock(16,32)
#         self.enc_layer4=ConvBlock(32,64)
#         self.enc_layer5=ConvBlock(64,128)
#         self.bottleneck=ConvBlock(128,256,kernel_size=4,stride=1,padding=0)
    
#         self.dec_layer1=ConvTranBlock(256,128,kernel_size=4,stride=1,padding=0)
#         self.dec_layer2=ConvTranBlock(256,64)
#         self.dec_layer3=ConvTranBlock(128,32)
#         self.dec_layer4=ConvTranBlock(64,16)
#         self.dec_layer5=ConvTranBlock(32,8)
#         self.dec_layer6=ConvTranBlock(16,3)
#         self.dec_layer7=nn.ConvTranspose2d(6,3,kernel_size=1,stride=1,padding=0)

#     def forward(self,x,z):
#         z=z.view(-1,1,128,128)
#         x=x.view(-1,3,128,128)
#         x_noisy=torch.cat([z,x],1) # (-1,4,128,128)
#         enc1=self.enc_layer1(x_noisy) # (-1,8,64,64)
#         enc2=self.enc_layer2(enc1) # (-1,16,32,32)
#         enc3=self.enc_layer3(enc2)  # (-1,32,16,16)
#         enc4=self.enc_layer4(enc3)  # (-1,64,8,8)
#         enc5=self.enc_layer5(enc4) # (-1,128,4,4)
#         latent=self.bottleneck(enc5) # (-1,256,1,1)
#         dec1=torch.cat([self.dec_layer1(latent),enc5],1)
#         dec2=torch.cat([self.dec_layer2(dec1),enc4],1)
#         dec3=torch.cat([self.dec_layer3(dec2),enc3],1)
#         dec4=torch.cat([self.dec_layer4(dec3),enc2],1)
#         dec5=torch.cat([self.dec_layer5(dec4),enc1],1)
#         dec6=torch.cat([self.dec_layer6(dec5),x],1)
#         output=self.dec_layer7(dec6)
    
#         return output
# class cDiscriminator64_old(nn.Module):
#     # x is the sketch
#     # y is the colored on (real or fake)
#     def __init__(self):
#         super(cDiscriminator64_old, self).__init__()
#         self.model=nn.Sequential(
#             ConvBlock(6,8),
#             ConvBlock(8,16),
#             ConvBlock(16,32),
#             ConvBlock(32,64),
#             ConvBlock(64,128),
#             nn.Conv2d(128,1,kernel_size=4,stride=1,padding=0),
#             nn.Sigmoid()
#         )
#     def forward(self,x,y):
#         x=x.view(-1,3,128,128)
#         y=y.view(-1,3,128,128)
#         concat=torch.cat([x,y],1)
#         out=self.model(concat)
#         label=out.view(-1,1)
#         return label #real/fake    
    
class cGenerator64(nn.Module):
    def __init__(self):
        super(cGenerator64, self).__init__()
        
        self.enc_layer1=ConvBlock(3, 8)
        self.enc_layer2=ConvBlock(8, 16)
        self.enc_layer3=ConvBlock(16, 32)
        self.enc_layer4=ConvBlock(32, 64)
        # self.enc_layer5=ConvBlock(64, 128)
        self.bottleneck=ConvBlock(64, 128, kernel_size=4, stride=1, padding=0)
    
        self.dec_layer1=ConvTranBlock(128, 64, kernel_size=4,stride=1,padding=0)
        self.dec_layer2=ConvTranBlock(128, 32)
        self.dec_layer3=ConvTranBlock(64, 16)
        self.dec_layer4=ConvTranBlock(32, 8)
        self.dec_layer5=ConvTranBlock(16, 3)
        # self.dec_layer6=ConvTranBlock(12, 3)
        self.dec_layer7=nn.ConvTranspose2d(6,3,kernel_size=1,stride=1,padding=0)

    def forward(self, x):
        # x (-1, 3, 64, 64)
        x = x.view(-1, 3, 64, 64)
        enc1 = self.enc_layer1(x) # (-1,8,32,32)
        enc2 = self.enc_layer2(enc1) # (-1,16,16,16)
        enc3 = self.enc_layer3(enc2)  # (-1,32,8,8)
        enc4 = self.enc_layer4(enc3)  # (-1,64,4,4)
        # enc5 = self.enc_layer5(enc4) #  (-1,128,2,2)
        latent = self.bottleneck(enc4) # (-1,128,1,1)
        dec1 = torch.cat([self.dec_layer1(latent), enc4], 1)  # (-1,128,4,4)
        dec2 = torch.cat([self.dec_layer2(dec1), enc3], 1)    # (-1,64,8,8)
        dec3 = torch.cat([self.dec_layer3(dec2), enc2], 1)    # (-1,32,16,16)
        dec4 = torch.cat([self.dec_layer4(dec3), enc1], 1)    # (-1,16,32,32)
        dec5 = torch.cat([self.dec_layer5(dec4), x], 1)    # (-1,6,64,64)
        # dec6 = torch.cat([self.dec_layer6(dec5), x], 1)      # (-1,3,64,64)
        out = self.dec_layer7(dec5) # (-1,3,64,64)
    
        return out
    
class cGenerator64_z(nn.Module):
    def __init__(self):
        super(cGenerator64, self).__init__()
        
        self.enc_layer1=ConvBlock(4, 8)
        self.enc_layer2=ConvBlock(8, 16)
        self.enc_layer3=ConvBlock(16, 32)
        self.enc_layer4=ConvBlock(32, 64)
        # self.enc_layer5=ConvBlock(64, 128)
        self.bottleneck=ConvBlock(64, 128, kernel_size=4, stride=1, padding=0)
    
        self.dec_layer1=ConvTranBlock(128, 64, kernel_size=4,stride=1,padding=0)
        self.dec_layer2=ConvTranBlock(128, 32)
        self.dec_layer3=ConvTranBlock(64, 16)
        self.dec_layer4=ConvTranBlock(32, 8)
        self.dec_layer5=ConvTranBlock(16, 3)
        # self.dec_layer6=ConvTranBlock(12, 3)
        self.dec_layer7=nn.ConvTranspose2d(6,3,kernel_size=1,stride=1,padding=0)

    def forward(self, x, z):
        # x (-1, 3, 64, 64)
        x = x.view(-1, 3, 64, 64)
        z = z.view(-1, 1, 64, 64)
        x_cat=torch.cat([z, x],1)  # (-1, 4, 64, 64)
        enc1 = self.enc_layer1(x) # (-1,8,32,32)
        enc2 = self.enc_layer2(enc1) # (-1,16,16,16)
        enc3 = self.enc_layer3(enc2)  # (-1,32,8,8)
        enc4 = self.enc_layer4(enc3)  # (-1,64,4,4)
        # enc5 = self.enc_layer5(enc4) #  (-1,128,2,2)
        latent = self.bottleneck(enc4) # (-1,128,1,1)
        dec1 = torch.cat([self.dec_layer1(latent), enc4], 1)  # (-1,128,4,4)
        dec2 = torch.cat([self.dec_layer2(dec1), enc3], 1)    # (-1,64,8,8)
        dec3 = torch.cat([self.dec_layer3(dec2), enc2], 1)    # (-1,32,16,16)
        dec4 = torch.cat([self.dec_layer4(dec3), enc1], 1)    # (-1,16,32,32)
        dec5 = torch.cat([self.dec_layer5(dec4), x], 1)    # (-1,6,64,64)
        # dec6 = torch.cat([self.dec_layer6(dec5), x], 1)      # (-1,3,64,64)
        out = self.dec_layer7(dec5) # (-1,3,64,64)
    
        return out

class cDiscriminator64(nn.Module):
    # x is the sketch
    # y is the colored on (real or fake)
    def __init__(self):
        super(cDiscriminator64, self).__init__()
        self.model=nn.Sequential( #(-1, 6, 64, 64)
            ConvBlock(6,8), #(-1, 12, 32, 32)
            ConvBlock(8,16), #(-1, 24, 16, 16)
            ConvBlock(16,32), #(-1, 48, 8, 8)
            ConvBlock(32,64),  #(-1, 96, 4, 4)
            # ConvBlock(96,192), #(-1, 192, 2, 2)
            nn.Conv2d(64,1,kernel_size=4,stride=1,padding=0),#(-1, 1, 1,1)
            nn.Sigmoid(),
        )
    def forward(self,x,y):
        x=x.view(-1, 3, 64, 64)
        y=y.view(-1, 3, 64, 64)
        concat=torch.cat([x,y], 1) #(-1, 6, 64, 64)
        out=self.model(concat) # 
        label=out.view(-1,1)
        
        return label #real/fake

if __name__ == "__main__":
    cgenerator = cGenerator64()
    
    input_x = torch.zeros((60, 3, 64, 64))
    out = cgenerator(input_x) 
    print(out.shape)
    
    cdiscriminator = cDiscriminator64()
    
    sketch = torch.zeros((60, 3, 64, 64))
    colored = torch.zeros((60, 3, 64, 64))
    out = cdiscriminator(sketch, colored) 
    print(out.shape)
    
