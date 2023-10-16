import torch
import torch.nn as nn 
import torchvision.models as models 
from torch import nn, einsum
from einops import rearrange, repeat
from torch.nn import functional as F

class SPSNet(nn.Module):

    def __init__(self, n_class):
        super(SPSNet, self).__init__()

        self.num_resnet_layers = 152

        if self.num_resnet_layers == 18:
            resnet_raw_model2 = models.resnet18(pretrained=True)
            self.inplanes = 512
        elif self.num_resnet_layers == 34:
            resnet_raw_model2 = models.resnet34(pretrained=True)
            self.inplanes = 512
        elif self.num_resnet_layers == 50:
            resnet_raw_model2 = models.resnet50(pretrained=True)
            self.inplanes = 2048
        elif self.num_resnet_layers == 101:
            resnet_raw_model2 = models.resnet101(pretrained=True)
            self.inplanes = 2048
        elif self.num_resnet_layers == 152:
            resnet_raw_model2 = models.resnet152(pretrained=True)
            self.inplanes = 2048

        ######## ENCODER  ########
 
        self.encoder_conv1 = resnet_raw_model2.conv1
        self.encoder_bn1 = resnet_raw_model2.bn1
        self.encoder_relu = resnet_raw_model2.relu
        self.encoder_maxpool = resnet_raw_model2.maxpool
        self.encoder_layer1 = resnet_raw_model2.layer1
        self.encoder_layer2 = resnet_raw_model2.layer2
        self.encoder_layer3 = resnet_raw_model2.layer3
        self.encoder_layer4 = BottleStack(dim=1024,fmap_size=(30,40),dim_out=2048,proj_factor = 4,num_layers=3,heads=4,dim_head=512)

        ##################   Decoder   #########################
        self.decoder5 = Decoder5(in_channel=2048,out_channel=1024)
        self.decoder4 = Decoder4(in_channel=2048,out_channel=512)
        self.decoder3 = Decoder3(in_channel=512*3,out_channel=256)
        self.decoder2 = Decoder2(in_channel=256*4,out_channel=128)
        self.decoder1 = Decoder1(in_channel=576,out_channel=n_class)

 
    def forward(self, rgb):

        verbose = False

        # encoder

        ######################################################################

        if verbose: print("rgb.size() original: ", rgb.size())  
        ######################################################################

        rgb = self.encoder_conv1(rgb)
        if verbose: print("rgb.size() after conv1: ", rgb.size()) 
        rgb = self.encoder_bn1(rgb)
        if verbose: print("rgb.size() after bn1: ", rgb.size())  
        rgb = self.encoder_relu(rgb)
        if verbose: print("rgb.size() after relu: ", rgb.size())  

        decoder_input_1 = rgb
        ######################################################################
        rgb = self.encoder_maxpool(rgb)
        if verbose: print("rgb.size() after maxpool: ", rgb.size()) 
        rgb = self.encoder_layer1(rgb)
        if verbose: print("rgb.size() after layer1: ", rgb.size()) 
        decoder_input_2 = rgb
        ######################################################################
        rgb = self.encoder_layer2(rgb)
        if verbose: print("rgb.size() after layer2: ", rgb.size()) 
        decoder_input_3 = rgb
        ######################################################################

        rgb = self.encoder_layer3(rgb)
        if verbose: print("rgb.size() after layer3: ", rgb.size()) # (30, 40)
        decoder_input_4 = rgb
        ######################################################################
        rgb = self.encoder_layer4(rgb)
        if verbose: print("rgb.size() after layer4: ", rgb.size()) # (15, 20)
        decoder_input_5  = rgb


        #skip
        decoder_output_54,decoder_output_53,decoder_output_52,decoder_output_51 = self.decoder5(decoder_input_5)
        if verbose: print("decoder_output_54.size() : ", decoder_output_54.size()) 
        if verbose: print("decoder_output_53.size() : ", decoder_output_53.size())   
        if verbose: print("decoder_output_52.size() : ", decoder_output_52.size())   
        if verbose: print("decoder_output_51.size() : ", decoder_output_51.size())   

        decoder_input_4 = torch.cat((decoder_input_4,decoder_output_54),dim=1)
        decoder_output_43,decoder_output_42,decoder_output_41 = self.decoder4(decoder_input_4)
        if verbose: print("decoder_output_43.size() : ", decoder_output_43.size())   
        if verbose: print("decoder_output_42.size() : ", decoder_output_42.size())   
        if verbose: print("decoder_output_41.size() : ", decoder_output_41.size())   

        decoder_input_3 = torch.cat((decoder_input_3,decoder_output_53,decoder_output_43),dim=1)
        decoder_output_32,decoder_output_31 = self.decoder3(decoder_input_3)
        if verbose: print("decoder_output_32.size() : ", decoder_output_32.size())   
        if verbose: print("decoder_output_31.size() : ", decoder_output_31.size())   


        decoder_input_2 = torch.cat((decoder_input_2,decoder_output_52,decoder_output_42,decoder_output_32),dim=1)
        decoder_output_21 = self.decoder2(decoder_input_2)
        if verbose: print("decoder_output_21.size() : ", decoder_output_21.size())   

        decoder_input_1 = torch.cat((decoder_input_1,decoder_output_51,decoder_output_41,decoder_output_31,decoder_output_21),dim=1)
        decoder_output_1 = self.decoder1(decoder_input_1)
        if verbose: print("decoder_output_21.size() : ", decoder_output_1.size())   


        return decoder_output_1
  
class Attention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        fmap_size,
        heads = 4,
        dim_head = 128,
        rel_pos_emb = False
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = heads * dim_head

        self.to_qkv = nn.Conv2d(dim, inner_dim * 3, 1, bias = False)
        #rel_pos_class = AbsPosEmb if not rel_pos_emb else RelPosEmb
        #rel_pos_class = AbsPosEmb
        #self.pos_emb = rel_pos_class(fmap_size, dim_head)
        self.pos_emb = AbsPosEmb(fmap_size, dim_head)
        

    def forward(self, fmap):
        heads, b, c, h, w = self.heads, *fmap.shape

        q, k, v = self.to_qkv(fmap).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h d) x y -> b h (x y) d', h = heads), (q, k, v))

        sim = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        sim += self.pos_emb(q)

        attn = sim.softmax(dim = -1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = h, y = w)

        return out

class AbsPosEmb(nn.Module):
    def __init__(
        self,
        fmap_size,
        dim_head
    ):
        super().__init__()
        scale = dim_head ** -0.5
        self.scale = scale
        self.height = nn.Parameter(torch.randn(fmap_size[0], dim_head) * scale)
        self.width = nn.Parameter(torch.randn(fmap_size[1], dim_head) * scale)

    def forward(self, q):
        emb = rearrange(self.height, 'h d -> h () d') + rearrange(self.width, 'w d -> () w d')
        emb = rearrange(emb, ' h w d -> (h w) d')
        logits = einsum('b h i d, j d -> b h i j', q, emb) * self.scale
        return logits

class BottleBlock(nn.Module):
    def __init__(
        self,
        *,
        dim,
        fmap_size,
        dim_out,
        proj_factor,
        downsample,
        heads = 4,
        dim_head = 128,
        rel_pos_emb = False,
        activation = nn.ReLU()
    ):
        super().__init__()

        # shortcut

        if dim != dim_out or downsample:  
            kernel_size, stride, padding = (3, 2, 1) if downsample else (1, 1, 0)

            self.shortcut = nn.Sequential(
                nn.Conv2d(dim, dim_out, kernel_size, stride = stride, padding = padding, bias = False),
                nn.BatchNorm2d(dim_out),
                activation
            )
        else:
            self.shortcut = nn.Identity()

        # contraction and expansion
        attention_dim = dim_out // proj_factor

        self.net = nn.Sequential(
            nn.Conv2d(dim, attention_dim, 1, bias = False),
            nn.BatchNorm2d(attention_dim),
            activation,
            Attention(
                dim = attention_dim,
                fmap_size = fmap_size,
                heads = heads,
                dim_head = dim_head,
                rel_pos_emb = rel_pos_emb
            ),
            nn.AvgPool2d((2, 2)) if downsample else nn.Identity(),
            nn.BatchNorm2d(heads*dim_head),
            activation,
            nn.Conv2d(heads*dim_head, dim_out, 1, bias = False),
            nn.BatchNorm2d(dim_out)
        )

        # init last batch norm gamma to zero
        nn.init.zeros_(self.net[-1].weight)

        # final activation
        self.activation = activation

    def forward(self, x):
        shortcut = self.shortcut(x)
        x = self.net(x)
        x += shortcut
        return self.activation(x)

# main bottle stack

class BottleStack(nn.Module):
    def __init__(
        self,
        *,
        dim,
        fmap_size,
        dim_out = 2048,
        proj_factor = 4,
        num_layers = 3,
        heads = 4,
        dim_head = 128,
        downsample = True,
        rel_pos_emb = False,
        activation = nn.ReLU()
    ):
        super().__init__()
        self.dim = dim
        self.fmap_size = fmap_size

        layers = []

        for i in range(num_layers):
            is_first = i == 0
            dim = (dim if is_first else dim_out)
            layer_downsample = is_first and downsample
            layer_fmap_size = (fmap_size[0] // (2 if downsample and not is_first else 1),fmap_size[1] // (2 if downsample and not is_first else 1))
            layers.append(BottleBlock(
                dim = dim,
                fmap_size = layer_fmap_size,
                dim_out = dim_out,
                proj_factor = proj_factor,
                heads = heads,
                dim_head = dim_head,
                downsample = layer_downsample,
                rel_pos_emb = rel_pos_emb,
                activation = activation
            ))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        _, c, h, w = x.shape
        assert c == self.dim, f'channels of feature map {c} must match channels given at init {self.dim}'
        assert h == self.fmap_size[0] and w == self.fmap_size[1], f'height and width of feature map must match the fmap_size given at init {self.fmap_size}'
        return self.net(x)


class Decoder5(nn.Module):
    def __init__(self, in_channel,out_channel):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channel,out_channels=out_channel,kernel_size=3,stride=1,padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=out_channel)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels=out_channel,out_channels=out_channel,kernel_size=3,stride=1,padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=out_channel)
        self.relu2 = nn.ReLU()

        self.Tconv1 = nn.ConvTranspose2d(in_channels=out_channel,out_channels=out_channel,kernel_size=2,stride=2)
        self.Tbn1 = nn.BatchNorm2d(num_features=out_channel)
        self.Trelu1 = nn.ReLU()

        self.Tconv2 = nn.ConvTranspose2d(in_channels=out_channel,out_channels=out_channel//2,kernel_size=4,stride=4)
        self.Tbn2 = nn.BatchNorm2d(num_features=out_channel//2)
        self.Trelu2 = nn.ReLU()

        self.Tconv3 = nn.ConvTranspose2d(in_channels=out_channel,out_channels=out_channel//4,kernel_size=8,stride=8)
        self.Tbn3 = nn.BatchNorm2d(num_features=out_channel//4)
        self.Trelu3 = nn.ReLU()

        self.Tconv4 = nn.ConvTranspose2d(in_channels=out_channel,out_channels=out_channel//8,kernel_size=16,stride=16)
        self.Tbn4 = nn.BatchNorm2d(num_features=out_channel//8)
        self.Trelu4 = nn.ReLU()


    def forward(self, x):

        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.relu1(x1)

        x2 = self.conv2(x1)
        x2 = self.bn2(x2)
        x2 = self.relu2(x2)

        x = x1+x2

        out1 = self.Tconv1(x)
        out1 = self.Tbn1(out1)
        out1 = self.Trelu1(out1)

        out2 = self.Tconv2(x)
        out2 = self.Tbn2(out2)
        out2 = self.Trelu2(out2)

        out3 = self.Tconv3(x)
        out3 = self.Tbn3(out3)
        out3 = self.Trelu3(out3)

        out4 = self.Tconv4(x)
        out4 = self.Tbn4(out4)
        out4 = self.Trelu4(out4)

        return out1,out2,out3,out4


class Decoder4(nn.Module):
    def __init__(self, in_channel,out_channel):
        super().__init__()

        self.transfer = nn.Conv2d(in_channels=in_channel,out_channels=in_channel//2,kernel_size=1)
        self.transfer_bn1 = nn.BatchNorm2d(num_features=in_channel//2)
        self.transfer_relu1 = nn.ReLU()

        self.conv1 = nn.Conv2d(in_channels=in_channel//2,out_channels=out_channel,kernel_size=3,stride=1,padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=out_channel)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels=out_channel,out_channels=out_channel,kernel_size=3,stride=1,padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=out_channel)
        self.relu2 = nn.ReLU()

        self.Tconv1 = nn.ConvTranspose2d(in_channels=out_channel,out_channels=out_channel,kernel_size=2,stride=2)
        self.Tbn1 = nn.BatchNorm2d(num_features=out_channel)
        self.Trelu1 = nn.ReLU()

        self.Tconv2 = nn.ConvTranspose2d(in_channels=out_channel,out_channels=out_channel//2,kernel_size=4,stride=4)
        self.Tbn2 = nn.BatchNorm2d(num_features=out_channel//2)
        self.Trelu2 = nn.ReLU()

        self.Tconv3 = nn.ConvTranspose2d(in_channels=out_channel,out_channels=out_channel//4,kernel_size=8,stride=8)
        self.Tbn3 = nn.BatchNorm2d(num_features=out_channel//4)
        self.Trelu3 = nn.ReLU()


    def forward(self, x):

        x = self.transfer(x)
        x = self.transfer_bn1(x)
        x = self.transfer_relu1(x)

        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.relu1(x1)

        x2 = self.conv2(x1)
        x2 = self.bn2(x2)
        x2 = self.relu2(x2)

        x = x1+x2

        out1 = self.Tconv1(x)
        out1 = self.Tbn1(out1)
        out1 = self.Trelu1(out1)

        out2 = self.Tconv2(x)
        out2 = self.Tbn2(out2)
        out2 = self.Trelu2(out2)

        out3 = self.Tconv3(x)
        out3 = self.Tbn3(out3)
        out3 = self.Trelu3(out3)

        return out1,out2,out3

class Decoder3(nn.Module):
    def __init__(self, in_channel,out_channel):
        super().__init__()

        self.transfer = nn.Conv2d(in_channels=in_channel,out_channels=in_channel//3,kernel_size=1)
        self.transfer_bn1 = nn.BatchNorm2d(num_features=in_channel//3)
        self.transfer_relu1 = nn.ReLU()

        self.conv1 = nn.Conv2d(in_channels=in_channel//3,out_channels=out_channel,kernel_size=3,stride=1,padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=out_channel)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels=out_channel,out_channels=out_channel,kernel_size=3,stride=1,padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=out_channel)
        self.relu2 = nn.ReLU()

        self.Tconv1 = nn.ConvTranspose2d(in_channels=out_channel,out_channels=out_channel,kernel_size=2,stride=2)
        self.Tbn1 = nn.BatchNorm2d(num_features=out_channel)
        self.Trelu1 = nn.ReLU()

        self.Tconv2 = nn.ConvTranspose2d(in_channels=out_channel,out_channels=out_channel//2,kernel_size=4,stride=4)
        self.Tbn2 = nn.BatchNorm2d(num_features=out_channel//2)
        self.Trelu2 = nn.ReLU()


    def forward(self, x):

        x = self.transfer(x)
        x = self.transfer_bn1(x)
        x = self.transfer_relu1(x)

        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.relu1(x1)

        x2 = self.conv2(x1)
        x2 = self.bn2(x2)
        x2 = self.relu2(x2)

        x = x1+x2

        out1 = self.Tconv1(x)
        out1 = self.Tbn1(out1)
        out1 = self.Trelu1(out1)

        out2 = self.Tconv2(x)
        out2 = self.Tbn2(out2)
        out2 = self.Trelu2(out2)


        return out1,out2


class Decoder2(nn.Module):
    def __init__(self, in_channel,out_channel):
        super().__init__()

        self.transfer = nn.Conv2d(in_channels=in_channel,out_channels=in_channel//4,kernel_size=1)
        self.transfer_bn1 = nn.BatchNorm2d(num_features=in_channel//4)
        self.transfer_relu1 = nn.ReLU()

        self.conv1 = nn.Conv2d(in_channels=in_channel//4,out_channels=out_channel,kernel_size=3,stride=1,padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=out_channel)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels=out_channel,out_channels=out_channel,kernel_size=3,stride=1,padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=out_channel)
        self.relu2 = nn.ReLU()

        self.Tconv1 = nn.ConvTranspose2d(in_channels=out_channel,out_channels=out_channel,kernel_size=2,stride=2)
        self.Tbn1 = nn.BatchNorm2d(num_features=out_channel)
        self.Trelu1 = nn.ReLU()


    def forward(self, x):

        x = self.transfer(x)
        x = self.transfer_bn1(x)
        x = self.transfer_relu1(x)

        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.relu1(x1)

        x2 = self.conv2(x1)
        x2 = self.bn2(x2)
        x2 = self.relu2(x2)

        x = x1+x2

        out1 = self.Tconv1(x)
        out1 = self.Tbn1(out1)
        out1 = self.Trelu1(out1)

        return out1

class Decoder1(nn.Module):
    def __init__(self, in_channel,out_channel):
        super().__init__()

        self.transfer = nn.Conv2d(in_channels=in_channel,out_channels=64,kernel_size=1)
        self.transfer_bn1 = nn.BatchNorm2d(64)
        self.transfer_relu1 = nn.ReLU()

        self.conv1 = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=64)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=64)
        self.relu2 = nn.ReLU()

        self.Tconv1 = nn.ConvTranspose2d(in_channels=64,out_channels=out_channel,kernel_size=2,stride=2)
        self.Tbn1 = nn.BatchNorm2d(num_features=out_channel)
        self.Trelu1 = nn.ReLU()


    def forward(self, x):
        x = self.transfer(x)
        x = self.transfer_bn1(x)
        x = self.transfer_relu1(x)

        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.relu1(x1)

        x2 = self.conv2(x1)
        x2 = self.bn2(x2)
        x2 = self.relu2(x2)

        x = x1+x2

        out1 = self.Tconv1(x)
        out1 = self.Tbn1(out1)
        out1 = self.Trelu1(out1)

        return out1

def unit_test():
    num_minibatch = 2
    rgb = torch.randn(num_minibatch, 3, 480, 640).cuda(1)
    rtf_net = SPSNet(9).cuda(1)
    rtf_net(rgb)
    #print('The model: ', rtf_net.modules)

if __name__ == '__main__':
    unit_test()
