from core.models.superesnet import SE_Block
from plinio.cost import params
from plinio.methods.pit import PIT
from torchinfo import summary
import torch
from torch.nn.parameter import Parameter
from torch.nn import Sequential

from plinio.methods.pit.nn.conv1d import PITConv1d
from plinio.methods.pit.nn.features_masker import PITFeaturesMasker
from plinio.methods.pit.nn.timestep_masker import PITTimestepMasker
from plinio.methods.pit.nn.dilation_masker import PITDilationMasker

def main():
    in_channels=16
    mid_channels=32
    out_channels=128
    kernel_size=5
    stride=2
    groups=1

    input_shape=(in_channels,262)

    f = PITFeaturesMasker(mid_channels)
    t = PITTimestepMasker(kernel_size)
    d = PITDilationMasker(1)

    f2 = PITFeaturesMasker(out_channels)
    t2 = PITTimestepMasker(1)
    d2 = PITDilationMasker(1)

    depth_block=Sequential(
        PITConv1d(torch.nn.Conv1d(   #first conv  PITConv1d(
            in_channels=in_channels,
            out_channels=mid_channels,
            kernel_size=kernel_size,
            stride=stride,
            groups=1),f,t,d),

        torch.nn.Conv1d(   #depthwise_conv
            in_channels=mid_channels,
            out_channels=mid_channels,
            kernel_size=kernel_size,
            stride=stride,
            groups=mid_channels),

        torch.nn.Conv1d(   #pointwise conv PITConv1d(
            in_channels=mid_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            groups=groups,
            padding='same'),  #f2,t2,d2)
    )

    pit_block = PIT(depth_block,input_shape=input_shape)
    rnd_alpha = torch.randint(0, 2, (mid_channels,), dtype=torch.float32)
    pit_block.seed._modules['0'].out_features_masker.alpha = Parameter(rnd_alpha)

    print("out features match: ", pit_block.seed._modules['1'].in_features_opt, int(torch.sum(rnd_alpha)))
    print("mask match: ",torch.equal(pit_block.seed._modules['1'].out_features_masker.alpha, rnd_alpha))

    exported_block = pit_block.export()

    summary(exported_block, input_size=input_shape, depth=12, col_names=["kernel_size", "input_size", "output_size", "num_params", "mult_adds"])

    #pit_block.seed._modules['0'].features_mask.data_ptr() == pit_block.seed._modules['1'].input_features_calculator.features_mask.data_ptr()

    out_channels = 128
    se_ch_low=4
    in_dim=262
    input_shape=(1,128,262)

    se = SE_Block(out_channels, 16, se_ch_low, in_dim=in_dim)

    summary(se, input_size=input_shape, depth=12)
    print(se(torch.randn(input_shape).cuda()).shape)

    se = PIT(se,
            input_shape=input_shape[1:],
            cost=params,
            )

    se.cuda()
    se.export()
    summary(se, input_size=input_shape, depth=12)

if __name__=="__main__":
        main()