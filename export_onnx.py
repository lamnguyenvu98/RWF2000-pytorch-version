import torch
import torch.nn as nn
from model import FGN
import argparse

def calc_same_padding(kernel_size, stride, input_size):
    if isinstance(kernel_size, tuple):
        kernel_size = kernel_size[0]

    if isinstance(stride, tuple):
        stride = stride[0]

    if isinstance(input_size, tuple):
        input_size = input_size[0]

    pad = ((stride - 1) * input_size - stride + kernel_size) / 2
    return int(pad)

def replace_conv2d_with_same_padding(m: nn.Module, input_size=224):
    if isinstance(m, nn.Conv3d):
        if m.padding == "same":
            m.padding = calc_same_padding(
                kernel_size=m.kernel_size,
                stride=m.stride,
                input_size=input_size
            )
if __name__ == '__main__':
    parser  = argparse.ArgumentParser()
    parser.add_argument('--savepath', '-d', required=True, type=str, help='Path to write result')
    parser.add_argument('--checkpoint', '-c', required=True, type=str, help='Path to checkpoint')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = FGN()
    trained_ckp = torch.load(args.checkpoint, map_location='cpu')['state_dict']
    model.load_state_dict(trained_ckp)
    model = model.to(device)
    # model.apply(lambda m: replace_conv2d_with_same_padding(m, 224))

    x = torch.randn(1, 5, 64, 224, 224)

    model.to_onnx(file_path=args.savepath,
                  input_sample=x,
                  export_params=True,
                  input_names=["input"],
                  output_names=["output"])





