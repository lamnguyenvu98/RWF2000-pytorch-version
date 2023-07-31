import torch
import torch.nn as nn
from src.models.fgn_model import FlowGatedNetwork
import argparse
import os
import platform

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

def split_file_name(path):
    abs_path = os.path.abspath(path)
    if not os.path.exists(abs_path):
        raise FileExistsError("File doesn't exist. Verify the path again:\n\t{}".format(abs_path))
    if platform.system() == "Linux":
        sep = "/"
    else:
        sep = "\\"
    
    filename = abs_path.split(sep)[-1]
    filename = filename.split(".")[0]
    return filename
    

def main():
    parser = argparse.ArgumentParser(
        prog='Export script', 
        epilog="""See '<command> --help' to read about a specific sub-command.\n
        For example:\n
        \texport-script onnx --help"""
    )
    base_parser  = argparse.ArgumentParser(add_help=False)
    base_parser.add_argument('--out-dir', '-d', required=True, type=str, help='Path of directory to export model')
    base_parser.add_argument('--checkpoint', '-c', required=True, type=str, help='Path to checkpoint')
    base_parser.add_argument("--name", "-n", required=False, type=str, help="[Optional] Name of torchscript/onnx file. It will take the name of checkpoint file by default")
    
    subparsers = parser.add_subparsers(dest='type', help='Sub-commands')
    subparsers.add_parser('torchscript', parents=[base_parser], help="Export checkpoint to torchscript")
    onnx_parser = subparsers.add_parser('onnx', parents=[base_parser], help="Export checkpoint to ONNX")
    onnx_parser.add_argument("--opset-version", "-op", default=11, type=int, help="Opset version of ONNX")
        
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = FlowGatedNetwork()
    trained_ckp = torch.load(args.checkpoint, map_location='cpu')['state_dict']
    model_state = model.state_dict()
    for k, v in model_state.items():
        model_state[k] = trained_ckp["model."+ k]
    model.load_state_dict(model_state)
    model = model.to(device)
    # model.apply(lambda m: replace_conv2d_with_same_padding(m, 224))

    x = torch.randn(1, 5, 64, 224, 224)
    
    if args.name is None:
        model_file_name = split_file_name(args.checkpoint)
    else:
        model_file_name = args.name
    
    with torch.no_grad():
        if args.type == "torchscript":
            tc = torch.jit.trace(model, x)
            tc.save(os.path.join(args.out_dir, model_file_name + '.tc'))
            
        elif args.type == "onnx":
            
            dynamic_batch = {
                "input": [0],
                "output": [0]
            }
            
            torch.onnx.export(model=model, 
                                args=x, 
                                f=os.path.join(args.out_dir, model_file_name + ".onnx"), 
                                export_params=True, 
                                input_names=["input"], 
                                output_names=["output"],
                                opset_version=args.opset_version,
                                dynamic_axes=dynamic_batch)
        
        else:
            assert '{} is not supported. Supported convert file: "onnx" and "torchscript"'

if __name__ == '__main__':
    main()






