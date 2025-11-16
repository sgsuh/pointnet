"""
Create: 2022.06.22
Author: SG.SUH
Python: 3.7
PyTorch: 1.8
"""

import numpy as np
import torch
import onnx
import argparse

from onnxsim import simplify

from source.model import PointNetCus

def make_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dst_data_path",
                        type=str,
                        default="sample.bin")
    parser.add_argument("--weight_path",
                        type=str,
                        default="pointnet.pth")
    parser.add_argument("--input_names",
                        type=str,
                        default="input_0")
    parser.add_argument("--output_names",
                        type=str,
                        default="output_0")
    parser.add_argument("--onnx_path",
                        type=str,
                        default="pointnet.onnx")
    parser.add_argument("--onnx_sim_path",
                        type=str,
                        default="pointnet_sim.onnx")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = make_parser()

    dst_data_path = args.dst_data_path

    pc_data2 = np.fromfile(dst_data_path, dtype = np.float32).reshape(-1, 4)
    pc_data2 = pc_data2[:, :3]

    # Point Sampler
    total_point = pc_data2.shape[0]
    rand_idx = np.random.choice(total_point, size = 128, replace = False)
    sample_points = pc_data2[rand_idx]

    # Normalize
    norm_points = sample_points - np.mean(sample_points, axis = 0)
    norm_points /= np.max(np.linalg.norm(norm_points, axis = 1))

    # To Tensor
    norm_points = torch.from_numpy(norm_points)

    net = PointNetCus()
    net.load_state_dict(torch.load(args.weight_path))
    net.to('cuda:0')
    net.eval()

    with torch.no_grad():
        norm_points = norm_points.unsqueeze(0).float().to('cuda:0')
        norm_points = norm_points.transpose(1, 2)

        out = net(norm_points)
        _, pred = torch.max(out.data, 1)

    print(pred.cpu().numpy())

    input_names = [args.input_names]
    output_names = [args.output_names]

    torch.onnx.export(net, norm_points, args.onnx_path, input_names = input_names, output_names = output_names, verbose = True, opset_version = 11)

    onnx_raw = onnx.load(args.onnx_path)
    onnx_sim, check = simplify(onnx_raw)

    assert check, 'Simplified ONNX model could not be validated'

    onnx.save(onnx_sim, args.onnx_sim_path)