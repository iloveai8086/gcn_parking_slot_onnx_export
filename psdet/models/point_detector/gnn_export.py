import torch
from torch import nn
from copy import deepcopy
import torch.nn.functional as F
from dgcnn import DGCNN
import numpy as np
import onnx
import onnxruntime


class DebugOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, name):
        return x

    @staticmethod
    def symbolic(g, x, name):
        return g.op("my::Debug", x, name_s=name)


debug_apply = DebugOp.apply


class Debugger():
    def __init__(self):
        super().__init__()
        self.torch_value = dict()
        self.onnx_value = dict()
        self.output_debug_name = []

    def debug(self, x, name):
        self.torch_value[name] = x.detach().cpu().numpy()
        return debug_apply(x, name)

    def extract_debug_model(self, input_path, output_path):
        model = onnx.load(input_path)
        inputs = [input.name for input in model.graph.input]
        orin_out = [output.name for output in model.graph.output][0]
        outputs = []

        for node in model.graph.node:
            if node.op_type == 'Debug':
                debug_name = node.attribute[0].s.decode('ASCII')
                self.output_debug_name.append(debug_name)

                output_name = node.output[0]
                outputs.append(output_name)

                node.op_type = 'Identity'
                node.domain = ''
                del node.attribute[:]
        outputs.append(orin_out)
        e = onnx.utils.Extractor(model)
        extracted = e.extract_model(inputs, outputs)
        onnx.save(extracted, output_path)

    def run_debug_model(self, input, debug_model):
        sess = onnxruntime.InferenceSession(debug_model,
                                            providers=['CPUExecutionProvider'])

        onnx_outputs = sess.run(None, input)
        for name, value in zip(self.output_debug_name, onnx_outputs):
            self.onnx_value[name] = value

    def print_debug_result(self):
        for name in self.torch_value.keys():
            if name in self.onnx_value:
                mse = np.mean(self.torch_value[name] - self.onnx_value[name]) ** 2
                print(f"{name} MSE: {mse}")


debugger = Debugger()


def MLP(channels: list, do_bn=True, drop_out=False):
    """ Multi-layer perceptron """
    n = len(channels)
    layers = []
    for i in range(1, n):
        layers.append(
            nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True))
        if i < (n - 1):
            if do_bn:
                layers.append(nn.BatchNorm1d(channels[i]))
            layers.append(nn.ReLU())
            if drop_out:
                layers.append(nn.Dropout(0.5))

    return nn.Sequential(*layers)


class EdgePredictor(nn.Module):
    "Edge connectivity predictor using MLPs"

    def __init__(self, cfg):
        super(EdgePredictor, self).__init__()
        self.encoder = MLP([cfg.input_dim] + cfg.layers + [1], drop_out=True)
        nn.init.constant_(self.encoder[-1].bias, 0.0)

    def forward(self, data_dict):
        x = data_dict['descriptors']
        b, c, n = x.shape
        inputs = torch.zeros(b, c * 2, n * n).cuda()
        for i in range(n):
            for j in range(n):
                idx = i * n + j
                inputs[:, :, idx] = torch.cat([x[:, :, i], x[:, :, j]], dim=1)

        preds = torch.sigmoid(self.encoder(inputs))
        data_dict['edges_pred'] = preds
        return data_dict


class PointEncoder(nn.Module):
    """ Joint encoding of depth, intensity and location (x, y, z) using MLPs"""

    def __init__(self, input_dim, feature_dim, layers):
        super(PointEncoder, self).__init__()
        self.encoder = MLP([input_dim] + layers + [feature_dim])
        nn.init.constant_(self.encoder[-1].bias, 0.0)

    def forward(self, pts):
        return self.encoder(pts)


def attention(query, key, value):
    dim = query.shape[1]
    scores = torch.einsum('bdhn,bdhm->bhnm', query, key)

    # scores = debugger.debug(scores, 'einsum_0')

    scores = scores / dim ** .5

    prob = torch.nn.functional.softmax(scores, dim=-1)
    out0 = torch.einsum('bhnm,bdhm->bdhn', prob, value)

    # out0 = debugger.debug(out0, 'einsum_1')

    return out0, prob


class MultiHeadedAttention(nn.Module):
    """ Multi-head attention to increase model expressivitiy """

    def __init__(self, num_heads: int, d_model: int):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % num_heads == 0
        self.dim = d_model // num_heads
        self.num_heads = num_heads
        self.merge = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.proj = nn.ModuleList([deepcopy(self.merge) for _ in range(3)])

    def forward(self, query, key, value):
        batch_dim = query.size(0)
        query, key, value = [l(x).view(batch_dim, self.dim, self.num_heads, -1)
                             for l, x in zip(self.proj, (query, key, value))]
        x, _ = attention(query, key, value)
        return self.merge(x.contiguous().view(batch_dim, self.dim * self.num_heads, -1))


class AttentionalPropagation(nn.Module):
    def __init__(self, feature_dim: int, num_heads: int):
        super(AttentionalPropagation, self).__init__()
        self.attn = MultiHeadedAttention(num_heads, feature_dim)
        self.mlp = MLP([feature_dim * 2, feature_dim * 2, feature_dim])
        nn.init.constant_(self.mlp[-1].bias, 0.0)

    def forward(self, x, source):
        message = self.attn(x, source, source)
        return self.mlp(torch.cat([x, message], dim=1))


class AttentionalGNN(nn.Module):
    def __init__(self, feature_dim: int, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([
            AttentionalPropagation(feature_dim, 4)
            for _ in range(num_layers)])

    def forward(self, desc0, desc1=None):
        if desc1 is None:
            return self.forward_self_attention(desc0)

        for i, layer in enumerate(self.layers):
            if i % 2 == 0:
                src0, src1 = desc0, desc1
            else:  # cross attention
                src0, src1 = desc1, desc0
            delta0, delta1 = layer(desc0, src0), layer(desc1, src1)
            desc0, desc1 = (desc0 + delta0), (desc1 + delta1)
        return desc0

    def forward_self_attention(self, x):
        for layer in self.layers:
            src = x
            delta = layer(x, src)
            x = x + delta
        return x


class GCNEncoder(nn.Module):
    def __init__(self):
        super(GCNEncoder, self).__init__()
        # self.cfg = cfg
        self.feat_dim = 128

        self.point_encoder = PointEncoder(2, self.feat_dim, [32, 64])

        self.gnn = AttentionalGNN(self.feat_dim, 1)

        self.proj = nn.Conv1d(self.feat_dim, 64, kernel_size=1, bias=True)

    def forward(self, points, descriptors):
        points = points[:, :, :2]  # [B, N num_points, 2]
        # points = debugger.debug(points, 'slice')

        points = points.permute(0, 2, 1)  # [B, 2, num_points]
        # points = debugger.debug(points, 'permute')

        x = self.point_encoder(points)  # [B, desc_dim, num_points]

        desc = descriptors  # [B, desc_dim, num_points]

        x += desc

        # Multi-layer Transformer network.
        x = self.gnn(x)

        # MLP projection.
        x = self.proj(x)

        descriptors = x
        # descriptors = debugger.debug(descriptors, 'descriptors')
        return descriptors


torch.manual_seed(1)  # 固定随机种子（CPU）

model = GCNEncoder()
model.eval()
print(model)

dummy_2 = torch.ones(1, 128, 10)
dummy_1 = torch.ones(1, 10, 2)

out = model(dummy_1, dummy_2).detach().numpy()
print(out)
print(out.shape)

torch.onnx.export(model, (dummy_1, dummy_2), "model.onnx", input_names=["one", "two"],
                  output_names=["prob"], opset_version=12)

# debugger.extract_debug_model('model.onnx', 'after_debug.onnx')  # 可视化一下
#
# debugger.run_debug_model({'one': dummy_1.numpy(), 'two': dummy_2.numpy()}, 'after_debug.onnx')
# debugger.print_debug_result()
sess = onnxruntime.InferenceSession('model.onnx')
ort_output = sess.run(None, {'one': dummy_1.numpy(), 'two': dummy_2.numpy()})[0]
print(ort_output)
print(ort_output.shape)
assert np.allclose(out, ort_output, rtol=1e-3, atol=1e-5)

# export LD_LIBRARY_PATH=/media/ros/A666B94D66B91F4D/ros/new_deploy/TensorRT-8.5.1.7.Linux.x86_64-gnu.cuda-11.8.cudnn8.6/TensorRT-8.5.1.7/lib:$LD_LIBRARY_PATH
