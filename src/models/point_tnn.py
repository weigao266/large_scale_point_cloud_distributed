import gin
import torch
import torch.nn as nn
import torch.nn.functional as F
import MinkowskiEngine as ME
from MinkowskiEngine.MinkowskiKernelGenerator import KernelGenerator

from src.models.transformer_base import LocalSelfAttentionBase, ResidualBlockWithPointsBase
from src.models.common import stride_centroids, downsample_points, downsample_embeddings
import src.cuda_ops.functions.sparse_ops as ops

from src.tnn_module.tnn_layer import TnnLayer

class MaxPoolWithPoints(nn.Module):
# class MaxPoolWithPoints(LightningModule):
    def __init__(self, kernel_size=2, stride=2):
        assert kernel_size == 2 and stride == 2
        super(MaxPoolWithPoints, self).__init__()
        self.pool = ME.MinkowskiMaxPooling(kernel_size=kernel_size, stride=stride, dimension=3)
        # self.pool = nn.MaxPool3d(kernel_size=kernel_size, stride=stride)

    ## to modify
    def forward(self, stensor, points, counts):
        assert isinstance(stensor, ME.SparseTensor)
        assert len(stensor) == len(points)
        cm = stensor.coordinate_manager
        down_stensor = self.pool(stensor)
        cols, rows = cm.stride_map(stensor.coordinate_map_key, down_stensor.coordinate_map_key)
        size = torch.Size([len(down_stensor), len(stensor)])
        down_points, down_counts = stride_centroids(points, counts, rows, cols, size)
        return down_stensor, down_points, down_counts

def biggest_power_of_2(x):  
    return 0 if x == 0 else 2 ** ((x - 1).bit_length() - 1)

####################################
# Layers
####################################
@gin.configurable
class TnnLayer_ME(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels=None,
        kernel_size=3,
        stride=1,
        dilation=1,
        num_heads=8,
        rpe_embedding=64, 
        glu_dim=128
    ):
        out_channels = in_channels if out_channels is None else out_channels
        assert out_channels % num_heads == 0
        assert kernel_size % 2 == 1
        assert stride == 1, "Currently, this layer only supports stride == 1"
        assert dilation == 1,"Currently, this layer only supports dilation == 1"
        super(TnnLayer_ME, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.intra_pos_mlp = nn.Sequential(
            nn.Linear(3, 3, bias=False),
            nn.BatchNorm1d(3),
            nn.ReLU(inplace=True),
            nn.Linear(3, in_channels, bias=False),
            nn.BatchNorm1d(in_channels),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels, in_channels)
        )

        self.tnn = TnnLayer(
            dim=in_channels, num_heads=num_heads, 
            rpe_embedding=rpe_embedding, glu_dim=glu_dim
            )
        self.bn0 = nn.BatchNorm1d(in_channels)

        if self.in_channels != self.out_channels:
            self.linear = nn.Conv1d(in_channels, out_channels, kernel_size = 1)
            self.bn1 = nn.BatchNorm1d(out_channels)

        self.relu = nn.ReLU(inplace=True)
        
        self.kernel_generator = KernelGenerator(kernel_size=kernel_size,
                                                    stride=stride,
                                                    dilation=dilation,
                                                    dimension=3)    

    def get_kernel_map_and_out_key(self, stensor):    
        cm = stensor.coordinate_manager
        in_key = stensor.coordinate_key
        out_key = cm.stride(in_key, self.kernel_generator.kernel_stride)

        return out_key

    def forward(self, stensor, norm_points):
        # dtype = stensor._F.dtype
        # device = stensor._F.device

        # print(stensor.size())
        # print(norm_points.size())
        # print('---------------------')
        # query, key, value, and relative positional encoding
        intra_pos_enc = self.intra_pos_mlp(norm_points)
        # print(intra_pos_enc.size())
        stensor = stensor + intra_pos_enc ## g_i
        # print(stensor.size())

        # key-query map
        out_key = self.get_kernel_map_and_out_key(stensor)
        # print(out_key.size())
        # print(stensor.F.size())

        # n = stensor.F.size(0)
        # out_F = stensor.F.unsqueeze(0).transpose(-1, -2).contiguous()
        # sampled_n = biggest_power_of_2(n)
        # sampled_F = stensor.F[:sampled_n, :]
        # out_F[:, :, :sampled_n] = self.tnn(sampled_F.unsqueeze(0)).transpose(-1, -2).contiguous()

        out_F = self.tnn(stensor.F.unsqueeze(0)).transpose(-1, -2).contiguous()

        out_F = self.relu(self.bn0(out_F))

        if self.in_channels != self.out_channels:
            out_F = self.relu(self.bn1(self.linear(out_F)))

        out_F = out_F.transpose(-1, -2).contiguous()

        return ME.SparseTensor(out_F.squeeze(0),
                               coordinate_map_key=out_key,
                               coordinate_manager=stensor.coordinate_manager)


# ####################################
# # Blocks
# ####################################
# @gin.configurable
# class LightweightSelfAttentionBlock(ResidualBlockWithPointsBase):
#     LAYER = LightweightSelfAttentionLayer


####################################
# Models
####################################
@gin.configurable
class PointTNN(nn.Module):
    ENC_DIM = 32
    LAYERS = (2, 3, 4, 6, 2, 2, 2, 2)
    PLANES = (32, 64, 128, 384, 640, 384, 384, 256, 128)
    # PLANES = (16, 32, 48, 96, 128, 96, 48, 32)
    # PLANES = (8, 16, 32, 64, 96, 64, 32, 16)
    # PLANES = (32, 64, 128, 256, 128, 128, 128, 64)
    # LAYERS = (1, 1, 2, 3, 1, 1, 1, 1)
    # LAYERS = (1, 1, 1, 1, 1, 1, 1, 1)
    QMODE = ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE

    ## the input dim after voxelization: in_channels + ENC_DIM
    ## the dim of first TNN/TF: in_channels + ENC_DIM --> PLANE[0]
    # PLANES = (8, 16, 32, 64, 96, 64, 32, 16, 8)
    U_layers = 4
    assert len(PLANES) == len(LAYERS) + 1
    assert U_layers*2 == len(LAYERS)

    def __init__(self, in_channels, out_channels, activation_checkpointing):
        super(PointTNN, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.activation_checkpointing = activation_checkpointing
        
        self.enc_mlp = nn.Sequential(
            nn.Linear(3, self.ENC_DIM, bias=False),
            nn.BatchNorm1d(self.ENC_DIM),
            nn.Tanh(),
            nn.Linear(self.ENC_DIM, self.ENC_DIM, bias=False),
            nn.BatchNorm1d(self.ENC_DIM),
            nn.Tanh()
        )


        ## the first layer
        self.init_feature_mappings = TnnLayer_ME(
            in_channels=in_channels + self.ENC_DIM, out_channels =  self.PLANES[0], 
            num_heads=8, rpe_embedding=32, glu_dim=64
            )

        self.down_feature_mappings = nn.ModuleList()
        self.down_blocks = nn.ModuleList()
        
        for i in range(self.U_layers):
            self.down_feature_mappings.append(
                TnnLayer_ME(
                    in_channels=self.PLANES[i], out_channels =  self.PLANES[i+1], 
                    num_heads=8, rpe_embedding=32, glu_dim=64
                    )
            )
            self.down_blocks.append(
                nn.ModuleList([TnnLayer_ME(self.PLANES[i+1]) for _ in range(self.LAYERS[i])])
            )

        ## ========================================upsampling
        self.up_feature_mappings = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        for i in range(self.U_layers):
            self.up_feature_mappings.append(
                TnnLayer_ME(
                    in_channels = self.PLANES[i+self.U_layers] + self.PLANES[self.U_layers-i], out_channels = self.PLANES[i+self.U_layers+1],
                    num_heads=8, rpe_embedding=32, glu_dim=64
                    )
            )
            self.up_blocks.append(
                nn.ModuleList([TnnLayer_ME(self.PLANES[i+self.U_layers+1]) for _ in range(self.LAYERS[i+self.U_layers])])
            )

        self.final = nn.Sequential(
            nn.Linear(self.PLANES[-1] + self.ENC_DIM, self.PLANES[-1], bias=False),
            nn.BatchNorm1d(self.PLANES[-1]),
            nn.ReLU(inplace=True),
            nn.Linear(self.PLANES[-1], out_channels)
        )

        self.pool = MaxPoolWithPoints()
        self.pooltr = ME.MinkowskiPoolingTranspose(kernel_size=2, stride=2, dimension=3)

    @torch.no_grad()
    def normalize_points(self, points, centroids, tensor_map):
        tensor_map = tensor_map if tensor_map.dtype == torch.int64 else tensor_map.long()
        norm_points = points - centroids[tensor_map]
        return norm_points

    @torch.no_grad()
    def normalize_centroids(self, down_points, coordinates, tensor_stride):
        norm_points = (down_points - coordinates[:, 1:]) / tensor_stride - 0.5
        return norm_points

    def voxelize_with_centroids(self, x: ME.TensorField):
        cm = x.coordinate_manager
        points = x.C[:, 1:]
        # print(type(x))
        # print(x.size())
        out = x.sparse()
        # print(out.size())

        size = torch.Size([len(out), len(x)])
        tensor_map, field_map = cm.field_to_sparse_map(x.coordinate_key, out.coordinate_key)
        points_p1, count_p1 = downsample_points(points, tensor_map, field_map, size)

        norm_points = self.normalize_points(points, points_p1, tensor_map)

        pos_embs = self.enc_mlp(norm_points)
        down_pos_embs = downsample_embeddings(pos_embs, tensor_map, size, mode="avg")
        # print(out.F.size())
        # print(down_pos_embs.size())
        out = ME.SparseTensor(torch.cat([out.F, down_pos_embs], dim=1),
                            coordinate_map_key=out.coordinate_key,
                            coordinate_manager=cm)

        norm_points_p1 = self.normalize_centroids(points_p1, out.C, out.tensor_stride[0])
        return out, norm_points_p1, points_p1, count_p1, pos_embs

    def devoxelize_with_centroids(self, out: ME.SparseTensor, x: ME.TensorField, h_embs):
        # print(torch.cat([out.slice(x).F, h_embs], dim=1).size())
        # print(out.slice(x).F.size())
        # print(h_embs.size())
        out = self.final(torch.cat([out.slice(x).F, h_embs], dim=1))
        return out

    def forward(self, x):
        ## voxelization
        # print(x.size())
        # print(x.C[:, 1:].size())
        # print(type(x.C[:, 1:]))
        out, norm_points_p1, points_p1, count_p1, pos_embs = self.voxelize_with_centroids(x)


        ## feature mapping
        if not self.activation_checkpointing:
            out = self.init_feature_mappings(out, norm_points_p1) ## feature mapping
        else:
            def create_custom_forward(module_in):
                def custom_forward(*inputs):
                    # None for past_key_value
                    return module_in(*inputs)
                return custom_forward
            out = torch.utils.checkpoint.checkpoint(create_custom_forward(self.init_feature_mappings), out, norm_points_p1)

        # print(out.size())
        # print(out.get_device())
        # print(norm_points_p1.get_device())

        outs = []
        norm_points = []; norm_points.append(norm_points_p1)
        points = points_p1
        counts = count_p1
        for i in range(self.U_layers):
            # print(norm_points[i].get_device())
            # print(out.get_device())
            if not self.activation_checkpointing:
                outs_tmp = self.down_feature_mappings[i](out, norm_points[i])
            else:
                def create_custom_forward(module_in):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module_in(*inputs)
                    return custom_forward
                outs_tmp = torch.utils.checkpoint.checkpoint(create_custom_forward(self.down_feature_mappings[i]), out, norm_points[i])

            out, points, counts = self.pool(outs_tmp, points, counts)
            tmp_norm_points = self.normalize_centroids(points, out.C, out.tensor_stride[0])
            for module in self.down_blocks[i]:
                if not self.activation_checkpointing:
                    out = module(out, tmp_norm_points)
                else:
                    def create_custom_forward(module_in):
                        def custom_forward(*inputs):
                            # None for past_key_value
                            return module_in(*inputs)
                        return custom_forward
                    out = torch.utils.checkpoint.checkpoint(create_custom_forward(module), out, tmp_norm_points)

            ## have the i + 1 item
            norm_points.append(tmp_norm_points)
            outs.append(outs_tmp)
            # print(out.size())

        # ## ======================================================================================
        
        for i in range(self.U_layers):
            out = self.pooltr(out)
            out = ME.cat(out, outs[-(i+1)])
            
            if not self.activation_checkpointing:
                out = self.up_feature_mappings[i](out, norm_points[-(i+2)])
            else:
                def create_custom_forward(module_in):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module_in(*inputs)
                    return custom_forward
                out = torch.utils.checkpoint.checkpoint(create_custom_forward(self.up_feature_mappings[i]), out, norm_points[-(i+2)])
            for module in self.up_blocks[i]:
                if not self.activation_checkpointing:
                    out = module(out, norm_points[-(i+2)])
                else:
                    def create_custom_forward(module_in):
                        def custom_forward(*inputs):
                            # None for past_key_value
                            return module_in(*inputs)
                        return custom_forward
                    out = torch.utils.checkpoint.checkpoint(create_custom_forward(module), out, norm_points[-(i+2)])
            # print(out.size())


        # print(out.F.size())
        out = self.devoxelize_with_centroids(out, x, pos_embs)
        # print(out.size())
        return out


@gin.configurable
class PointTNNSmall(PointTNN):
    LAYERS = (2, 2, 2, 2, 2, 2)
    ## the input dim after voxelization: in_channels + ENC_DIM
    ## the dim of first TNN/TF: in_channels + ENC_DIM --> PLANE[0]
    PLANES = (32, 64, 128, 256, 128, 64, 32)
    U_layers = 3

    ENC_DIM = 32


@gin.configurable
class PointTNNSmaller(PointTNN):
    ## the input dim after voxelization: in_channels + ENC_DIM
    ## the dim of first TNN/TF: in_channels + ENC_DIM --> PLANE[0]
    LAYERS = (1, 1, 1, 1, 1, 1)
    PLANES = (16, 32, 64, 128, 64, 32, 16)
    U_layers = 3

    ENC_DIM = 16
