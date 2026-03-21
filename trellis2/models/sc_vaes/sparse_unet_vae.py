from typing import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from ...modules.utils import convert_module_to_f16, convert_module_to_f32, zero_module
from ...modules import sparse as sp
from ...modules.norm import LayerNorm32
from tqdm import tqdm


def chunked_apply(module: nn.Module, x: torch.Tensor, chunk_size: int) -> torch.Tensor:
    if chunk_size <= 0 or x.shape[0] <= chunk_size:
        return module(x)
    
    # Process first chunk to determine output shape and dtype
    out_0 = module(x[0:chunk_size])
    out_shape = (x.shape[0],) + out_0.shape[1:]
    out = torch.empty(out_shape, device=x.device, dtype=out_0.dtype)
    out[0:chunk_size] = out_0
    
    # Process remaining chunks
    for i in range(chunk_size, x.shape[0], chunk_size):
        out[i:i+chunk_size] = module(x[i:i+chunk_size])
    return out


class SparseResBlock3d(nn.Module):
    def __init__(
        self,
        channels: int,
        out_channels: Optional[int] = None,
        downsample: bool = False,
        upsample: bool = False,
        resample_mode: Literal['nearest', 'spatial2channel'] = 'nearest',
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.downsample = downsample
        self.upsample = upsample
        self.resample_mode = resample_mode
        self.use_checkpoint = use_checkpoint
        self.low_vram = False
        self.chunk_size = 65536
        
        assert not (downsample and upsample), "Cannot downsample and upsample at the same time"

        self.norm1 = LayerNorm32(channels, elementwise_affine=True, eps=1e-6)
        self.norm2 = LayerNorm32(self.out_channels, elementwise_affine=False, eps=1e-6)
        if resample_mode == 'nearest':
            self.conv1 = sp.SparseConv3d(channels, self.out_channels, 3)
        elif resample_mode =='spatial2channel' and not self.downsample:
            self.conv1 = sp.SparseConv3d(channels, self.out_channels * 8, 3)
        elif resample_mode =='spatial2channel' and self.downsample:
            self.conv1 = sp.SparseConv3d(channels, self.out_channels // 8, 3)
        self.conv2 = zero_module(sp.SparseConv3d(self.out_channels, self.out_channels, 3))
        if resample_mode == 'nearest':
            self.skip_connection = sp.SparseLinear(channels, self.out_channels) if channels != self.out_channels else nn.Identity()
        elif resample_mode =='spatial2channel' and self.downsample:
            self.skip_connection = lambda x: x.replace(x.feats.reshape(x.feats.shape[0], out_channels, channels * 8 // out_channels).mean(dim=-1))
        elif resample_mode =='spatial2channel' and not self.downsample:
            self.skip_connection = lambda x: x.replace(x.feats.repeat_interleave(out_channels // (channels // 8), dim=1))
        self.updown = None
        if self.downsample:
            if resample_mode == 'nearest':
                self.updown = sp.SparseDownsample(2)
            elif resample_mode =='spatial2channel':
                self.updown = sp.SparseSpatial2Channel(2)
        elif self.upsample:
            self.to_subdiv = sp.SparseLinear(channels, 8)
            if resample_mode == 'nearest':
                self.updown = sp.SparseUpsample(2)
            elif resample_mode =='spatial2channel':
                self.updown = sp.SparseChannel2Spatial(2)

    def _updown(self, x: sp.SparseTensor, subdiv: sp.SparseTensor = None) -> sp.SparseTensor:
        if self.downsample:
            x = self.updown(x)
        elif self.upsample:
            x = self.updown(x, subdiv.replace(subdiv.feats > 0))
        if self.low_vram:
            if hasattr(x, 'clear_neighbor_cache'):
                x.clear_neighbor_cache()
            torch.cuda.empty_cache()
        return x

    def _forward(self, x: sp.SparseTensor) -> sp.SparseTensor:
        subdiv = None
        if self.upsample:
            subdiv = self.to_subdiv(x)
        h = x.replace(self.norm1(x.feats))
        h = h.replace(F.silu(h.feats))
        if self.resample_mode == 'spatial2channel':
            h = self.conv1(h)
        h = self._updown(h, subdiv)
        x = self._updown(x, subdiv)
        if self.resample_mode == 'nearest':
            h = self.conv1(h)
        h = h.replace(self.norm2(h.feats))
        h = h.replace(F.silu(h.feats))
        h = self.conv2(h)
        h = h + self.skip_connection(x)
        if self.low_vram:
            if hasattr(h, 'clear_neighbor_cache'):
                h.clear_neighbor_cache()
            torch.cuda.empty_cache()
        if self.upsample:
            return h, subdiv
        return h
    
    def forward(self, x: sp.SparseTensor) -> sp.SparseTensor:
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, use_reentrant=False)
        else:
            return self._forward(x)


class SparseResBlockDownsample3d(nn.Module):
    def __init__(
        self,
        channels: int,
        out_channels: Optional[int] = None,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_checkpoint = use_checkpoint
        
        self.norm1 = LayerNorm32(channels, elementwise_affine=True, eps=1e-6)
        self.norm2 = LayerNorm32(self.out_channels, elementwise_affine=False, eps=1e-6)
        self.conv1 = sp.SparseConv3d(channels, self.out_channels, 3)
        self.conv2 = zero_module(sp.SparseConv3d(self.out_channels, self.out_channels, 3))
        self.skip_connection = sp.SparseLinear(channels, self.out_channels) if channels != self.out_channels else nn.Identity()
        self.updown = sp.SparseDownsample(2)

    def _forward(self, x: sp.SparseTensor) -> sp.SparseTensor:
        h = x.replace(self.norm1(x.feats))
        h = h.replace(F.silu(h.feats))
        h = self.updown(h)
        x = self.updown(x)
        h = self.conv1(h)
        h = h.replace(self.norm2(h.feats))
        h = h.replace(F.silu(h.feats))
        h = self.conv2(h)
        h = h + self.skip_connection(x)
        return h
    
    def forward(self, x: sp.SparseTensor) -> sp.SparseTensor:
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, use_reentrant=False)
        else:
            return self._forward(x)


class SparseResBlockUpsample3d(nn.Module):
    def __init__(
        self,
        channels: int,
        out_channels: Optional[int] = None,
        use_checkpoint: bool = False,
        pred_subdiv: bool = True,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_checkpoint = use_checkpoint
        self.pred_subdiv = pred_subdiv
        
        self.norm1 = LayerNorm32(channels, elementwise_affine=True, eps=1e-6)
        self.norm2 = LayerNorm32(self.out_channels, elementwise_affine=False, eps=1e-6)
        self.conv1 = sp.SparseConv3d(channels, self.out_channels, 3)
        self.conv2 = zero_module(sp.SparseConv3d(self.out_channels, self.out_channels, 3))
        self.skip_connection = sp.SparseLinear(channels, self.out_channels) if channels != self.out_channels else nn.Identity()
        if self.pred_subdiv:
            self.to_subdiv = sp.SparseLinear(channels, 8)
        self.updown = sp.SparseUpsample(2)

    def _forward(self, x: sp.SparseTensor, subdiv: sp.SparseTensor = None) -> sp.SparseTensor:
        if self.pred_subdiv:
            subdiv = self.to_subdiv(x)
        h = x.replace(self.norm1(x.feats))
        h = h.replace(F.silu(h.feats))
        subdiv_binarized = subdiv.replace(subdiv.feats > 0) if subdiv is not None else None
        h = self.updown(h, subdiv_binarized)
        x = self.updown(x, subdiv_binarized)
        h = self.conv1(h)
        h = h.replace(self.norm2(h.feats))
        h = h.replace(F.silu(h.feats))
        h = self.conv2(h)
        h = h + self.skip_connection(x)
        if self.pred_subdiv:
            return h, subdiv
        else:
            return h
    
    def forward(self, x: sp.SparseTensor) -> sp.SparseTensor:
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, use_reentrant=False)
        else:
            return self._forward(x)


class SparseResBlockS2C3d(nn.Module):
    def __init__(
        self,
        channels: int,
        out_channels: Optional[int] = None,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_checkpoint = use_checkpoint
        
        self.norm1 = LayerNorm32(channels, elementwise_affine=True, eps=1e-6)
        self.norm2 = LayerNorm32(self.out_channels, elementwise_affine=False, eps=1e-6)
        self.conv1 = sp.SparseConv3d(channels, self.out_channels // 8, 3)
        self.conv2 = zero_module(sp.SparseConv3d(self.out_channels, self.out_channels, 3))
        self.skip_connection = lambda x: x.replace(x.feats.reshape(x.feats.shape[0], out_channels, channels * 8 // out_channels).mean(dim=-1))
        self.updown = sp.SparseSpatial2Channel(2)

    def _forward(self, x: sp.SparseTensor) -> sp.SparseTensor:
        h = x.replace(self.norm1(x.feats))
        h = h.replace(F.silu(h.feats))
        h = self.conv1(h)
        h = self.updown(h)
        x = self.updown(x)
        h = h.replace(self.norm2(h.feats))
        h = h.replace(F.silu(h.feats))
        h = self.conv2(h)
        h = h + self.skip_connection(x)
        return h
    
    def forward(self, x: sp.SparseTensor) -> sp.SparseTensor:
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, use_reentrant=False)
        else:
            return self._forward(x)


class SparseResBlockC2S3d(nn.Module):
    def __init__(
        self,
        channels: int,
        out_channels: Optional[int] = None,
        use_checkpoint: bool = False,
        pred_subdiv: bool = True,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_checkpoint = use_checkpoint
        self.pred_subdiv = pred_subdiv
        
        self.norm1 = LayerNorm32(channels, elementwise_affine=True, eps=1e-6)
        self.norm2 = LayerNorm32(self.out_channels, elementwise_affine=False, eps=1e-6)
        self.conv1 = sp.SparseConv3d(channels, self.out_channels * 8, 3)
        self.conv2 = zero_module(sp.SparseConv3d(self.out_channels, self.out_channels, 3))
        self.skip_connection = lambda x: x.replace(x.feats.repeat_interleave(out_channels // (channels // 8), dim=1))
        if pred_subdiv:
            self.to_subdiv = sp.SparseLinear(channels, 8)
        self.updown = sp.SparseChannel2Spatial(2)

    def _forward(self, x: sp.SparseTensor, subdiv: sp.SparseTensor = None) -> sp.SparseTensor:
        if self.pred_subdiv:
            subdiv = self.to_subdiv(x)
        h = x.replace(self.norm1(x.feats))
        h = h.replace(F.silu(h.feats))
        h = self.conv1(h)
        subdiv_binarized = subdiv.replace(subdiv.feats > 0) if subdiv is not None else None
        h = self.updown(h, subdiv_binarized)
        x = self.updown(x, subdiv_binarized)
        h = h.replace(self.norm2(h.feats))
        h = h.replace(F.silu(h.feats))
        h = self.conv2(h)
        h = h + self.skip_connection(x)
        if self.pred_subdiv:
            return h, subdiv
        else:
            return h
    
    def forward(self, x: sp.SparseTensor, subdiv: sp.SparseTensor = None) -> sp.SparseTensor:
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, subdiv, use_reentrant=False)
        else:
            return self._forward(x, subdiv)
        
    
class SparseConvNeXtBlock3d(nn.Module):
    def __init__(
        self,
        channels: int,
        mlp_ratio: float = 4.0,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.channels = channels
        self.use_checkpoint = use_checkpoint
        
        self.norm = LayerNorm32(channels, elementwise_affine=True, eps=1e-6)
        self.conv = sp.SparseConv3d(channels, channels, 3)
        self.mlp = nn.Sequential(
            nn.Linear(channels, int(channels * mlp_ratio)),
            nn.SiLU(),
            zero_module(nn.Linear(int(channels * mlp_ratio), channels)),
        )
        self.low_vram = False
        self.chunk_size = 65536

    def _forward(self, x: sp.SparseTensor) -> sp.SparseTensor:
        h = self.conv(x)
        if self.low_vram:
            if hasattr(h, 'clear_neighbor_cache'):
                h.clear_neighbor_cache()
        if self.low_vram:
            h = h.replace(chunked_apply(self.norm, h.feats, self.chunk_size))
            h = h.replace(chunked_apply(self.mlp, h.feats, self.chunk_size))
        else:
            h = h.replace(self.norm(h.feats))
            h = h.replace(self.mlp(h.feats))
        return h + x
    
    def forward(self, x: sp.SparseTensor) -> sp.SparseTensor:
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, use_reentrant=False)
        else:
            return self._forward(x)


class SparseUnetVaeEncoder(nn.Module):
    """
    Sparse Swin Transformer Unet VAE model.
    """
    def __init__(
        self,
        in_channels: int,
        model_channels: List[int],
        latent_channels: int,
        num_blocks: List[int],
        block_type: List[str],
        down_block_type: List[str],
        block_args: List[Dict[str, Any]],
        use_fp16: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.num_blocks = num_blocks
        self.dtype = torch.float16 if use_fp16 else torch.float32
        self._low_vram = False
        self.chunk_size = 65536

        self.input_layer = sp.SparseLinear(in_channels, model_channels[0])
        self.to_latent = sp.SparseLinear(model_channels[-1], 2 * latent_channels)
        
        self.blocks = nn.ModuleList([])
        for i in range(len(num_blocks)):
            self.blocks.append(nn.ModuleList([]))
            for j in range(num_blocks[i]):
                self.blocks[-1].append(
                    globals()[block_type[i]](
                        model_channels[i],
                        **block_args[i],
                    )
                )
            if i < len(num_blocks) - 1:
                self.blocks[-1].append(
                    globals()[down_block_type[i]](
                        model_channels[i],
                        model_channels[i+1],
                        **block_args[i],
                    )
                )
                
        self.initialize_weights()
        if use_fp16:
            self.convert_to_fp16()

    @property
    def low_vram(self) -> bool:
        return self._low_vram
    
    @low_vram.setter
    def low_vram(self, value: bool):
        self._low_vram = value
        for m in self.modules():
            if hasattr(m, 'low_vram') and m is not self:
                m.low_vram = value

    @property
    def device(self) -> torch.device:
        """
        Return the device of the model.
        """
        return next(self.parameters()).device

    @device.setter
    def device(self, value):
        pass

    def convert_to_fp16(self) -> None:
        """
        Convert the torso of the model to float16.
        """
        self.blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self) -> None:
        """
        Convert the torso of the model to float32.
        """
        self.blocks.apply(convert_module_to_f32)

    def initialize_weights(self) -> None:
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

    def spatial_chunked_forward(self, x: sp.SparseTensor, tile_size: int = 128, overlap: int = 24, sample_posterior=False, return_raw=False):
        """
        Forward pass with spatial chunking to save memory.
        """
        device = x.device
        coords = x.coords
        
        if coords.shape[0] == 0:
            out_channels = self.to_latent.out_features
            merged_feats = torch.zeros((0, out_channels), device=device, dtype=torch.float32)
            h_merged = x.replace(merged_feats.to(self.to_latent.weight.dtype))
            mean, logvar = h_merged.feats.chunk(2, dim=-1)
            if return_raw:
                return h_merged.replace(mean), mean, logvar
            else:
                return h_merged.replace(mean)
        
        # Calculate bounding box
        min_coord = coords[:, 1:].min(dim=0).values
        max_coord = coords[:, 1:].max(dim=0).values
        
        x_range = range(min_coord[0].item(), max_coord[0].item() + 1, tile_size)
        y_range = range(min_coord[1].item(), max_coord[1].item() + 1, tile_size)
        z_range = range(min_coord[2].item(), max_coord[2].item() + 1, tile_size)
        
        all_h_coords = []
        all_h_feats = []
        out_scale = None
        
        pbar = tqdm(total=len(x_range) * len(y_range) * len(z_range), desc="Tiled Encoding")
        
        for xi in x_range:
            for yi in y_range:
                for zi in z_range:
                    lower = torch.tensor([xi, yi, zi], device=device)
                    upper = lower + tile_size
                    
                    # Mask with margin
                    mask = torch.all((coords[:, 1:] >= lower - overlap) & (coords[:, 1:] < upper + overlap), dim=1)
                    
                    if mask.any():
                        # Extract sub-tensor
                        sub_x = sp.SparseTensor(
                            feats=x.feats[mask],
                            coords=x.coords[mask],
                            scale=x._scale
                        )
                        
                        # Process sub-tensor through the encoder blocks
                        h = self.input_layer(sub_x)
                        h = h.type(self.dtype)
                        for res in self.blocks:
                            for block in res:
                                h = block(h)
                        
                        # Point-wise operations
                        h = h.type(self.to_latent.weight.dtype)
                        h = h.replace(F.layer_norm(h.feats, h.feats.shape[-1:]))
                        h = self.to_latent(h)
                        
                        all_h_coords.append(h.coords)
                        all_h_feats.append(h.feats.float())
                        out_scale = h._scale
                        
                        del h, sub_x
                        torch.cuda.empty_cache()
                        
                    pbar.update(1)
        
        pbar.close()
        
        if len(all_h_coords) == 0:
            out_channels = self.to_latent.out_features
            merged_feats = torch.zeros((0, out_channels), device=device, dtype=torch.float32)
            h_merged = x.replace(merged_feats.to(self.to_latent.weight.dtype))
        else:
            all_h_coords = torch.cat(all_h_coords, dim=0)
            all_h_feats = torch.cat(all_h_feats, dim=0)
            
            unique_coords, inverse_indices = torch.unique(all_h_coords, dim=0, return_inverse=True)
            merged_feats = torch.zeros((unique_coords.shape[0], all_h_feats.shape[1]), device=device, dtype=torch.float32)
            counts = torch.zeros((unique_coords.shape[0], 1), device=device, dtype=torch.float32)
            
            merged_feats.scatter_add_(0, inverse_indices.unsqueeze(1).expand_as(all_h_feats), all_h_feats)
            counts.scatter_add_(0, inverse_indices.unsqueeze(1), torch.ones_like(all_h_feats[:, :1]))
            
            merged_feats /= counts.clamp(min=1.0)
            
            # Cache reconstruction pass
            cache_h = sp.SparseTensor(
                feats=x.feats[:, :1].clone(), 
                coords=x.coords, 
                scale=x._scale,
                spatial_shape=x.spatial_shape,
            )
            for res in self.blocks:
                for block in res:
                    if hasattr(block, 'updown') and type(block.updown).__name__ in ['SparseSpatial2Channel', 'SparseDownsample']:
                        cache_h = block.updown(cache_h)
            
            h_merged = sp.SparseTensor(
                feats=merged_feats.to(self.to_latent.weight.dtype),
                coords=unique_coords,
                scale=out_scale
            )
            h_merged._spatial_cache = cache_h._spatial_cache
        
        # Sample from the posterior distribution
        mean, logvar = h_merged.feats.chunk(2, dim=-1)
        if sample_posterior:
            std = torch.exp(0.5 * logvar)
            z = mean + std * torch.randn_like(std)
        else:
            z = mean
        z = h_merged.replace(z)
            
        if return_raw:
            return z, mean, logvar
        else:
            return z

    def forward(self, x: sp.SparseTensor, sample_posterior=False, return_raw=False, use_tiled=False, tile_size=128, overlap=24):
        if use_tiled:
            return self.spatial_chunked_forward(x, tile_size=tile_size, overlap=overlap, sample_posterior=sample_posterior, return_raw=return_raw)
            
        h = self.input_layer(x)
        h = h.type(self.dtype)
        for i, res in enumerate(self.blocks):
            for j, block in enumerate(res):
                h = block(h)
        
        if self.low_vram:
            def fused_finalize(t):
                t = t.to(self.to_latent.weight.dtype)
                t = F.layer_norm(t, (t.shape[-1],))
                t = F.linear(t, self.to_latent.weight, self.to_latent.bias)
                return t
            h = h.replace(chunked_apply(fused_finalize, h.feats, self.chunk_size))
        else:
            h = h.type(self.to_latent.weight.dtype)
            h = h.replace(F.layer_norm(h.feats, h.feats.shape[-1:]))
            h = self.to_latent(h)
        
        # Sample from the posterior distribution
        mean, logvar = h.feats.chunk(2, dim=-1)
        if sample_posterior:
            std = torch.exp(0.5 * logvar)
            z = mean + std * torch.randn_like(std)
        else:
            z = mean
        z = h.replace(z)
            
        if return_raw:
            return z, mean, logvar
        else:
            return z
    
    
class SparseUnetVaeDecoder(nn.Module):
    """
    Sparse Swin Transformer Unet VAE model.
    """
    def __init__(
        self,
        out_channels: int,
        model_channels: List[int],
        latent_channels: int,
        num_blocks: List[int],
        block_type: List[str],
        up_block_type: List[str],
        block_args: List[Dict[str, Any]],
        use_fp16: bool = False,
        pred_subdiv: bool = True,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.model_channels = model_channels
        self.num_blocks = num_blocks
        self.use_fp16 = use_fp16
        self.pred_subdiv = pred_subdiv
        self.dtype = torch.float16 if use_fp16 else torch.float32
        self.low_vram = False
        
        self.output_layer = sp.SparseLinear(model_channels[-1], out_channels)
        self.from_latent = sp.SparseLinear(latent_channels, model_channels[0])
        
        self.blocks = nn.ModuleList([])
        for i in range(len(num_blocks)):
            self.blocks.append(nn.ModuleList([]))
            for j in range(num_blocks[i]):
                self.blocks[-1].append(
                    globals()[block_type[i]](
                        model_channels[i],
                        **block_args[i],
                    )
                )
            if i < len(num_blocks) - 1:
                self.blocks[-1].append(
                    globals()[up_block_type[i]](
                        model_channels[i],
                        model_channels[i+1],
                        pred_subdiv=pred_subdiv,
                        **block_args[i],
                    )
                )
                    
        self.initialize_weights()
        self._low_vram = False
        self.chunk_size = 65536
        if use_fp16:
            self.convert_to_fp16()
            
    @property
    def low_vram(self) -> bool:
        return self._low_vram
    
    @low_vram.setter
    def low_vram(self, value: bool):
        self._low_vram = value
        for m in self.modules():
            if hasattr(m, 'low_vram') and m is not self:
                m.low_vram = value
            
    @property
    def device(self) -> torch.device:
        """
        Return the device of the model.
        """
        return next(self.parameters()).device

    @device.setter
    def device(self, value):
        pass

    def convert_to_fp16(self) -> None:
        """
        Convert the torso of the model to float16.
        """
        self.blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self) -> None:
        """
        Convert the torso of the model to float32.
        """
        self.blocks.apply(convert_module_to_f32)

    def initialize_weights(self) -> None:
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

    def forward(self, x: sp.SparseTensor, guide_subs: Optional[List[sp.SparseTensor]] = None, return_subs: bool = False) -> sp.SparseTensor:
        assert guide_subs is None or self.pred_subdiv == False, "Only decoders with pred_subdiv=False can be used with guide_subs"
        assert return_subs == False or self.pred_subdiv == True, "Only decoders with pred_subdiv=True can be used with return_subs"
        
        h = self.from_latent(x)
        h = h.type(self.dtype)
        subs_gt = []
        subs = []
        for i, res in enumerate(self.blocks):
            # Special handling for guided decoding to save VRAM
            curr_guide = None
            if guide_subs is not None and i < len(guide_subs):
                curr_guide = guide_subs[i]
                if curr_guide is not None and curr_guide.device.type == 'cpu':
                    curr_guide = curr_guide.to(h.device)
            
            for j, block in enumerate(res):
                if i < len(self.blocks) - 1 and j == len(res) - 1:
                    if self.pred_subdiv:
                        if self.training:
                            subs_gt.append(h.get_spatial_cache('subdivision'))
                        h, sub = block(h)
                        subs.append(sub)
                    else:
                        h = block(h, subdiv=curr_guide)
                else:
                    h = block(h)
                
                if self.low_vram:
                    torch.cuda.empty_cache()
            
            if self.low_vram:
                torch.cuda.empty_cache()

        if self.low_vram:
            def fused_finalize(t):
                w_dtype = self.output_layer.weight.dtype
                t = t.to(w_dtype)
                t = F.layer_norm(t, (t.shape[-1],))
                t = F.linear(t, self.output_layer.weight, self.output_layer.bias)
                return t
            h = h.replace(chunked_apply(fused_finalize, h.feats, self.chunk_size))
        else:
            h = h.type(self.output_layer.weight.dtype)
            h = h.replace(F.layer_norm(h.feats, h.feats.shape[-1:]))
            h = self.output_layer(h)

        if self.training and self.pred_subdiv:
            return h, subs_gt, subs
        else:
            if return_subs:
                return h, subs
            else:
                return h

    def _tiled_forward(self, x: sp.SparseTensor, guide_subs: Optional[List[sp.SparseTensor]] = None, tile_size: int = 120, overlap: int = 48) -> sp.SparseTensor:
        """
        Forward pass with spatial chunking to save memory during decoding.
        """
        device = x.device
        target_device = self.output_layer.weight.device
        cpu_device = torch.device('cpu')
        
        if guide_subs is None and not self.pred_subdiv:
            # Reconstruct guides if missing (simplified version for SegviGen)
            guide_subs = []
            curr_h = sp.SparseTensor(
                feats=torch.zeros((x.coords.shape[0], 1), device=device, dtype=x.dtype),
                coords=x.coords.detach(),
                scale=x._scale,
                spatial_shape=x.spatial_shape,
                spatial_cache=getattr(x, '_spatial_cache', {})
            )
            for i in range(len(self.blocks) - 1):
                res = self.blocks[i]
                if len(res) == 0:
                    guide_subs.append(None)
                    continue
                up_block = res[-1]
                if hasattr(up_block, 'updown') and type(up_block.updown).__name__ in ['SparseChannel2Spatial', 'SparseUpsample']:
                    sub_feats = curr_h.get_spatial_cache('subdivision')
                    if sub_feats is not None:
                        g_sub = sp.SparseTensor(feats=sub_feats, coords=curr_h.coords, scale=curr_h._scale, spatial_shape=curr_h.spatial_shape)
                        guide_subs.append(g_sub)
                        with torch.no_grad():
                            curr_h = up_block.updown(curr_h, g_sub)
                    else:
                        guide_subs.append(None)
                else:
                    guide_subs.append(None)
            del curr_h
            
        x = x.cpu()
        if hasattr(x, 'clear_neighbor_cache'):
            x.clear_neighbor_cache()
        if guide_subs is not None:
            guide_subs = [g.cpu() if g is not None else None for g in guide_subs]
            for g in guide_subs:
                if g is not None and hasattr(g, 'clear_neighbor_cache'):
                    g.clear_neighbor_cache()
                
        coords = x.coords
        x_feats_cpu = x.feats

        min_coord = coords[:, 1:].min(dim=0).values
        max_coord = coords[:, 1:].max(dim=0).values
        
        x_range = range(min_coord[0].item(), max_coord[0].item() + 1, tile_size)
        y_range = range(min_coord[1].item(), max_coord[1].item() + 1, tile_size)
        z_range = range(min_coord[2].item(), max_coord[2].item() + 1, tile_size)
        
        all_h_coords = []
        all_h_feats = []
        out_scale = None
        saved_spatial_cache = {}
        
        pbar = tqdm(total=len(x_range) * len(y_range) * len(z_range), desc="Tiled Decoding")
        
        for xi in x_range:
            for yi in y_range:
                for zi in z_range:
                    lower = torch.tensor([xi, yi, zi], device=cpu_device)
                    upper = lower + tile_size
                    mask = torch.all((coords[:, 1:] >= lower - overlap) & (coords[:, 1:] < upper + overlap), dim=1)
                    
                    if mask.any():
                        tile_cache = {}
                        for _sk, _sv in x._spatial_cache.items():
                            if isinstance(_sv, dict):
                                tile_cache[_sk] = {ik: iv for ik, iv in _sv.items() if 'neighbor_cache' not in ik}
                            else:
                                tile_cache[_sk] = _sv
                        
                        sub_x = sp.SparseTensor(
                            feats=x_feats_cpu[mask].to(target_device),
                            coords=coords[mask].to(target_device),
                            scale=x._scale,
                            spatial_cache=tile_cache,
                        )
                        
                        sub_guide_subs = []
                        if guide_subs is not None:
                            s_x = torch.tensor([float(s) for s in x._scale], device=cpu_device)
                            for i, g_sub in enumerate(guide_subs):
                                if g_sub is None:
                                    sub_guide_subs.append(None)
                                    continue
                                s_g = torch.tensor([float(s) for s in g_sub._scale], device=cpu_device)
                                ratio = s_x / s_g
                                g_lower = lower.float() * ratio
                                g_upper = upper.float() * ratio
                                gm = torch.all((g_sub.coords[:, 1:] >= g_lower - overlap * ratio) & (g_sub.coords[:, 1:] < g_upper + overlap * ratio), dim=1)
                                sub_guide_subs.append(sp.SparseTensor(feats=g_sub.feats[gm], coords=g_sub.coords[gm], scale=g_sub._scale, spatial_shape=g_sub.spatial_shape))
                        else:
                            sub_guide_subs = None
                            
                        h = self.forward(sub_x, sub_guide_subs, return_subs=False)
                        all_h_coords.append(h.coords.cpu())
                        all_h_feats.append(h.feats.float().cpu())
                        
                        for k in ['shape', 'subdivision']:
                            if k in h._spatial_cache:
                                saved_spatial_cache[k] = h._spatial_cache[k]
                        
                        out_scale = h._scale
                        del h, sub_x, sub_guide_subs
                        torch.cuda.empty_cache()
                    
                    pbar.update(1)
        
        pbar.close()
        
        if len(all_h_coords) == 0:
            out_channels = self.output_layer.out_features
            merged_feats = torch.zeros((0, out_channels), device=device, dtype=torch.float32)
            h_merged = x.replace(merged_feats.to(self.output_layer.weight.dtype))
        else:
            all_h_coords = torch.cat(all_h_coords, dim=0)
            all_h_feats = torch.cat(all_h_feats, dim=0)
            unique_coords, inverse_indices = torch.unique(all_h_coords, dim=0, return_inverse=True)
            merged_feats = torch.zeros((unique_coords.shape[0], all_h_feats.shape[1]), device='cpu', dtype=torch.float32)
            counts = torch.zeros((unique_coords.shape[0], 1), device='cpu', dtype=torch.float32)
            merged_feats.scatter_add_(0, inverse_indices.unsqueeze(1).expand_as(all_h_feats), all_h_feats)
            counts.scatter_add_(0, inverse_indices.unsqueeze(1), torch.ones((all_h_feats.shape[0], 1), device='cpu'))
            merged_feats /= counts.clamp(min=1.0)
            
            h_merged = sp.SparseTensor(
                feats=merged_feats.to(device).to(self.output_layer.weight.dtype),
                coords=unique_coords.to(device),
                scale=out_scale,
                spatial_cache=saved_spatial_cache,
            )
        return h_merged

    def _tiled_upsample(self, x: sp.SparseTensor, upsample_times: int, tile_size: int = 16, overlap: int = 2) -> torch.Tensor:
        """
        Tiled version of upsample() to avoid OOM on large meshes.
        """
        device = x.device
        x_cpu = x.cpu()
        if hasattr(x_cpu, 'clear_neighbor_cache'):
            x_cpu.clear_neighbor_cache()

        coords_cpu = x_cpu.coords
        feats_cpu  = x_cpu.feats
        min_coord = coords_cpu[:, 1:].min(dim=0).values
        max_coord = coords_cpu[:, 1:].max(dim=0).values
        
        x_range = range(min_coord[0].item(), max_coord[0].item() + 1, tile_size)
        y_range = range(min_coord[1].item(), max_coord[1].item() + 1, tile_size)
        z_range = range(min_coord[2].item(), max_coord[2].item() + 1, tile_size)
        
        scale_factor = 2 ** upsample_times
        all_out_coords = []
        prev_low_vram = self.low_vram
        self.low_vram = True

        pbar = tqdm(total=len(x_range) * len(y_range) * len(z_range), desc="Tiled Upsampling")
        for xi in x_range:
            for yi in y_range:
                for zi in z_range:
                    lower = torch.tensor([xi, yi, zi])
                    upper = lower + tile_size
                    mask = torch.all((coords_cpu[:, 1:] >= lower - overlap) & (coords_cpu[:, 1:] <  upper + overlap), dim=1)
                    if mask.any():
                        sub_x = sp.SparseTensor(feats=feats_cpu[mask].to(device), coords=coords_cpu[mask].to(device), scale=x._scale)
                        tile_coords = self.upsample(sub_x, upsample_times)
                        if tile_coords.shape[0] > 0:
                            out_lower = lower * scale_factor
                            out_upper = upper * scale_factor
                            keep = torch.all((tile_coords[:, 1:] >= out_lower.to(device)) & (tile_coords[:, 1:] <  out_upper.to(device)), dim=1)
                            all_out_coords.append(tile_coords[keep].cpu())
                        del sub_x, tile_coords
                        torch.cuda.empty_cache()
                    pbar.update(1)
        self.low_vram = prev_low_vram
        pbar.close()
        if not all_out_coords:
            return torch.zeros((0, 4), dtype=torch.long, device=device)
        result = torch.cat(all_out_coords, dim=0)
        result = torch.unique(result, dim=0)
        return result.to(device)

    def upsample(self, x: sp.SparseTensor, upsample_times: int) -> torch.Tensor:
        assert self.pred_subdiv == True, "Only decoders with pred_subdiv=True can be used with upsampling"
        h = self.from_latent(x)
        h = h.type(self.dtype)
        for i, res in enumerate(self.blocks):
            if i == upsample_times:
                return h.coords
            for j, block in enumerate(res):
                if i < len(self.blocks) - 1 and j == len(res) - 1:
                    h, sub = block(h)
                    if h.coords.shape[0] == 0: return h.coords
                else:
                    h = block(h)
                if self.low_vram:
                    if hasattr(h, 'clear_neighbor_cache'): h.clear_neighbor_cache()
                    torch.cuda.empty_cache()
        return h.coords
       
