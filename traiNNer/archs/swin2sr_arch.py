from collections.abc import Sequence
from typing import Literal

from spandrel.architectures.Swin2SR import Swin2SR

from traiNNer.utils.registry import SPANDREL_REGISTRY

upsampler_type = Literal[
    "pixelshuffle_hf",
    "pixelshuffle",
    "pixelshuffle_aux",
    "pixelshuffledirect",
    "nearest+conv",
    "",
]


def swin2sr(
    scale: int = 4,
    img_size: int = 64,
    patch_size: int = 1,
    in_chans: int = 3,
    embed_dim: int = 96,
    depths: Sequence[int] = [6, 6, 6, 6],
    num_heads: Sequence[int] = [6, 6, 6, 6],
    window_size: int = 7,
    mlp_ratio: float = 4.0,
    qkv_bias: bool = True,
    drop_rate: float = 0.0,
    attn_drop_rate: float = 0.0,
    drop_path_rate: float = 0.1,
    ape: bool = False,
    patch_norm: bool = True,
    use_checkpoint: bool = False,
    img_range: float = 1.0,
    upsampler: upsampler_type = "",
    resi_connection: str = "1conv",
) -> Swin2SR:
    return Swin2SR(
        upscale=scale,
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans,
        embed_dim=embed_dim,
        depths=depths,
        num_heads=num_heads,
        window_size=window_size,
        mlp_ratio=mlp_ratio,
        qkv_bias=qkv_bias,
        drop_rate=drop_rate,
        attn_drop_rate=attn_drop_rate,
        drop_path_rate=drop_path_rate,
        ape=ape,
        patch_norm=patch_norm,
        use_checkpoint=use_checkpoint,
        img_range=img_range,
        upsampler=upsampler,
        resi_connection=resi_connection,
    )


@SPANDREL_REGISTRY.register()
def swin2sr_l(
    scale: int = 4,
    img_size: int = 64,
    patch_size: int = 1,
    in_chans: int = 3,
    embed_dim: int = 240,
    depths: Sequence[int] = [6, 6, 6, 6, 6, 6, 6, 6, 6],
    num_heads: Sequence[int] = [8, 8, 8, 8, 8, 8, 8, 8, 8],
    window_size: int = 8,
    mlp_ratio: float = 2.0,
    qkv_bias: bool = True,
    drop_rate: float = 0.0,
    attn_drop_rate: float = 0.0,
    drop_path_rate: float = 0.1,
    ape: bool = False,
    patch_norm: bool = True,
    use_checkpoint: bool = False,
    img_range: float = 1.0,
    upsampler: upsampler_type = "nearest+conv",
    resi_connection: str = "3conv",
) -> Swin2SR:
    return Swin2SR(
        upscale=scale,
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans,
        embed_dim=embed_dim,
        depths=depths,
        num_heads=num_heads,
        window_size=window_size,
        mlp_ratio=mlp_ratio,
        qkv_bias=qkv_bias,
        drop_rate=drop_rate,
        attn_drop_rate=attn_drop_rate,
        drop_path_rate=drop_path_rate,
        ape=ape,
        patch_norm=patch_norm,
        use_checkpoint=use_checkpoint,
        img_range=img_range,
        upsampler=upsampler,
        resi_connection=resi_connection,
    )


@SPANDREL_REGISTRY.register()
def swin2sr_m(
    scale: int = 4,
    img_size: int = 64,
    patch_size: int = 1,
    in_chans: int = 3,
    embed_dim: int = 180,
    depths: Sequence[int] = [6, 6, 6, 6, 6, 6],
    num_heads: Sequence[int] = [6, 6, 6, 6, 6, 6],
    window_size: int = 8,
    mlp_ratio: float = 2.0,
    qkv_bias: bool = True,
    drop_rate: float = 0.0,
    attn_drop_rate: float = 0.0,
    drop_path_rate: float = 0.1,
    ape: bool = False,
    patch_norm: bool = True,
    use_checkpoint: bool = False,
    img_range: float = 1.0,
    upsampler: upsampler_type = "pixelshuffle",
    resi_connection: str = "1conv",
) -> Swin2SR:
    return Swin2SR(
        upscale=scale,
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans,
        embed_dim=embed_dim,
        depths=depths,
        num_heads=num_heads,
        window_size=window_size,
        mlp_ratio=mlp_ratio,
        qkv_bias=qkv_bias,
        drop_rate=drop_rate,
        attn_drop_rate=attn_drop_rate,
        drop_path_rate=drop_path_rate,
        ape=ape,
        patch_norm=patch_norm,
        use_checkpoint=use_checkpoint,
        img_range=img_range,
        upsampler=upsampler,
        resi_connection=resi_connection,
    )


@SPANDREL_REGISTRY.register()
def swin2sr_s(
    scale: int = 4,
    img_size: int = 64,
    patch_size: int = 1,
    in_chans: int = 3,
    embed_dim: int = 60,
    depths: Sequence[int] = [6, 6, 6, 6],
    num_heads: Sequence[int] = [6, 6, 6, 6],
    window_size: int = 8,
    mlp_ratio: float = 2.0,
    qkv_bias: bool = True,
    drop_rate: float = 0.0,
    attn_drop_rate: float = 0.0,
    drop_path_rate: float = 0.1,
    ape: bool = False,
    patch_norm: bool = True,
    use_checkpoint: bool = False,
    img_range: float = 1.0,
    upsampler: upsampler_type = "pixelshuffledirect",
    resi_connection: str = "1conv",
) -> Swin2SR:
    return Swin2SR(
        upscale=scale,
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans,
        embed_dim=embed_dim,
        depths=depths,
        num_heads=num_heads,
        window_size=window_size,
        mlp_ratio=mlp_ratio,
        qkv_bias=qkv_bias,
        drop_rate=drop_rate,
        attn_drop_rate=attn_drop_rate,
        drop_path_rate=drop_path_rate,
        ape=ape,
        patch_norm=patch_norm,
        use_checkpoint=use_checkpoint,
        img_range=img_range,
        upsampler=upsampler,
        resi_connection=resi_connection,
    )
