from spandrel.architectures.DAT import DAT
from traiNNer.utils.config import Config
from traiNNer.utils.registry import SPANDREL_REGISTRY


@SPANDREL_REGISTRY.register()
def dat(in_chans=3,
        img_size=64,
        img_range=1.,
        split_size=(8, 32),
        depth=(6, 6, 6, 6, 6, 6),
        embed_dim=180,
        num_heads=(6, 6, 6, 6, 6, 6),
        expansion_factor=4,
        resi_connection='1conv',
        **kwargs):
    return DAT(upscale=Config.get_scale(),
               in_chans=in_chans,
               img_size=img_size,
               img_range=img_range,
               split_size=split_size,
               depth=depth,
               embed_dim=embed_dim,
               num_heads=num_heads,
               expansion_factor=expansion_factor,
               resi_connection=resi_connection,
               **kwargs)


@SPANDREL_REGISTRY.register()
def dat_2(**kwargs):
    return DAT(upscale=Config.get_scale(), in_chans=3, img_size=64, img_range=1., split_size=[8, 32],
               depth=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6], expansion_factor=2,
               resi_connection='1conv', **kwargs)
