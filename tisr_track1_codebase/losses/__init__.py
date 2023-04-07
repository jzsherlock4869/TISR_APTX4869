from copy import deepcopy

from basicsr.utils import get_root_logger
from basicsr.utils.registry import LOSS_REGISTRY
from .msssim_loss import MSSSIMLoss, SSIMLoss
from .sobel_loss import SobelLoss
from .hinge_loss import MyHingeLoss

__all__ = [
    'MSSSIMLoss', 'SSIMLoss', 'SobelLoss', 'MyHingeLoss'
]

def build_loss(opt):
    """Build loss from options.

    Args:
        opt (dict): Configuration. It must contain:
            type (str): Model type.
    """
    opt = deepcopy(opt)
    loss_type = opt.pop('type')
    loss = LOSS_REGISTRY.get(loss_type)(**opt)
    logger = get_root_logger()
    logger.info(f'Loss [{loss.__class__.__name__}] is created.')
    return loss
