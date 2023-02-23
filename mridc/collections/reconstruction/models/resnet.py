# coding=utf-8
__author__ = "Dimitrios Karkalousos"

from abc import ABC

import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from torch.nn import L1Loss

import mridc.collections.common.losses.ssim as losses
import mridc.collections.common.parts.fft as fft
import mridc.collections.common.parts.utils as utils
import mridc.collections.reconstruction.models.base as base_models
# import mridc.collections.reconstruction.models.resnet_base.resnet_block as resnet_block
import mridc.core.classes.common as common_classes
from mridc.collections.reconstruction.models.resnet_base.resnet_block import BasicBlock, ResNetwork

__all__ = ["ResNet"]


class ResNet(base_models.BaseMRIReconstructionModel, ABC):
    """
    Implementation of the ResNet, as presented in Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun.

    References
    ----------
    ..

        He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. 
        In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 770-778).

    """

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        # init superclass
        super().__init__(cfg=cfg, trainer=trainer)

        cfg_dict = OmegaConf.to_container(cfg, resolve=True)

        self.fft_centered = cfg_dict.get("fft_centered")
        self.fft_normalization = cfg_dict.get("fft_normalization")
        self.spatial_dims = cfg_dict.get("spatial_dims")
        self.coil_dim = cfg_dict.get("coil_dim")
        
        # self.basic_block = resnet_block.BasicBlock(64,64)
        # self.resnet = resnet_block.ResNet(self.basic_block)
        self.model = ResNetwork(block=BasicBlock, nb_res_blocks=6)
        self.coil_combination_method = cfg_dict.get("coil_combination_method")

        self.unrolled_iterations = cfg_dict.get("unrolled_iterations", 10)
        self.mu = torch.Tensor([0.05])

        # initialize weights if not using pretrained unet
        # TODO if not cfg_dict.get("pretrained", False):

        if cfg_dict.get("train_loss_fn") == "ssim":
            self.train_loss_fn = losses.SSIMLoss()
        elif cfg_dict.get("train_loss_fn") == "l1":
            self.train_loss_fn = L1Loss()
        elif cfg_dict.get("train_loss_fn") == "mse":
            self.train_loss_fn = torch.nn.MSELoss()
        else:
            raise ValueError("Unknown loss function: {}".format(cfg_dict.get("train_loss_fn")))
        if cfg_dict.get("val_loss_fn") == "ssim":
            self.val_loss_fn = losses.SSIMLoss()
        elif cfg_dict.get("val_loss_fn") == "l1":
            self.val_loss_fn = L1Loss()
        elif cfg_dict.get("val_loss_fn") == "mse":
            self.val_loss_fn = torch.nn.MSELoss()
        else:
            raise ValueError("Unknown loss function: {}".format(cfg_dict.get("val_loss_fn")))

        self.accumulate_estimates = False

    def sens_expand(self, x: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
        """
        Expand the sensitivity maps to the same size as the input.

        Parameters
        ----------
        x: Input data.
        sens_maps: Coil Sensitivity maps.

        Returns
        -------
        SENSE reconstruction expanded to the same size as the input sens_maps.
        """
        return fft.fft2(
            utils.complex_mul(x, sens_maps),
            centered=self.fft_centered,
            normalization=self.fft_normalization,
            spatial_dims=self.spatial_dims,
        )

    def sens_reduce(self, x: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
        """
        Reduce the sensitivity maps.

        Parameters
        ----------
        x: Input data.
        sens_maps: Coil Sensitivity maps.

        Returns
        -------
        SENSE coil-combined reconstruction.
        """
        x = fft.ifft2(
            x, centered=self.fft_centered, normalization=self.fft_normalization, spatial_dims=self.spatial_dims
        )
        return utils.complex_mul(x, utils.complex_conj(sens_maps)).sum(dim=self.coil_dim, keepdim=True)
    
    @common_classes.typecheck()
    def forward(
        self,
        y: torch.Tensor,
        sensitivity_maps: torch.Tensor,
        mask: torch.Tensor,
        init_pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass of the network.

        Parameters
        ----------
        y: Subsampled k-space data.
            torch.Tensor, shape [batch_size, n_coils, n_x, n_y, 2]
        sensitivity_maps: Coil sensitivity maps.
            torch.Tensor, shape [batch_size, n_coils, n_x, n_y, 2]
        mask: Sampling mask.
            torch.Tensor, shape [1, 1, n_x, n_y, 1]
        init_pred: Initial prediction.
            torch.Tensor, shape [batch_size, n_x, n_y, 2]
        target: Target data to compute the loss.
            torch.Tensor, shape [batch_size, n_x, n_y, 2]

        Returns
        -------
        pred: list of torch.Tensor, shape [batch_size, n_x, n_y, 2], or  torch.Tensor, shape [batch_size, n_x, n_y, 2]
             If self.accumulate_loss is True, returns a list of all intermediate estimates.
             If False, returns the final estimate.
        """
        self.mu = self.mu.clone().to(y)
        zero = torch.zeros(1, 1, 1, 1, 1).to(y)
        pred = y.clone()

        eta = torch.view_as_complex(
            utils.coil_combination(
                fft.ifft2(
                    y, centered=self.fft_centered, normalization=self.fft_normalization, spatial_dims=self.spatial_dims
                ),
                sensitivity_maps,
                method=self.coil_combination_method,
                dim=self.coil_dim,
            )
        )
        eta = torch.view_as_real(eta).unsqueeze(self.coil_dim)
        for _ in range(self.unrolled_iterations):
            eta = self.model(eta)
            # eta = self.sens_expand(eta, sensitivity_maps) * mask
            # eta = self.sens_reduce(eta, sensitivity_maps)
        return torch.view_as_complex(eta).squeeze(self.coil_dim)
