from pathlib import Path
from typing import Tuple

import numpy as np
import timm
import torch
from sklearn.metrics import roc_auc_score
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from indad.utils import (
    GaussianBlur,
    NativeGaussianBlur,
    get_coreset_idx_randomp,
    get_tqdm_params,
)

EXPORT_DIR = Path("./exports")

if not EXPORT_DIR.exists():
    EXPORT_DIR.mkdir()


class KNNExtractor(torch.nn.Module):
    def __init__(
        self,
        backbone_name: str = "resnet50",
        out_indices: Tuple = None,
        pool_last: bool = False,
    ):
        super().__init__()
        # you create a feature extractor with
        self.feature_extractor = timm.create_model(
            backbone_name,
            out_indices=out_indices,
            features_only=True,
            pretrained=True,
            exportable=True,
        )
        ## kwargs is consider out_indices and features_only
        ## outindices passed to be (1,2,3,-1)
        # exportable Set layer config so that model is traceable / ONNX exportable
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        # you have a feature extracture that iss able to evaluate
        # no gradients
        self.feature_extractor.eval()

        self.pool = torch.nn.AdaptiveAvgPool2d(1) if pool_last else None
        self.backbone_name = backbone_name  # for results metadata
        self.out_indices = out_indices

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # move features extractor to device
        self.feature_extractor = self.feature_extractor.to(self.device)

    def extract(self, x: Tensor):
        '''
        Function that move the features to the cpu and returns the
        alll the feature maps and the last one
        :param x:
        :return:
        '''
        with torch.no_grad():
            # you extract the feature maps
            feature_maps = self.feature_extractor(x.to(self.device))
        # loop over features map and set them to cpu
        feature_maps = [fmap.to("cpu") for fmap in feature_maps]
        if self.pool is not None:
            # spit into fmaps and z
            # x is the last oneap
            # fmaps contains everything up to last one
            # last one you pass it by self.pool
            return feature_maps[:-1], self.pool(feature_maps[-1])
        else:
            # this is used for PADIM
            return feature_maps

    def fit(self, _: DataLoader):
        raise NotImplementedError

    def predict(self, _: Tensor):
        raise NotImplementedError

    def evaluate(self, test_dl: DataLoader) -> Tuple[float, float]:
        """Calls predict step for each test sample."""
        image_preds = []
        image_labels = []
        pixel_preds = []
        pixel_labels = []

        for sample, mask, label in tqdm(test_dl, **get_tqdm_params()):
            # This is the forward
            # you are goign over the test
            z_score, fmap = self.forward(sample)

            image_preds.append(z_score.numpy()) # the last layer
            image_labels.append(label)

            pixel_preds.extend(fmap.flatten().numpy())
            pixel_labels.extend(mask.flatten().numpy())

        image_labels = np.stack(image_labels)
        image_preds = np.stack(image_preds)
        #this is score oover labels
        # what you are doing in here is very interesting
        image_rocauc = roc_auc_score(image_labels, image_preds)
        pixel_rocauc = roc_auc_score(pixel_labels, pixel_preds)
        # we donot have the think in the paper of being anlomlious or not over teta variable
        return image_rocauc, pixel_rocauc

    def get_parameters(self, extra_params: dict = None) -> dict:
        return {
            "backbone_name": self.backbone_name,
            "out_indices": self.out_indices,
            **extra_params,
        }


class SPADE(KNNExtractor):
    def __init__(
        self,
        k: int = 5,
        backbone_name: str = "resnet18",
    ):
        # this is spade using the network passed
        super().__init__(
            backbone_name=backbone_name,
            out_indices=(1, 2, 3, -1),
            pool_last=True,
        )
        # setting pool last to True
        self.k = k
        self.image_size = 224
        # in here we have the aveerage pooling
        self.z_lib = []
        # in here we have the rest of the features
        # you have a list of lists in here
        # this is very important
        self.feature_maps = []
        self.threshold_z = None
        self.threshold_fmaps = None
        self.blur = NativeGaussianBlur()

    def fit(self, train_dl):
        # you are creating in this step the gallery of the features
        # you call the fit function at the first step
        # looping for the training stepWe construct a gallery of features
        # at all pixel locations of the K nearest neighbors
        #
        for sample, _ in tqdm(train_dl, **get_tqdm_params()):
            feature_maps, z = self.extract(sample)
            # you get the fucking feature maps for this current sample
            # this part after calling the child you get the Z
            # featureamaps is the heat map
            # z vector
            # the average pooling
            self.z_lib.append(z)
            # added the paramter after the pooling

            # feature maps
            if len(self.feature_maps) == 0:
                # if we are in the first itersation
                for fmap in feature_maps:
                    self.feature_maps.append([fmap])
                    # putting it into a list
            else:
                # indicating based on the idx the feature map
                for idx, fmap in enumerate(feature_maps):
                    self.feature_maps[idx].append(fmap)
        # stacking very thing
        self.z_lib = torch.vstack(self.z_lib)

        for idx, fmap in enumerate(self.feature_maps):
            self.feature_maps[idx] = torch.vstack(fmap)
        # i want to understand how the stacking really work

    def forward(self, sample):
        # I need to understand what is difference between this and the fit function

        feature_maps, z = self.extract(sample)
        # T
        # z is the averaged pooling
        # first feature map [1,256,56,56]
        # second feature map [1,512,28,28]
        # third feature map [1,1024,14,14]
        ##################################
        # z has the shape of [1,2048,1,1]
        # what does z_lib made of
        # [320,2048,1,1] from z_lib because i have 320 feature map
        # you subtract z in dim of 2048
        # you calculating the norm in here
        distances = torch.linalg.norm(self.z_lib - z, dim=1)
        # distances has the shape of [320,1,1] and this is the result
        values, indices = torch.topk(distances.squeeze(), self.k, largest=False)
        #distances.squeeze() get you the array of size 320
        # choose the best ones
        # you get back the top 50 k choices
        # values haas the shape of 50 you calculate the mean
        z_score = values.mean()
        # you calculate the mean of all the values for all the distances

        # you calcualte the mean which you call z_zore

        # Build the feature gallery out of the k nearest neighbours.
        # The authors migh have concatenated all features maps first, then check the minimum norm per pixel.
        # Here, we check for the minimum norm first, then concatenate (sum) in the final layer.
        scaled_s_map = torch.zeros(1, 1, self.image_size, self.image_size)
        for idx, fmap in enumerate(feature_maps):
            # self.feature_maps[idx] This is the main ideaa
            # [320, 256, 56, 56]
            # indices are the 50 nearest neightbour  of this current test set
            # training set
            # you select the nearest featuremaps
            # you choose based on channel 0
            nearest_fmaps = torch.index_select(self.feature_maps[idx], 0, indices)
            # min() because kappa=1 in the paper
            # nearest_fmaps has the shape of 50
            # fmap has shape of 1
            # you do the normal allong channel 1

            s_map, _ = torch.min(
                torch.linalg.norm(nearest_fmaps - fmap, dim=1), dim=0, keepdim=True
            )
            # when you do the subtraction you get something like
            # 50,56,56
            # you return the smallested one in the 50
            # indicating smallest distance
            # you subtract both
            # this is why you do the interpolation
            # becuase you can go up to the iamge size
            scaled_s_map += torch.nn.functional.interpolate(
                s_map.unsqueeze(0),
                size=(self.image_size, self.image_size),
                mode="bilinear",
            )

        scaled_s_map = self.blur(scaled_s_map)
        return z_score, scaled_s_map

    def get_parameters(self):
        # this is the extra paramters of the method
        # this way we can run the model and get the paramters if we want
        return super().get_parameters(
            {
                "k": self.k,
            }
        )

    def export(self, save_name: str):
        scripted_predictor = torch.jit.script(self)
        scripted_predictor.save(f"{EXPORT_DIR}/{save_name}.pt")

        tensor_x = torch.rand((1, 3, 224, 224), dtype=torch.float32)
        onnx_program = torch.onnx.dynamo_export(self, tensor_x)
        onnx_program.save(f"{EXPORT_DIR}/{save_name}.onnx")


class PaDiM(KNNExtractor):
    def __init__(
        self,
        d_reduced: int = 100,
        backbone_name: str = "resnet18",
    ):
        super().__init__(
            backbone_name=backbone_name,
            out_indices=(1, 2, 3),
        )# self.pool is none so he isnot using the fucking
        # adaptive pooling last layer
        # pool_last is set to none
        # you donot return the adatpive average pooling
        self.image_size = 224
        self.epsilon = 0.04  # cov regularization
        self.patch_lib = []
        self.resize = None
        # reduction parmas
        self.d_reduced = d_reduced  # your RAM will thank you
        self.r_indices = None  # this will get set in fit

    def fit(self, train_dl):
        # this gets called first
        for sample, _ in tqdm(train_dl, **get_tqdm_params()):
            feature_maps = self.extract(sample)
            # you are calling the extract method !!!
            # the extracted features maps
            if self.resize is None:
                # this is set to none in first iteration
                largest_fmap_size = feature_maps[0].shape[-2:]
                # This is the first one
                # this is the largest feature map you are getting spatial dimension whihc is 56 by 56
                # you are accessing the last 2 dimension indicated you are
                # accessing the spatial dimension
                self.resize = torch.nn.AdaptiveAvgPool2d(largest_fmap_size)
                # to know the size of the adaptive average pooling what doyou want to reach
                # using adptive average pooling with the size of largerest to resize
            # you are resizing the feature maps with the largest one
            resized_maps = [self.resize(fmap) for fmap in feature_maps]
            # you are concating all of them in the same place along channel dimenion
            resized_maps = torch.cat(resized_maps, 1)
            if resized_maps.shape[1] > self.d_reduced:
                if self.r_indices is None:
                    self.r_indices = torch.randperm(resized_maps.shape[1])[:self.d_reduced]
                resized_maps = resized_maps[:, self.r_indices, ...]
                # print(resized_maps.shape)
            self.patch_lib.append(resized_maps)
            # you concat everything alonmg the zero channel
            # they all have the same size of the biggest dimension which is
            #  1,256,56,56
            # you are concate all the 3 dimension all long dim 1
            # you have diffferent number of features maps
            # they all have the same size
            # due to the adaptive average pooling
        # concating everything allong batch dimension
        # so each teaching sample we have all this day
        self.patch_lib = torch.cat(self.patch_lib, 0)

        # random projection
        # . We noticed that randomly selecting few dimensions is more efficient that
        # a classic Principal Component Analysis (PCA) algorithm [

        # calcs
        # calculate the mean along batch dimension
        # you have the size of batch size ,features ,56,56
        # this is before the reduction
        self.means = torch.mean(self.patch_lib, dim=0, keepdim=True)
        # The original feature maps capture all the information from the extracted features.
        # By calculating the mean along the batch dimension, you're essentially creating a "reference"
        # for the normal (or training) data distribution. This will later help to identify deviations
        # from the normal distribution (i.e., anomalies).
        #
        # getting the means for the reduced
        x_ = self.patch_lib - self.means

        # cov calc
        # this is how we calcualte the converiance matrix???
        # The covariance matrix captures the relationship between different features in the reduced space.
        # It helps in understanding how different dimensions (features) of the data vary with each other.
        # Center the data: Subtract the mean from the data to ensure it's centered.
        # x_.permute([1, 0, 2, 3]): Transposes the first two dimensions of x_ so that the features
        # (across channels) can be multiplied.
        # torch.einsum("abkl,bckl->ackl", x_.permute([1, 0, 2, 3]), x_): This computes the outer product of
        # the centered features, which is how covariance is typically calculated.
        # The outer product represents the interaction between different dimensions (features),
        # leading to a matrix of covariances.
        #  The result is divided by the number of samples (self.patch_lib.shape[0] - 1)
        #  to account for the degrees of freedom in the data.
        self.E = (
            torch.einsum(
                "abkl,bckl->ackl",
                x_.permute([1, 0, 2, 3]),  # transpose first two dims
                x_,
            )
            * 1
            / (self.patch_lib.shape[0] - 1)
        )
        #
        # self d reduction = 350
        self.E += self.epsilon * torch.eye(self.d_reduced).unsqueeze(-1).unsqueeze(-1)
        # eye make matrix that has dignonal stuff only
        # o avoid singularity (when the covariance matrix is not invertible), a small value (epsilon * torch.eye(self.d_reduced))
        # is added to the diagonal elements of the covariance matrix. This ensures numerical stability.
        # torch.Size([350, 350, 1, 1])
        self.E_inv = torch.linalg.inv(self.E.permute([2, 3, 0, 1])).permute(
            [2, 3, 0, 1]
        )
        # Invert the covariance matrix: The inverse of the covariance matrix is often used in
        # anomaly detection to measure how far a sample is from the normal distribution in a
        # multivariate sense (e.g., using the Mahalanobis distance).

    def forward(self, sample):
        feature_maps = self.extract(sample)
        resized_maps = [self.resize(fmap) for fmap in feature_maps]
        fmap = torch.cat(resized_maps, 1)

        # reduce
        x_ = fmap[:, self.r_indices, ...] - self.means

        # we use the Mahalanobis distance [31] M (xij) to give an anomaly score to the patch in position
        # (i, j) of a test image. M (xij) can be interpreted as the distance between the test
        # patch embedding xij and learned distribution N (μij, Σij), where M (xij) is computed as follows:
        left = torch.einsum("abkl,bckl->ackl", x_, self.E_inv)
        s_map = torch.sqrt(torch.einsum("abkl,abkl->akl", left, x_))
        scaled_s_map = torch.nn.functional.interpolate(
            s_map.unsqueeze(0), size=(self.image_size, self.image_size), mode="bilinear"
        )

        return torch.max(s_map), scaled_s_map[0, ...]

    def get_parameters(self):
        return super().get_parameters(
            {
                "d_reduced": self.d_reduced,
                "epsilon": self.epsilon,
            }
        )


class PatchCore(KNNExtractor):
    def __init__(
        self,
        f_coreset: float = 0.01,  # fraction the number of training samples
        backbone_name: str = "resnet18",
        coreset_eps: float = 0.90,  # sparse projection parameter
    ):
        super().__init__(
            backbone_name=backbone_name,
            out_indices=(2, 3),
        )
        self.f_coreset = f_coreset
        self.coreset_eps = coreset_eps
        self.image_size = 224
        self.average = torch.nn.AvgPool2d(3, stride=1)
        self.blur = NativeGaussianBlur()
        self.n_reweight = 3

        self.patch_lib = []
        self.resize = None

    def fit(self, train_dl):
        for sample, _ in tqdm(train_dl, **get_tqdm_params()):
            feature_maps = self.extract(sample)

            if self.resize is None:
                self.largest_fmap_size = feature_maps[0].shape[-2:]
                self.resize = torch.nn.AdaptiveAvgPool2d(self.largest_fmap_size)
            resized_maps = [self.resize(self.average(fmap)) for fmap in feature_maps]
            patch = torch.cat(resized_maps, 1)
            patch = patch.reshape(patch.shape[1], -1).T

            self.patch_lib.append(patch)

        self.patch_lib = torch.cat(self.patch_lib, 0)

        if self.f_coreset < 1:
            self.coreset_idx = get_coreset_idx_randomp(
                self.patch_lib,
                n=int(self.f_coreset * self.patch_lib.shape[0]),
                eps=self.coreset_eps,
            )
            self.patch_lib = self.patch_lib[self.coreset_idx]

    def forward(self, sample):
        feature_maps = self.extract(sample)
        resized_maps = [self.resize(self.average(fmap)) for fmap in feature_maps]
        patch = torch.cat(resized_maps, 1)
        patch = patch.reshape(patch.shape[1], -1).T

        dist = torch.cdist(patch, self.patch_lib)
        min_val, min_idx = torch.min(dist, dim=1)

        # Instead of indexing with s_idx, use masked_select
        s_star, s_idx = torch.max(min_val, dim=0)

        # reweighting
        m_test = patch.select(0, s_idx).unsqueeze(0)
         # anomalous patch
        m_star = self.patch_lib[min_idx[s_idx]].unsqueeze(0)  # closest neighbour
        w_dist = torch.cdist(m_star, self.patch_lib)  # find knn to m_star pt.1
        _, nn_idx = torch.topk(w_dist, k=self.n_reweight, largest=False)  # pt.2
        # equation 7 from the paper
        m_star_knn = torch.linalg.norm(m_test - self.patch_lib[nn_idx[0, 1:]], dim=1)
        # Softmax normalization trick as in transformers.
        # As the patch vectors grow larger, their norm might differ a lot.
        # exp(norm) can give infinities.
        D = torch.sqrt(torch.tensor(patch.shape[1]))
        w = 1 - (torch.exp(s_star / D) / (torch.sum(torch.exp(m_star_knn / D))))
        s = w * s_star

        # segmentation map
        s_map = min_val.view(1, 1, *self.largest_fmap_size)
        s_map = torch.nn.functional.interpolate(
            s_map, size=(self.image_size, self.image_size), mode="bilinear"
        )
        s_map = self.blur(s_map)

        return s, s_map

    def get_parameters(self):
        return super().get_parameters(
            {
                "f_coreset": self.f_coreset,
                "n_reweight": self.n_reweight,
            }
        )

    def export(self, save_name: str):
        scripted_predictor = torch.jit.script(self)
        scripted_predictor.save(f"{EXPORT_DIR}/{save_name}.pt")

        # TODO: does not work yet
        # tensor_x = torch.rand((1, 3, 224, 224), dtype=torch.float32)
        # onnx_program = torch.onnx.dynamo_export(self, tensor_x)
        # onnx_program.save(f"{EXPORT_DIR}/{save_name}.onnx")
