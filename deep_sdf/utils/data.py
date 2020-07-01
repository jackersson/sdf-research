
import numpy as np
import random
import glob

import typing as typ

import torch
import torch.utils.data

# Initial implementation:
# https://github.com/facebookresearch/DeepSDF/blob/master/deep_sdf/data.py


def list_files(folder: str, file_format: str = '.obj') -> typ.List[str]:
    pattern = '{0}/*{1}'.format(folder, file_format)
    filenames = glob.glob(pattern)
    return filenames


def remove_nans(tensor: torch.Tensor) -> torch.Tensor:
    """Check id sdf distance is not None"""
    tensor_nan = torch.isnan(tensor[:, 3])
    return tensor[~tensor_nan, :]


def unpack_sdf_samples(filename: str, subsample: int = None):
    npz = np.load(filename)
    if subsample is None:
        return npz
    pos_tensor = remove_nans(torch.from_numpy(npz["pos"]))
    neg_tensor = remove_nans(torch.from_numpy(npz["neg"]))

    # split the sample into half
    half = int(subsample / 2)

    # take half positive and negative samples
    random_pos = (torch.rand(half) * pos_tensor.shape[0]).long()
    random_neg = (torch.rand(half) * neg_tensor.shape[0]).long()

    sample_pos = torch.index_select(pos_tensor, 0, random_pos)
    sample_neg = torch.index_select(neg_tensor, 0, random_neg)

    samples = torch.cat([sample_pos, sample_neg], 0)

    return samples


def unpack_sdf_samples_from_ram(data: typ.Tuple[torch.Tensor, torch.Tensor], subsample: int = None):
    if subsample is None:
        return data
    pos_tensor = data[0]
    neg_tensor = data[1]

    # split the sample into half
    half = int(subsample / 2)

    pos_size = pos_tensor.shape[0]
    neg_size = neg_tensor.shape[0]

    pos_start_ind = random.randint(0, pos_size - half)
    sample_pos = pos_tensor[pos_start_ind: (pos_start_ind + half)]

    if neg_size <= half:
        random_neg = (torch.rand(half) * neg_tensor.shape[0]).long()
        sample_neg = torch.index_select(neg_tensor, 0, random_neg)
    else:
        neg_start_ind = random.randint(0, neg_size - half)
        sample_neg = neg_tensor[neg_start_ind: (neg_start_ind + half)]

    samples = torch.cat([sample_pos, sample_neg], 0)

    return samples


class SDFSamples(torch.utils.data.Dataset):
    # Documentation:
    # https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset
    def __init__(
        self,
        filenames: typ.List[str],
        subsample: int = 16384,
        load_ram: bool = False,
    ):
        self.subsample = subsample
        self.npyfiles = filenames

        self.load_ram = load_ram

        if load_ram:
            self.loaded_data = []
            for f in self.npyfiles:
                npz = np.load(f)
                # negative, when sdf <= 0
                # positive, when sdf > 0
                # Look: PreprocessMesh.cpp
                pos_tensor = remove_nans(torch.from_numpy(npz["pos"]))
                neg_tensor = remove_nans(torch.from_numpy(npz["neg"]))
                self.loaded_data.append(
                    [
                        pos_tensor[torch.randperm(pos_tensor.shape[0])],
                        neg_tensor[torch.randperm(neg_tensor.shape[0])],
                    ]
                )

    def __len__(self):
        return len(self.npyfiles)

    def __getitem__(self, idx):
        filename = self.npyfiles[idx]
        if self.load_ram:
            return (
                unpack_sdf_samples_from_ram(self.loaded_data[idx], self.subsample),
                idx,
            )
        else:
            return unpack_sdf_samples(filename, self.subsample), idx
