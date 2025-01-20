import h5py as h5
import numpy as np
from torch.utils.data import Dataset

from utils.utils_functions import c2r
from utils.utils_fourier import *
from utils.utils_mask import define_Mask

from scipy.fftpack import *

class Loader(Dataset):
    def __init__(self, mode, dataset_path, mask, sigma=0.01):
        """
        :sigma: std of Gaussian noise to be added in the k-space
        """
        self.prefix = 'trn' if mode == 'train' else 'tst'
        self.dataset_path = dataset_path
        self.mask = mask
        self.sigma = sigma

        with h5.File(self.dataset_path, 'r') as f:
            self.num_data = len(f[self.prefix+'Org'])  # Total number of samples

    def __getitem__(self, index):
        """
        Args.
            index: Data index
        Returns:
            x0: undersampled reconstructed image (2, H, W) - float32
            gt: original image (2, H, W) - float32
            mask: sampling mask (H, W) - float32
        """

        with h5.File(self.dataset_path, 'r') as f:
            # Read reconstruction rss image (H, W) - float32
            gt = f[self.prefix+'Org'][index]

        if 'fMRI' in self.mask:
            mask_1d = define_Mask(self.mask)
            mask_1d = mask_1d[:, np.newaxis]
            mask = np.repeat(mask_1d, gt.shape[0], axis=1).transpose((1, 0))
            self.mask = mask # (H, W)

        # normalisation and undersampled with noise
        gt_image = preprocess_normalisation(gt)
        x0 = undersample_kspace(gt_image, self.mask)

        # Convert to torch tensor and split real part imaginary part
        x0 = torch.from_numpy(c2r(x0))              # (2, H, W)
        gt = torch.from_numpy(c2r(gt_image))        # (H, W)
        mask = torch.from_numpy(self.mask.astype(np.float32)) # (H, W)

        return x0, gt, mask

    def __len__(self):

        return self.num_data

def undersample_kspace(x, mask):
    if isinstance(x, torch.Tensor):
        x = x.numpy()
    if isinstance(mask, torch.Tensor):
        mask = mask.numpy()

    fft = fft2(x)
    fft = fftshift(fft)
    fft = fft * mask

    # if is_noise:
    #     raise NotImplementedError
    #     fft = fft + generate_gaussian_noise(fft, noise_level, noise_var)

    fft = ifftshift(fft)
    x = ifft2(fft)

    return x

def preprocess_normalisation(img):

    img = img / abs(img).max()

    return img