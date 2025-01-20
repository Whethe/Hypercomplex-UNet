import h5py
from PIL import Image
from scipy.fftpack import *

mask = 'fMRI_Reg_AF8_CF0.04_PE320'
path = "mri_slice_20.png"

import numpy as np

# def fft_new(image, ndim, normalized=False):
#     norm = "ortho" if normalized else None
#     dims = tuple(range(-ndim, 0))
#
#     image = torch.view_as_real(
#         torch.fft.fftn(  # type: ignore
#             torch.view_as_complex(image.contiguous()), dim=dims, norm=norm
#         )
#     )
#     return image
#
#
# def ifft_new(image, ndim, normalized=False):
#     norm = "ortho" if normalized else None
#     dims = tuple(range(-ndim, 0))
#     image = torch.view_as_real(
#         torch.fft.ifftn(  # type: ignore
#             torch.view_as_complex(image.contiguous()), dim=dims, norm=norm
#         )
#     )
#     return image
#
# def fftshift(x, dim=None):
#     """
#     Similar to np.fft.fftshift but applies to PyTorch Tensors
#     """
#     if dim is None:
#         dim = tuple(range(x.dim()))
#         shift = [dim // 2 for dim in x.shape]
#     elif isinstance(dim, int):
#         shift = x.shape[dim] // 2
#     else:
#         shift = [x.shape[i] // 2 for i in dim]
#     return roll(x, shift, dim)
#
#
# def ifftshift(x, dim=None):
#     """
#     Similar to np.fft.ifftshift but applies to PyTorch Tensors
#     """
#     if dim is None:
#         dim = tuple(range(x.dim()))
#         shift = [(dim + 1) // 2 for dim in x.shape]
#     elif isinstance(dim, int):
#         shift = (x.shape[dim] + 1) // 2
#     else:
#         shift = [(x.shape[i] + 1) // 2 for i in dim]
#     return roll(x, shift, dim)
#
# def roll(x, shift, dim):
#     """
#     Similar to np.roll but applies to PyTorch Tensors
#     """
#     if isinstance(shift, (tuple, list)):
#         assert len(shift) == len(dim)
#         for s, d in zip(shift, dim):
#             x = roll(x, s, d)
#         return x
#     shift = shift % x.size(dim)
#     if shift == 0:
#         return x
#     left = x.narrow(dim, 0, x.size(dim) - shift)
#     right = x.narrow(dim, x.size(dim) - shift, shift)
#     return torch.cat((right, left), dim=dim)
#
# def fft2(data):
#     """
#     Apply centered 2 dimensional Fast Fourier Transform.
#     Args:
#         data (torch.Tensor): Complex valued input data containing at least 3 dimensions: dimensions
#             -3 & -2 are spatial dimensions and dimension -1 has size 2. All other dimensions are
#             assumed to be batch dimensions.
#     Returns:
#         torch.Tensor: The FFT of the input.
#     """
#     assert data.size(-1) == 2
#     data = ifftshift(data, dim=(-3, -2))
#     data = fft_new(data, 2, normalized=True)
#     data = fftshift(data, dim=(-3, -2))
#     return data
#
# def ifft2(data):
#     """
#     Apply centered 2-dimensional Inverse Fast Fourier Transform.
#     Args:
#         data (torch.Tensor): Complex valued input data containing at least 3 dimensions: dimensions
#             -3 & -2 are spatial dimensions and dimension -1 has size 2. All other dimensions are
#             assumed to be batch dimensions.
#     Returns:
#         torch.Tensor: The IFFT of the input.
#     """
#     assert data.size(-1) == 2
#     data = ifftshift(data, dim=(-3, -2))
#     data = ifft_new(data, 2, normalized=True)
#     data = fftshift(data, dim=(-3, -2))
#     return data

def undersample_kspace(x, mask):
    fft = fft2(x)
    fft = fftshift(fft)
    fft = fft * mask

    # if is_noise:
    #     raise NotImplementedError
    #     fft = fft + generate_gaussian_noise(fft, noise_level, noise_var)

    fft = ifftshift(fft)
    x = ifft2(fft)

    return x

def save_image(image, output_path):
    """Save the real part of the image as a grayscale image using PIL."""
    # make sure data type is float32
    image = image.astype(np.float32)

    # normalize image to (0-255)
    image_normalized = ((image - np.min(image)) / (np.max(image) - np.min(image)) * 255).astype(np.uint8)

    pil_image = Image.fromarray(image_normalized)

    pil_image.save(output_path)
    print(f"Image saved to {output_path}")

hf = h5py.File('./file1000000.h5', 'r')
print('Keys:', list(hf.keys()))
gt = hf['reconstruction_rss'][20]

print(gt.dtype)
print(gt.shape)


gt_image = gt / abs(gt).max()

if 'fMRI' in mask:
    mask_1d = np.load("regular_af8_cf0.04_pe320.npy")
    mask_1d = mask_1d[:, np.newaxis]
    mask = np.repeat(mask_1d, gt_image.shape[0], axis=1).transpose((1, 0)) # (H, W)

masked_gt = undersample_kspace(gt_image, mask)

print(masked_gt.dtype)

save_image(masked_gt, path)

# image = hf["images"][20]
#
# save_image(image, path)

# import h5py
# import numpy as np
#
# # 打开原始HDF5文件
# with h5py.File('/media/NAS06/haosen/knee_path/multicoil_dataset/merged_dataset.h5', 'r') as hf:
#     # 读取所有图像
#     all_images = hf["images"][:]
#
#     # 确保数据类型为float32
#     all_images = all_images.astype(np.float32)
#
#     # 划分数据集
#     train_images = all_images[:1000]
#     val_images = all_images[1000:]
#
# # 创建新的HDF5文件并保存划分后的数据集
# with h5py.File('mri_dataset_split.h5', 'w') as new_hf:
#     # 保存训练集（Ground Truth）
#     new_hf.create_dataset("GT", data=train_images)
#
#     # 保存验证集
#     new_hf.create_dataset("val", data=val_images)
#
# print(f"数据集已保存到 'mri_dataset.h5' 文件")
# print(f"训练集（GT）：{train_images.shape}")
# print(f"验证集（val）：{val_images.shape}")
# import h5py
#
# # 打开原始的 H5 文件
# with h5py.File('dataset.h5', 'r') as old_file:
#     # 创建新的 H5 文件
#     with h5py.File('mri_dataset_renamed.h5', 'w') as new_file:
#         # 复制 'GT' 数据集到 'trnOrg'
#         new_file.create_dataset('trnOrg', data=old_file['GT'])
#
#         # 复制 'val' 数据集到 'tstOrg'
#         new_file.create_dataset('tstOrg', data=old_file['val'])
#
# print("数据集已重命名并保存到 'mri_dataset_renamed.h5'")
#
# # 验证新文件
# with h5py.File('mri_dataset_renamed.h5', 'r') as f:
#     print("新文件中的数据集:")
#     print(list(f.keys()))
#     print("trnOrg shape:", f['trnOrg'].shape)
#     print("tstOrg shape:", f['tstOrg'].shape)