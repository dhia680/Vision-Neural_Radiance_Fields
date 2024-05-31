import numpy as np
import math
import torch
from config import *
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.pyplot as plt


data = np.load("tiny_nerf_data.npz")
images = data["images"]
poses = data["poses"]
focal = float(data["focal"])

_, image_height, image_width, _ = images.shape

train_images, train_poses = images[:100], poses[:100]
val_image, val_pose = images[101], poses[101]

cx, cy = 0., 0.
f = focal

camera_instrinsics = {
    'cx': cx,
    'cy': cy,
    'f': f
}

print("Train Images Shape:", train_images.shape)
print("Train Poses Shape:", train_poses.shape)
print("Validation Image Shape:", val_image.shape)
print("Validation Pose Shape:", val_pose.shape)

print(f"Images shape: {images.shape}")
print(f"Poses shape: {poses.shape}")
print(f"Focal value: {focal:.5f}")

print('x-fov:', 2 * math.atan(image_width / (2 * f)) * 180 / math.pi)
print('y-fov:', 2 * math.atan(image_height / (2 * f)) * 180 / math.pi)

train_images = torch.tensor(train_images, device = device, dtype = dtype)
train_poses = torch.tensor(train_poses, device = device, dtype = dtype)
val_image = torch.tensor(val_image, device = device, dtype = dtype)
val_pose = torch.tensor(val_pose, device = device, dtype = dtype)

def show_dataset(figsize=(16, 16)):
    fig = plt.figure(figsize=figsize)
    grid = ImageGrid(fig, 111, nrows_ncols=(4, 4), axes_pad=0.1)
    random_images = train_images[np.random.choice(np.arange(train_images.shape[0]), 16)]
    for ax, image in zip(grid, random_images):#.cpu().numpy()
        ax.imshow(image.cpu().numpy())
    plt.title("Sample Images from Tiny-NeRF Data")
    plt.show()