"""
Scratch Paper Script: dsa ssl preprocessing

Here I am trying to learn as much as possible about 
the dsa image loading and preprocessing
while developing the dataloader for this project.


Most the code here is inspired by:
https://betterprogramming.pub/python-case-study-decorators-and-pipelines-96b950289aa2

Usage:
    $ python scratch_paper.py
"""

# Import necessary modules
import os
import typing
import functools
import pdb
import random
import argparse
import numpy as np
import cv2
import torch
import nibabel as nib
from skimage.filters import threshold_multiotsu


def load_image(nifti_path, debug: bool = False):
    """
    loads nifti file to ram.
    returns it as a numpy array
    """
    nifti = nib.load(nifti_path)
    if debug:
        print(nifti.header)
    image = nifti.get_fdata().T
    return image


Transformation = typing.Callable[[np.ndarray], np.ndarray]
pipeline = []


# define decorator factory
# (callable that produces the actual decorator, so we can configure it)
def preprocessing_step(f: Transformation) -> Transformation:
    """ """

    @functools.wraps(f)
    def wrapper(arr: np.ndarray, **kwargs) -> np.ndarray:
        # print(f.__name__)
        return f(arr, **kwargs)

    pipeline.append(wrapper)
    return wrapper


# @preprocessing_step
def trim_frames(image: np.ndarray):
    """
    Returns only the last 15 frames of a video
    """
    if image.shape[0] > 15:
        return image[-15:]
    return image


# @preprocessing_step  # decide whether we really need this
def crop_image(image: np.ndarray, tol=0.01) -> np.ndarray:
    """removes bars (with noise) on image edges. tol is the amount of noise allowed.
    args:
        tol: tolerance value above which axis are accepted to contain image information. If a whole row
                contains no information and is at the image edge it is assumed to contain no value for
                classifiation
    """
    # compute standard deviation for each frame?
    mask = np.std(image, axis=0) > tol

    # create a meshgrid where image has significant info
    idx = np.ix_(
        mask.any(axis=1),  # columns iwth significant image info
        mask.any(axis=0),  # rows with significant image info
    )
    return image[:, idx[0], idx[1]]  # I have no idea what this function does


def resize_frame(frame: np.ndarray) -> np.ndarray:
    """_summary_

    _extended_summary_

    Args:
        frame (np.ndarray): _description_

    Raises:
        Exception: _description_

    Returns:
        np.ndarray: _description_
    """
    normed_frame, og_max, og_min = normalize_image_range(frame)
    resized_frame = cv2.resize(normed_frame, (512, 512), interpolation=cv2.INTER_CUBIC)
    resize_frame = denormalize_image_range(
        normalized_image=resized_frame, original_min=og_min, original_max=og_max
    )
    return resized_frame


# @preprocessing_step
def resize_video(image: np.ndarray):
    """ """
    num_frames = len(image)
    frame_shape = (512, 512)
    resized_image = np.zeros((num_frames,) + frame_shape)
    for idx, frame in enumerate(image):
        resized_image[idx] = cv2.resize(
            frame, frame_shape, interpolation=cv2.INTER_CUBIC
        )
    return resized_image


# @preprocessing_step
def clip_image(image: np.ndarray, boundaries=None, mode="multiotsu") -> np.ndarray:
    """clips image values to an interval based on image intensities.
    Available are multiotsu thresholding, and above median
    args:
        boundaries: image is clipped by those fixed values, must be in the format(min,max). If mode is given,
                    boundaries are ignored.
        mode: must be either mutlitotsu or median. If multiotsu boundaries are based on two thresholds. If median
              the upper boundary is the median value and lower the min value of the image
    """
    # pdb.set_trace()
    if mode == "multiotsu":
        boundaries = threshold_multiotsu(image, classes=3)
    elif mode == "median":
        boundaries = [image.min(), None]
        boundaries[1] = np.median(image)
    elif not boundaries and not mode:
        boundaries = [image.min(), image.max()]
    clipped = np.clip(image, boundaries[0], boundaries[-1])
    if clipped.shape[1] != 512:
        raise Exception("shape has changed")
    return clipped


def normalize_image_range(image: np.ndarray):
    """
    not needed if we are using the torchxravidsion pretrained models
    """
    image_min = np.min(image)
    image_max = np.max(image)
    return (image - image_min) / (image_max - image_min), image_max, image_min


def denormalize_image_range(
    normalized_image: np.ndarray, original_min: float, original_max: float
):
    """
    Inverse of the normalize_image_range function.
    """
    return normalized_image * (original_max - original_min) + original_min


# #@preprocessing_step # I think this was causing the issue. a lot of values get sent to > 0 here
# and we all know what relu does to negative values
def normalize_for_torchxrayvision(image: np.ndarray):
    """ """
    image_max = image.max()
    image_min = image.min()
    image = 2 * (image - image_min) / (image_max - image_min) - 1
    image = image * 1024
    return image


# @preprocessing_step
def reshape_for_pretrained_resnet(image: np.ndarray):
    """ """
    return np.expand_dims(image, axis=1)


def subtract_temporal_mean(image: np.ndarray):
    """ """
    return image - np.mean(image, axis=0)


def apply_pipeline(image: np.ndarray, debug: bool = False) -> np.ndarray:
    """ """
    if debug:
        return functools.reduce(
            lambda image, func: generate_array_report(func(image)), pipeline, image
        )
    else:
        return functools.reduce(lambda image, func: func(image), pipeline, image)


def generate_array_report(arr: np.ndarray):
    print("Array Information Report:")
    print("-------------------------")
    print(f"Data Type: {arr.dtype}")
    print(f"Shape: {arr.shape}")
    print(f"Number of Dimensions: {arr.ndim}")
    print(f"Total Number of Elements: {arr.size}")
    print(f"Memory Usage: {arr.nbytes} bytes")
    print(f"Minimum Value: {np.min(arr)}")
    print(f"Maximum Value: {np.max(arr)}")
    print(f"Mean: {np.mean(arr)}")
    print(f"Standard Deviation: {np.std(arr)}")
    return arr


def shuffle_list(original_lst):
    """
    This format forces all four elements to
    be permuted.
    Credit to:
    https://stackoverflow.com/questions/15512058/python-shuffle-such-that-position-will-never-repeat
    """
    shuffled_lst = original_lst.copy()
    while True:
        random.shuffle(shuffled_lst)
        for a, b in zip(original_lst, shuffled_lst):
            if a == b:
                break
        else:
            return shuffled_lst


def create_shuffling_map(n_frames, n_frames_to_shuffle):
    """ """
    frames_to_shuffle = random.sample(range(n_frames), n_frames_to_shuffle)
    shuffled_frames = shuffle_list(frames_to_shuffle)
    mapping = dict()
    for i in range(n_frames_to_shuffle):
        mapping[frames_to_shuffle[i]] = shuffled_frames[i]
    return mapping


def create_shuffled_index(n_frames, n_frames_to_shuffle):
    """ """
    shuffle_mapping = create_shuffling_map(n_frames, n_frames_to_shuffle)
    shuffled_index = []
    for i in range(n_frames):
        if i in shuffle_mapping.keys():
            shuffled_index.append(shuffle_mapping[i])
        else:
            shuffled_index.append(i)
    return shuffled_index, shuffle_mapping


def shuffle_video(video: np.ndarray, n_frames_to_shuffle: int):
    """ """
    # assumes video is of shape (n_frames, H, W)
    # pdb.set_trace()
    n_frames = video.shape[0]
    shuffled_index, shuffle_mapping = create_shuffled_index(
        n_frames=n_frames, n_frames_to_shuffle=n_frames_to_shuffle
    )
    return video[shuffled_index], shuffle_mapping


def prepare_video(path2nifti, debug: bool = False):
    """ """
    video = load_image(path2nifti)
    if debug:
        generate_array_report(arr=video)
    video = apply_pipeline(image=video, debug=debug)
    shuffled_video, shuffle_mapping = shuffle_video(video=video, n_frames_to_shuffle=4)
    return torch.tensor(shuffled_video, dtype=torch.float32), shuffle_mapping


def parse_args():
    """ """
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", type=bool)
    parser.add_argument("--path2example")
    return parser.parse_args()


def main(args):
    """
    This function does something interesting.
    """
    video, shuffle_mapping = prepare_video(args.path2example, debug=args.debug)
    for key, value in shuffle_mapping.items():
        print(f"frame {key} shuffled to -> {value}")


# Main entry point of the script
if __name__ == "__main__":
    # Your code experimentation starts here
    print("Welcome to the scratch paper!")

    main(parse_args())
