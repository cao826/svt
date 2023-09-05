import os
import pdb
import argparse
from tqdm import tqdm, trange
import numpy as np
import nibabel as nib
import av


## these functions are good for utils


def compute_pid_output_filename(path2video: str) -> str:
    """NOTE: This function assumes that the PID is the
    directory name of the path, e.g:
    a/long/path/470001/filename.nifti

    Args:
        path2video (str): _description_

    Returns:
        output_filename (str): _description_
    """
    pdb.set_trace()
    pid = os.path.dirname(path2video).split("/")[-1]
    filename = os.path.basename(path2video)
    filename = filename.replace(".nii.gz", ".mp4")
    output_filename = f"{pid}__{filename}"
    return output_filename


def normalize_to_grayscale(image: np.ndarray, debug: bool = False) -> np.ndarray:
    """_summary_

    Args:
        image (np.ndarray): _description_
        debug (bool, optional): _description_. Defaults to False.

    Returns:
        np.ndarray: _description_
    """
    normed_image = (image.astype(np.float32) / 1024.0) * 255.0
    return normed_image.astype(np.uint8)  # [:, :, np.newaxis]


def np_grayscale_to_mp4(
    nifti_video: np.ndarray, savepath: str, debug: bool = False
) -> None:
    """_summary_

    Args:
        nifti_video (np.ndarray): _description_
        debug (bool, optional): _description_. Defaults to False.
    """
    # this assumes n_frames is last
    n_rows, n_columns, n_frames = nifti_video.shape
    # this is a hardcoded value that I am basing on the literature.
    # make sure to cite the paper that mentions this
    fps = 3
    duration = n_frames / fps
    if debug:
        print(savepath)
    try:
        container = av.open(savepath, mode="w")

        stream = container.add_stream(codec_name="h264", rate=fps)
        stream.width = n_columns
        stream.height = n_rows
        stream.pix_fmt = "gray"
        for frame_index in trange(n_frames):
            frame_array = normalize_to_grayscale(
                image=nifti_video[:, :, frame_index], debug=debug
            )
            frame = av.VideoFrame.from_ndarray(array=frame_array, format=stream.pix_fmt)
            for packet in stream.encode(frame):
                container.mux(packet)

        # flush the stream
        for packet in stream.encode():
            container.mux(packet)
        # close the file
        container.close()
    except:
        print("huh")


def convert_nifti_file_to_mp4(
    path2nifti: str, savepath: str, debug: bool = False
) -> None:
    """_summary_

    Args:
        path2nifti (str): _description_

    Returns:
        av.container.Container: _description_
    """
    pdb.set_trace()
    nifti_file = nib.load(path2nifti)
    nifti_video = nifti_file.get_fdata()
    if debug:
        print(f"shape of read in video {nifti_video.shape}")
    save_filename = compute_pid_output_filename(path2nifti)
    output_path = os.path.join(savepath, save_filename)
    np_grayscale_to_mp4(nifti_video=nifti_video, savepath=output_path, debug=debug)


def parse_args() -> argparse.Namespace:
    """_summary_

    Returns:
        argparse.Namespace: _description_
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path2niftis", type=str, help="Path to where the NIfTI files are."
    )
    parser.add_argument(
        "--savepath",
        "-savepath",
        "-sv",
        type=str,
        help="Path where you want to save the pyav containers to (if that is a thing)",
    )
    parser.add_argument("--debug", "-debug", "-d", action="store_true")
    args = parser.parse_args()
    return args


def main(args: argparse.Namespace) -> None:
    """_summary_

    Args:
        args (argparse.Namespace): _description_
    """
    convert_nifti_file_to_mp4(
        path2nifti=args.path2niftis, savepath=args.savepath, debug=args.debug
    )


if __name__ == "__main__":
    args = parse_args()
    main(args)
