import glob
import os
import random
import warnings
from PIL import Image
import torch
import torch.utils.data
import torchvision
import kornia

from datasets.transform import resize
from datasets.data_utils import (
    get_random_sampling_rate,
    tensor_normalize,
    spatial_sampling,
    pack_pathway_output,
)
from datasets.decoder import decode
from datasets.video_container import get_video_container
from datasets.transform import VideoDataAugmentationDINO
from datasets.kinetics import Kinetics
from einops import rearrange

if __name__ == "__main__":
    # import torch
    # from timesformer.datasets import Kinetics
    from utils.parser import parse_args, load_config
    from tqdm import tqdm

    args = parse_args()
    args.cfg_file = "/radraid/colivares/github_repos/svt/models/configs/Kinetics/TimeSformer_divST_8x32_224.yaml"
    config = load_config(args)
    config.DATA.PATH_TO_DATA_DIR = "/radraid/colivares/dsa_data/dsa_dfs"
    # config.DATA.PATH_TO_DATA_DIR = "/home/kanchanaranasinghe/data/kinetics400/k400-mini"
    config.DATA.PATH_PREFIX = "/radraid/colivares/dsa_data/dsa_images/nifti_raw_time"
    # dataset = Kinetics(cfg=config, mode="val", num_retries=10)
    dataset = Kinetics(cfg=config, mode="train", num_retries=10, get_flow=True)
    print(f"Loaded train dataset of length: {len(dataset)}")
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=4)
    for idx, i in enumerate(dataloader):
        print([x.shape for x in i[0]], i[1:3], [x.shape for x in i[3]["flow"]])
        break

    do_vis = False
    if do_vis:
        from PIL import Image
        from transform import undo_normalize

        vis_path = "/radraid/colivares/test"

        for aug_idx in range(len(i[0])):
            temp = i[0][aug_idx][3].permute(1, 2, 3, 0)
            temp = undo_normalize(
                temp, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )
            for idx in range(temp.shape[0]):
                im = Image.fromarray(temp[idx].numpy())
                im.resize((224, 224)).save(f"{vis_path}/aug_{aug_idx}_fr_{idx:02d}.jpg")
