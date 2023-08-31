#!/usr/bin/env python
script_description = """In this script, I just make a csv file that has paths to the
        patient NIfTIs and a random int between 0 and 200 to see what
        Kinetics class does with it."""

import os
import argparse
import pandas as pd
from typing import Union
import random

random.seed(42)


def your_function(arg1, arg2):
    """Description of your function goes here."""
    # Your function code here
    pass


def get_subdirectory_names(root_path: str):
    """_summary_

    _extended_summary_

    Args:
        root_path (str): _description_
    """
    all_subdirectories = os.listdir(root_path)
    pid_subdirectories = []
    for directory_name in all_subdirectories:
        if "_IR" in directory_name:
            pid_subdirectories.append(directory_name)
    return pid_subdirectories


class PidDirectory:
    """_summary_

    _extended_summary_
    """

    def __init__(self, path2pids):
        """_summary_

        _extended_summary_

        Args:
            path2pids (_type_): _description_
        """
        self.path2pids = path2pids
        self.pid_subdirectory_names = get_subdirectory_names(root_path=self.path2pids)

    def find_subdir_for(self, pid: Union[str, int]):
        """_summary_

        _extended_summary_

        Args:
            pid (Union[str, int]): _description_
        """
        for subdir in self.pid_subdirectory_names:
            if f"{pid}" in subdir:
                return subdir
        return None

    def compute_path_for(self, pid, filename):
        """_summary_

        _extended_summary_

        Args:
            pid (_type_): _description_
            filename (_type_): _description_
        """
        subdirectory = self.find_subdir_for(pid)
        if not subdirectory:
            raise ValueError("no subdirectory found for pid:{pid}")
        return os.path.join(self.path2pids, subdirectory, filename)


def main():
    parser = argparse.ArgumentParser(description=script_description)
    parser.add_argument(
        "--dsa_data_table",
        "-dsa",
        type=str,
        help="Path to a csv that has the data on the niftis.",
    )
    parser.add_argument(
        "--path2data", "-p2d", type=str, help="Path to where the NIfTI files are stored"
    )
    parser.add_argument(
        "--savepath",
        "-sv",
        type=str,
        help="Where you want to save the created csv to",
    )
    args = parser.parse_args()
    dsa_dataframe = pd.read_csv(args.dsa_data_table)
    pid_directory = PidDirectory(path2pids=args.path2data)
    video_paths = []
    dummy_labels = []
    for _, row in dsa_dataframe.iterrows():
        video_paths.append(
            pid_directory.compute_path_for(pid=row["PID"], filename=row["filename"])
        )
        dummy_labels.append(random.randint(a=1, b=200))
    kinetics_compatible_csv = pd.DataFrame(
        {"path_to_video": video_paths, "label": dummy_labels}
    )
    print(kinetics_compatible_csv.head(10))
    kinetics_compatible_csv.to_csv(
        "kinetics_format_dsa_data.csv", index=False, header=False, sep=" "
    )


if __name__ == "__main__":
    main()
