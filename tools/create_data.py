from os import path
import argparse

from data_converter import nuscenes_converter as nuscenes_converter
from data_converter.create_gt_database import create_groundtruth_database



def parse_args():
    parser = argparse.ArgumentParser(description="Data converter arg parser")
    parser.add_argument("dataset", metavar="kitti", help="name of the dataset")
    parser.add_argument(
        "--root-path",
        type=str,
        default="./data/nuscenes",
        help="specify the root path of dataset",
    )
    parser.add_argument(
        "--version",
        type=str,
        default="v1.0",
        required=False,
        help="specify the dataset version, no need for kitti",
    )
    parser.add_argument(
        "--max-sweeps",
        type=int,
        default=10,
        required=False,
        help="specify sweeps of lidar per example",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        required=False,
        help="name of info pkl",
    )
    parser.add_argument("--extra-tag", type=str, default="kitti")
    parser.add_argument("--painted", default=False, action="store_true")
    parser.add_argument(
        "--workers", type=int, default=4, help="number of threads to be used"
    )
    
    return parser.parse_args()


def nuscenes_data_prep(
    dataset_root_path,
    can_bus_root_path,
    info_prefix,
    version,
    dataset_name,
    out_dir,
    max_sweeps=10,
    load_augmented=None,
):
    """Prepare data related to nuScenes dataset.

    Related data consists of '.pkl' files recording basic infos,
    2D annotations and groundtruth database.

    Args:
        dataset_root_path (str): Path of dataset root.
        info_prefix (str): The prefix of info filenames.
        version (str): Dataset version.
        dataset_name (str): The dataset class name.
        out_dir (str): Output directory of the groundtruth database info.
        max_sweeps (int): Number of input consecutive frames. Default: 10
    """
    print(f"Version : {version}")
    nuscenes_converter.create_nuscenes_infos(
        dataset_root_path, out_dir, can_bus_root_path, 
        info_prefix, version=version, max_sweeps=max_sweeps
    )

    if version == 'v1.0-test':
        info_test_path = path.join(
            out_dir, f'{info_prefix}_infos_temporal_test.pkl')
        nuscenes_converter.export_2d_annotation(
            dataset_root_path, info_test_path, version=version)
    else:
        info_train_path = path.join(
            out_dir, f'{info_prefix}_infos_temporal_train.pkl')
        info_val_path = path.join(
            out_dir, f'{info_prefix}_infos_temporal_val.pkl')
        nuscenes_converter.export_2d_annotation(
            dataset_root_path, info_train_path, version=version)
        nuscenes_converter.export_2d_annotation(
            dataset_root_path, info_val_path, version=version)

def main():
    args = parse_args()    
    
    if args.dataset != "nuscenes": 
        print(f"Unsupported dataset: {args.dataset}.")
        return
            
    args.out_dir = args.root_path if args.out_dir is None else args.out_dir

    if args.dataset == "nuscenes" and args.version != "v1.0-mini":
        train_version = f"{args.version}-trainval"
        nuscenes_data_prep(
            dataset_root_path=path.join(args.root_path, args.dataset),
            can_bus_root_path=args.root_path,
            info_prefix=args.extra_tag,
            version=train_version,
            dataset_name="NuScenesDataset",
            out_dir=args.out_dir,
            max_sweeps=args.max_sweeps,
        )
        # test_version = f"{args.version}-test"
        # nuscenes_data_prep(
        #     dataset_root_path=path.join(args.root_path, args.dataset),
        #     can_bus_root_path=args.root_path,
        #     info_prefix=args.extra_tag,
        #     version=test_version,
        #     dataset_name="NuScenesDataset",
        #     out_dir=args.out_dir,
        #     max_sweeps=args.max_sweeps,
        # )
    elif args.dataset == "nuscenes" and args.version == "v1.0-mini":
        train_version = f"{args.version}"
        nuscenes_data_prep(
            dataset_root_path=path.join(args.root_path, args.dataset),
            can_bus_root_path=args.root_path,
            info_prefix=args.extra_tag,
            version=train_version,
            dataset_name="NuScenesDataset",
            out_dir=args.out_dir,
            max_sweeps=args.max_sweeps,
        )
    

if __name__ == "__main__":
    main()
