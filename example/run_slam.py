import argparse
import csv
import os.path
import subprocess

euroc_data_dir = "/mnt/data/home/hsiaochuan/data/euroc"
euroc_seqs = [
    "machine_hall/MH_01_easy/MH_01_easy/mav0",
    "machine_hall/MH_02_easy/MH_02_easy/mav0",
    "machine_hall/MH_03_medium/MH_03_medium/mav0",
    "machine_hall/MH_04_difficult/MH_04_difficult/mav0",
    "machine_hall/MH_05_difficult/MH_05_difficult/mav0",
    "vicon_room1/V1_01_easy/V1_01_easy/mav0",
    # "vicon_room1/V1_02_medium/V1_02_medium/mav0",
    "vicon_room1/V1_03_difficult/V1_03_difficult/mav0",
    "vicon_room2/V2_01_easy/V2_01_easy/mav0",
    "vicon_room2/V2_02_medium/V2_02_medium/mav0",
    "vicon_room2/V2_03_difficult/V2_03_difficult/mav0",
]
tum_rgbd_dir = "/mnt/data/home/hsiaochuan/data/tum_rgbd"
tum_rgbd_seqs = [
    "rgbd_dataset_freiburg1_desk",
    "rgbd_dataset_freiburg1_desk2",
]
SLAM_EXE = "../build/run_slam"
VOCAB_PATH = "./orb_vocab.fbow"

DATASETS = {
    "euroc": {
        "data_dir": euroc_data_dir,
        "sequences": euroc_seqs,
        "dataset_type": "euroc",
    },
    "tum_rgbd": {
        "data_dir": tum_rgbd_dir,
        "sequences": tum_rgbd_seqs,
        "dataset_type": "tum_rgbd",
    },
}

PRESETS = {
    "euroc_mono": {
        "dataset": "euroc",
        "config_fname": "./euroc/EuRoC_mono.yaml",
    },
    "euroc_stereo": {
        "dataset": "euroc",
        "config_fname": "./euroc/EuRoC_stereo.yaml",
    },
    "tum_mono": {
        "dataset": "tum_rgbd",
        "config_fname": "./tum_rgbd/TUM_RGBD_mono_1.yaml",
    },
    "tum_rgbd": {
        "dataset": "tum_rgbd",
        "config_fname": "./tum_rgbd/TUM_RGBD_rgbd_1.yaml",
    },
}

PRESET_ORDER = ["euroc_mono", "euroc_stereo", "tum_mono", "tum_rgbd"]


def _parse_value(raw):
    value = raw.strip()
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def parse_map_statistics(stat_path):
    stats = {}
    with open(stat_path, "r", encoding="utf-8") as f:
        for line in f:
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            key = key.strip()
            if not key:
                continue
            stats[key] = _parse_value(value)
    return stats


def save_summary_tables(rows, summary_stem="map_statistics_summary"):
    fixed_columns = [
        "sequence",
        "keyframes",
        "points",
        "average track length",
        "average reproj error",
    ]

    csv_path = f"./{summary_stem}.csv"
    md_path = f"./{summary_stem}.md"

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fixed_columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fixed_columns})

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("| " + " | ".join(fixed_columns) + " |\n")
        f.write("| " + " | ".join(["---"] * len(fixed_columns)) + " |\n")
        for row in rows:
            f.write("| " + " | ".join(str(row.get(k, "")) for k in fixed_columns) + " |\n")

    print(f"Saved: {csv_path}")
    print(f"Saved: {md_path}")


def summarize_map_statistics(data_dir, sequences, output_dir_name="openvslam_result", summary_stem="map_statistics_summary"):
    rows = []
    missing = []

    for seq in sequences:
        stat_path = os.path.join(data_dir, seq, output_dir_name, "map_statistics.txt")
        row = {"sequence": seq}
        if os.path.exists(stat_path):
            row.update(parse_map_statistics(stat_path))
        else:
            missing.append(stat_path)
        rows.append(row)

    save_summary_tables(rows, summary_stem)

    if missing:
        print(f"Warning: missing map_statistics.txt for {len(missing)} sequence(s)")
        for path in missing:
            print(f"  - {path}")


def make_run_slam_cmd(seq_dir, output_dir, config_fname, dataset_type):
    return [
        SLAM_EXE,
        "-v",
        VOCAB_PATH,
        "-d",
        seq_dir,
        "--config",
        config_fname,
        "--output",
        output_dir,
        "--eval-log",
        "--auto-term",
        "--start",
        "0",
        "--duration",
        "-1",
        "--dataset-type",
        dataset_type,
    ]


def run_dataset(data_dir, sequences, config_fname, dataset_type, output_dir_name="openvslam_result"):
    for seq in sequences:
        seq_dir = os.path.join(data_dir, seq)
        output_dir = os.path.join(seq_dir, output_dir_name)
        os.makedirs(output_dir, exist_ok=True)
        subprocess.run(
            make_run_slam_cmd(seq_dir, output_dir, config_fname, dataset_type),
            check=True,
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run selected SLAM configs: euroc_mono/euroc_stereo/tum_mono/tum_rgbd"
    )
    parser.add_argument(
        "--configs",
        required=True,
        nargs="+",
        choices=PRESET_ORDER,
        help=(
            "Required. Presets to run as a list. "
            "Example: --configs euroc_mono tum_rgbd"
        ),
    )
    return parser.parse_args()


def resolve_selected_presets(args):
    requested = args.configs

    selected = [p for p in PRESET_ORDER if p in set(requested)]
    if not selected:
        raise SystemExit("No presets selected. Please pass --configs.")
    return selected


def run_selected_presets(selected_presets):
    subprocess.run(
        ["make", "-C", "../build", "-j", "4"],
        check=True,
    )
    for preset_name in selected_presets:
        preset = PRESETS[preset_name]
        dataset = DATASETS[preset["dataset"]]
        output_dir_name = f"openvslam_result_{preset_name}"
        run_dataset(
            data_dir=dataset["data_dir"],
            sequences=dataset["sequences"],
            config_fname=preset["config_fname"],
            dataset_type=dataset["dataset_type"],
            output_dir_name=output_dir_name,
        )
        summarize_map_statistics(
            data_dir=dataset["data_dir"],
            sequences=dataset["sequences"],
            output_dir_name=output_dir_name,
            summary_stem=f"map_statistics_summary_{preset_name}",
        )


if __name__ == "__main__":
    args = parse_args()
    run_selected_presets(resolve_selected_presets(args))
