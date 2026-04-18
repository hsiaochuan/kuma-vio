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


def save_summary_tables(rows):
    fixed_columns = [
        "sequence",
        "keyframes",
        "points",
        "average track length",
        "average reproj error",
    ]

    csv_path = "./map_statistics_summary.csv"
    md_path = "./map_statistics_summary.md"

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


def summarize_map_statistics(data_dir, sequences):
    rows = []
    missing = []

    for seq in sequences:
        stat_path = os.path.join(data_dir, seq, "openvslam_result", "map_statistics.txt")
        row = {"sequence": seq}
        if os.path.exists(stat_path):
            row.update(parse_map_statistics(stat_path))
        else:
            missing.append(stat_path)
        rows.append(row)

    save_summary_tables(rows)

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


def run_dataset(data_dir, sequences, config_fname, dataset_type):
    for seq in sequences:
        seq_dir = os.path.join(data_dir, seq)
        output_dir = os.path.join(seq_dir, "openvslam_result")
        os.makedirs(output_dir, exist_ok=True)
        subprocess.run(
            make_run_slam_cmd(seq_dir, output_dir, config_fname, dataset_type),
            check=True,
        )


def run_all_sequences():
    subprocess.run(
        ["make", "-C", "../build", "-j", "4"],
        check=True,
    )
    EUROC_MONO_CONFIG = "./euroc/EuRoC_mono.yaml"
    EUROC_STEREO_CONFIG = "./euroc/EuRoC_stereo.yaml"
    run_dataset(
        data_dir=euroc_data_dir,
        sequences=euroc_seqs,
        config_fname=EUROC_MONO_CONFIG,
        dataset_type="euroc",
    )

    summarize_map_statistics(euroc_data_dir, euroc_seqs)

    TUM_RGBD_CONFIG = "./tum_rgbd/TUM_RGBD_rgbd_1.yaml"
    TUM_MONO_CONFIG = "./tum_rgbd/TUM_RGBD_mono_1.yaml"
    run_dataset(
        data_dir=tum_rgbd_dir,
        sequences=tum_rgbd_seqs,
        config_fname=TUM_MONO_CONFIG,
        dataset_type="tum_rgbd",
    )

    summarize_map_statistics(tum_rgbd_dir, tum_rgbd_seqs)


if __name__ == "__main__":
    run_all_sequences()
