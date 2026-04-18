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
SLAM_EXE = "../build/run_slam"


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


def run_all_sequences():
    subprocess.run(
        ["make", "-C", "../build", "-j", "4"],
        check=True,
    )

    is_stereo = False
    for seq in euroc_seqs:
        output_dir = os.path.join(euroc_data_dir, seq, "openvslam_result")
        if is_stereo:
            config_fname = './euroc/EuRoC_stereo.yaml'
        else:
            config_fname = './euroc/EuRoC_mono.yaml'
        os.makedirs(output_dir, exist_ok=True)
        subprocess.run([
            SLAM_EXE,
            "-v", "./orb_vocab.fbow",
            "-d", os.path.join(euroc_data_dir, seq),
            "--config", config_fname,
            "--output", output_dir,
            "--eval-log", "1",
            "--auto-term", "1",
            "--start", "0",
            "--duration", "-1",
        ], check=True)

    summarize_map_statistics(euroc_data_dir, euroc_seqs)


if __name__ == "__main__":
    run_all_sequences()
