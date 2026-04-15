import os.path
import subprocess

euroc_data_dir = "/mnt/data/home/hsiaochuan/data/euroc"
seqs = [
    "machine_hall/MH_01_easy/MH_01_easy/mav0",
    "machine_hall/MH_02_easy/MH_02_easy/mav0",
    "machine_hall/MH_03_medium/MH_03_medium/mav0",
    "machine_hall/MH_04_difficult/MH_04_difficult/mav0",
    "machine_hall/MH_05_difficult/MH_05_difficult/mav0",
    "vicon_room1/V1_01_easy/V1_01_easy/mav0",
    "vicon_room1/V1_02_medium/V1_02_medium/mav0",
    "vicon_room1/V1_03_difficult/V1_03_difficult/mav0",
    "vicon_room2/V2_01_easy/V2_01_easy/mav0",
    "vicon_room2/V2_02_medium/V2_02_medium/mav0",
    "vicon_room2/V2_03_difficult/V2_03_difficult/mav0",
]
SLAM_EXE = "../build/run_euroc_slam"

subprocess.run(
    ["make", "-C", "./build", "-j", "4"],
)
for seq in seqs:
    os.makedirs(os.path.join(euroc_data_dir,seq,"openvslam_result"), exist_ok=True)
    subprocess.run([
        SLAM_EXE,
        "-v", "./orb_vocab.fbow",
        "-d", os.path.join(euroc_data_dir, seq),
        "--config", "./euroc//EuRoC_mono.yaml",
        "--map-db", os.path.join(euroc_data_dir,seq,"openvslam_result", "map.db"),
        "--eval-log", os.path.join(euroc_data_dir,seq,"openvslam_result", "eval.log"),
    ])

