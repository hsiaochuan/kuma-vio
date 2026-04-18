#include "util/euroc_util.h"
#include "util/tum_rgbd_util.h"

#ifdef USE_PANGOLIN_VIEWER
#include "pangolin_viewer/viewer.h"
#elif USE_SOCKET_PUBLISHER
#include "socket_publisher/publisher.h"
#endif

#include "openvslam/system.h"
#include "openvslam/config.h"
#include "openvslam/util/stereo_rectifier.h"
#include "openvslam/util/image_converter.h"
#include "openvslam/util/yaml.h"

#include <iostream>
#include <algorithm>
#include <cctype>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <numeric>

#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <spdlog/spdlog.h>
#include <popl.hpp>

#ifdef USE_STACK_TRACE_LOGGER
#include <glog/logging.h>
#endif

#ifdef USE_GOOGLE_PERFTOOLS
#include <gperftools/profiler.h>
#endif

enum class dataset_type_t {
    EuRoC,
    TUM_RGBD
};

dataset_type_t parse_dataset_type(std::string dataset_type_str) {
    std::transform(dataset_type_str.begin(), dataset_type_str.end(), dataset_type_str.begin(),
                   [](const unsigned char c) { return static_cast<char>(std::tolower(c)); });

    if (dataset_type_str == "euroc") {
        return dataset_type_t::EuRoC;
    }
    if (dataset_type_str == "tum_rgbd" || dataset_type_str == "tum-rgbd" || dataset_type_str == "tum") {
        return dataset_type_t::TUM_RGBD;
    }
    throw std::runtime_error("Unsupported dataset type: " + dataset_type_str + " (expected: euroc, tum_rgbd)");
}

struct input_frame {
    int source_idx_;
    std::string left_img_path_;
    std::string right_img_path_;
    std::string depth_img_path_;
    double timestamp_;
};

std::vector<input_frame> load_input_frames(const dataset_type_t dataset_type,
                                           const std::string& sequence_dir_path,
                                           const double start_time,
                                           const double duration) {
    std::vector<input_frame> all_frames;

    if (dataset_type == dataset_type_t::EuRoC) {
        const euroc_sequence sequence(sequence_dir_path);
        const auto euroc_frames = sequence.get_frames();
        all_frames.reserve(euroc_frames.size());
        for (int i = 0; i < static_cast<int>(euroc_frames.size()); ++i) {
            const auto& frame = euroc_frames.at(i);
            all_frames.push_back({i, frame.left_img_path_, frame.right_img_path_, "", frame.timestamp_});
        }
    }
    else {
        const tum_rgbd_sequence sequence(sequence_dir_path);
        const auto tum_rgbd_frames = sequence.get_frames();
        all_frames.reserve(tum_rgbd_frames.size());
        for (int i = 0; i < static_cast<int>(tum_rgbd_frames.size()); ++i) {
            const auto& frame = tum_rgbd_frames.at(i);
            all_frames.push_back({i, frame.rgb_img_path_, "", frame.depth_img_path_, frame.timestamp_});
        }
    }

    if (all_frames.empty()) {
        throw std::runtime_error("No frames were found in dataset directory: " + sequence_dir_path);
    }

    const double start = start_time + all_frames.front().timestamp_;
    const double end = (duration < 0) ? all_frames.back().timestamp_ + 1e-6 : start + duration;

    std::vector<input_frame> selected_frames;
    selected_frames.reserve(all_frames.size());
    for (const auto& frame : all_frames) {
        if (frame.timestamp_ < end && frame.timestamp_ >= start) {
            selected_frames.push_back(frame);
        }
    }

    if (selected_frames.empty()) {
        throw std::runtime_error("No frames fall into the selected time range [" + std::to_string(start_time)
                                 + ", " + std::to_string(start_time + duration) + ")");
    }

    return selected_frames;
}

void tracking(const std::shared_ptr<openvslam::config>& cfg,
              const std::string& vocab_file_path, const std::string& sequence_dir_path,
              const unsigned int frame_skip, const bool no_sleep, const bool auto_term,
              const bool eval_log, const bool equal_hist, const std::string& output_dir,
              const double start_time, const double duration, const dataset_type_t dataset_type) {
    const auto frames = load_input_frames(dataset_type, sequence_dir_path, start_time, duration);

    const auto setup_type = cfg->camera_->setup_type_;
    const bool mono_mode = setup_type == openvslam::camera::setup_type_t::Monocular;
    const bool stereo_mode = setup_type == openvslam::camera::setup_type_t::Stereo;
    const bool rgbd_mode = setup_type == openvslam::camera::setup_type_t::RGBD;

    if (!(mono_mode || stereo_mode || rgbd_mode)) {
        throw std::runtime_error("Invalid setup type: " + cfg->camera_->get_setup_type_string());
    }
    if (dataset_type == dataset_type_t::EuRoC && rgbd_mode) {
        throw std::runtime_error("EuRoC dataset cannot be used with RGBD setup");
    }
    if (dataset_type == dataset_type_t::TUM_RGBD && stereo_mode) {
        throw std::runtime_error("TUM RGB-D dataset cannot be used with Stereo setup");
    }

    std::unique_ptr<openvslam::util::stereo_rectifier> rectifier;
    if (stereo_mode) {
        rectifier.reset(new openvslam::util::stereo_rectifier(cfg));
    }

    // build a SLAM system
    openvslam::system SLAM(cfg, vocab_file_path);
    // startup the SLAM process
    SLAM.startup();

    // create a viewer object
    // and pass the frame_publisher and the map_publisher
#ifdef USE_PANGOLIN_VIEWER
    pangolin_viewer::viewer viewer(
        openvslam::util::yaml_optional_ref(cfg->yaml_node_, "PangolinViewer"), &SLAM, SLAM.get_frame_publisher(), SLAM.get_map_publisher());
#elif USE_SOCKET_PUBLISHER
    socket_publisher::publisher publisher(
        openvslam::util::yaml_optional_ref(cfg->yaml_node_, "SocketPublisher"), &SLAM, SLAM.get_frame_publisher(), SLAM.get_map_publisher());
#endif

    std::vector<double> track_times;
    track_times.reserve(frames.size());

    cv::Mat mono_img, left_img, right_img, depth_img;
    cv::Mat left_img_rect, right_img_rect;

    // run the SLAM in another thread
    std::thread thread([&]() {
        for (int i = 0; i < static_cast<int>(frames.size()); ++i) {
            const auto& frame = frames.at(i);
            const bool should_track = (frame.source_idx_ % frame_skip == 0);

            mono_img.release();
            left_img.release();
            right_img.release();
            depth_img.release();

            if (stereo_mode) {
                if (equal_hist) {
                    left_img = cv::imread(frame.left_img_path_, cv::IMREAD_UNCHANGED);
                    right_img = cv::imread(frame.right_img_path_, cv::IMREAD_UNCHANGED);
                    openvslam::util::convert_to_grayscale(left_img, cfg->camera_->color_order_);
                    openvslam::util::convert_to_grayscale(right_img, cfg->camera_->color_order_);
                    openvslam::util::equalize_histogram(left_img);
                    openvslam::util::equalize_histogram(right_img);
                }
                else {
                    left_img = cv::imread(frame.left_img_path_, cv::IMREAD_GRAYSCALE);
                    right_img = cv::imread(frame.right_img_path_, cv::IMREAD_GRAYSCALE);
                }

                if (left_img.empty() || right_img.empty()) {
                    continue;
                }

                rectifier->rectify(left_img, right_img, left_img_rect, right_img_rect);
            }
            else if (rgbd_mode) {
                mono_img = cv::imread(frame.left_img_path_, cv::IMREAD_UNCHANGED);
                depth_img = cv::imread(frame.depth_img_path_, cv::IMREAD_UNCHANGED);

                if (mono_img.empty() || depth_img.empty()) {
                    continue;
                }

                if (equal_hist) {
                    openvslam::util::convert_to_grayscale(mono_img, cfg->camera_->color_order_);
                    openvslam::util::equalize_histogram(mono_img);
                }
            }
            else {
                mono_img = cv::imread(frame.left_img_path_, cv::IMREAD_UNCHANGED);

                if (mono_img.empty()) {
                    continue;
                }

                if (equal_hist) {
                    openvslam::util::convert_to_grayscale(mono_img, cfg->camera_->color_order_);
                    openvslam::util::equalize_histogram(mono_img);
                }
            }

            const auto tp_1 = std::chrono::steady_clock::now();

            if (should_track) {
                if (stereo_mode) {
                    // input the current frame and estimate the camera pose
                    SLAM.feed_stereo_frame(left_img_rect, right_img_rect, frame.timestamp_);
                }
                else if (rgbd_mode) {
                    // input the current frame and estimate the camera pose
                    SLAM.feed_RGBD_frame(mono_img, depth_img, frame.timestamp_);
                }
                else {
                    // input the current frame and estimate the camera pose
                    SLAM.feed_monocular_frame(mono_img, frame.timestamp_);
                }
            }

            const auto tp_2 = std::chrono::steady_clock::now();

            const auto track_time = std::chrono::duration_cast<std::chrono::duration<double>>(tp_2 - tp_1).count();
            if (should_track) {
                track_times.push_back(track_time);
            }

            // wait until the timestamp of the next frame
            if (!no_sleep && i < static_cast<int>(frames.size()) - 1) {
                const auto wait_time = frames.at(i + 1).timestamp_ - (frame.timestamp_ + track_time);
                if (0.0 < wait_time) {
                    std::this_thread::sleep_for(std::chrono::microseconds(static_cast<unsigned int>(wait_time * 1e6)));
                }
            }

            // check if the termination of SLAM system is requested or not
            if (SLAM.terminate_is_requested()) {
                break;
            }
        }

        // wait until the loop BA is finished
        while (SLAM.loop_BA_is_running()) {
            std::this_thread::sleep_for(std::chrono::microseconds(5000));
        }

        // automatically close the viewer
#ifdef USE_PANGOLIN_VIEWER
        if (auto_term) {
            viewer.request_terminate();
        }
#elif USE_SOCKET_PUBLISHER
        if (auto_term) {
            publisher.request_terminate();
        }
#endif
    });

    // run the viewer in the current thread
#ifdef USE_PANGOLIN_VIEWER
    viewer.run();
#elif USE_SOCKET_PUBLISHER
    publisher.run();
#endif

    thread.join();

    // shutdown the SLAM process
    SLAM.shutdown();

    // print final map statistics
    SLAM.print_map_statistics(output_dir + "/map_statistics.txt");

    if (eval_log) {
        // output the trajectories for evaluation
        SLAM.save_frame_trajectory(output_dir + "/frame_trajectory.txt", "TUM");
        SLAM.save_keyframe_trajectory(output_dir + "/keyframe_trajectory.txt", "TUM");
        // output the tracking times for evaluation
        std::ofstream ofs(output_dir + "/track_times.txt", std::ios::out);
        if (ofs.is_open()) {
            for (const auto track_time : track_times) {
                ofs << track_time << std::endl;
            }
            ofs.close();
        }
    }

    SLAM.save_map_database(output_dir + "/map.db");

    if (track_times.empty()) {
        std::cerr << "no tracking frames were processed" << std::endl;
        return;
    }

    std::sort(track_times.begin(), track_times.end());
    const auto total_track_time = std::accumulate(track_times.begin(), track_times.end(), 0.0);
    std::cout << "median tracking time: " << track_times.at(track_times.size() / 2) << "[s]" << std::endl;
    std::cout << "mean tracking time: " << total_track_time / static_cast<double>(track_times.size()) << "[s]" << std::endl;
}

int main(int argc, char* argv[]) {
#ifdef USE_STACK_TRACE_LOGGER
    google::InitGoogleLogging(argv[0]);
    google::InstallFailureSignalHandler();
#endif

    // create options
    popl::OptionParser op("Allowed options");
    auto help = op.add<popl::Switch>("h", "help", "produce help message");
    auto vocab_file_path = op.add<popl::Value<std::string>>("v", "vocab", "vocabulary file path");
    auto data_dir_path = op.add<popl::Value<std::string>>("d", "data-dir", "directory path which contains dataset");
    auto config_file_path = op.add<popl::Value<std::string>>("c", "config", "config file path");
    auto frame_skip = op.add<popl::Value<unsigned int>>("", "frame-skip", "interval of frame skip", 1);
    auto no_sleep = op.add<popl::Switch>("", "no-sleep", "not wait for next frame in real time");
    auto auto_term = op.add<popl::Switch>("", "auto-term", "automatically terminate the viewer");
    auto debug_mode = op.add<popl::Switch>("", "debug", "debug mode");
    auto eval_log = op.add<popl::Switch>("", "eval-log", "store trajectory and tracking times for evaluation");
    auto equal_hist = op.add<popl::Switch>("", "equal-hist", "apply histogram equalization");
    auto dataset_type = op.add<popl::Value<std::string>>("", "dataset-type", "dataset type: euroc or tum_rgbd", "euroc");
    auto start = op.add<popl::Value<double>>("", "start", "start timestamp [sec] in dataset timeline", 0.0);
    auto end_time = op.add<popl::Value<double>>("", "duration", "end timestamp [sec] in dataset timeline; negative means until end", -1.0);
    auto output = op.add<popl::Value<std::string>>("o", "output", "directory path to save output files (e.g., trajectory, map database)", ".");
    try {
        op.parse(argc, argv);
    }
    catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        std::cerr << std::endl;
        std::cerr << op << std::endl;
        return EXIT_FAILURE;
    }

    // check validness of options
    if (help->is_set()) {
        std::cerr << op << std::endl;
        return EXIT_FAILURE;
    }
    if (!vocab_file_path->is_set() || !data_dir_path->is_set() || !config_file_path->is_set()) {
        std::cerr << "invalid arguments" << std::endl;
        std::cerr << std::endl;
        std::cerr << op << std::endl;
        return EXIT_FAILURE;
    }

    // setup logger
    spdlog::set_pattern("[%Y-%m-%d %H:%M:%S.%e] %^[%L] %v%$");
    if (debug_mode->is_set()) {
        spdlog::set_level(spdlog::level::debug);
    }
    else {
        spdlog::set_level(spdlog::level::info);
    }

    // load configuration
    std::shared_ptr<openvslam::config> cfg;
    try {
        cfg = std::make_shared<openvslam::config>(config_file_path->value());
    }
    catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

#ifdef USE_GOOGLE_PERFTOOLS
    ProfilerStart("slam.prof");
#endif

    const auto parsed_dataset_type = parse_dataset_type(dataset_type->value());

    // run tracking
    tracking(cfg, vocab_file_path->value(), data_dir_path->value(),
             frame_skip->value(), no_sleep->is_set(), auto_term->is_set(),
             eval_log->is_set(), equal_hist->is_set(), output->value(),
             start->value(), end_time->value(), parsed_dataset_type);

#ifdef USE_GOOGLE_PERFTOOLS
    ProfilerStop();
#endif

    return EXIT_SUCCESS;
}
