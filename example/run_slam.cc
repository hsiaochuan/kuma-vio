#include "util/euroc_util.h"

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

void tracking(const std::shared_ptr<openvslam::config>& cfg,
              const std::string& vocab_file_path, const std::string& sequence_dir_path,
              const unsigned int frame_skip, const bool no_sleep, const bool auto_term,
              const bool eval_log, const bool equal_hist, const std::string& output_dir,
              const double start_time, const double duration) {
    const euroc_sequence sequence(sequence_dir_path);
    const auto euroc_frames = sequence.get_frames();

    const bool stereo_mode = cfg->camera_->setup_type_ == openvslam::camera::setup_type_t::Stereo;
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
    track_times.reserve(euroc_frames.size());

    const double start = start_time + euroc_frames.at(0).timestamp_;
    const double end = (duration < 0) ? euroc_frames.back().timestamp_ + 1e-6 : start + duration;
    std::vector<int> frames_ids;
    for (int i = 0; i < static_cast<int>(euroc_frames.size()); ++i) {
        if (euroc_frames.at(i).timestamp_ < end && euroc_frames.at(i).timestamp_ >= start) {
            frames_ids.push_back(i);
        }
    }

    cv::Mat left_img_rect, right_img_rect;

    // run the SLAM in another thread
    std::thread thread([&]() {
        for (const auto i : frames_ids) {
            const auto& euroc_frame = euroc_frames.at(i);
            cv::Mat img;
            cv::Mat left_img, right_img;
            if (stereo_mode) {
                if (equal_hist) {
                    left_img = cv::imread(euroc_frame.left_img_path_, cv::IMREAD_UNCHANGED);
                    right_img = cv::imread(euroc_frame.right_img_path_, cv::IMREAD_UNCHANGED);
                    openvslam::util::equalize_histogram(left_img);
                    openvslam::util::equalize_histogram(right_img);
                }
                else {
                    left_img = cv::imread(euroc_frame.left_img_path_, cv::IMREAD_GRAYSCALE);
                    right_img = cv::imread(euroc_frame.right_img_path_, cv::IMREAD_GRAYSCALE);
                }

                if (left_img.empty() || right_img.empty()) {
                    continue;
                }

                rectifier->rectify(left_img, right_img, left_img_rect, right_img_rect);
            }
            else {
                if (equal_hist) {
                    img = cv::imread(euroc_frame.left_img_path_, cv::IMREAD_UNCHANGED);
                    openvslam::util::equalize_histogram(img);
                }
                else {
                    img = cv::imread(euroc_frame.left_img_path_, cv::IMREAD_GRAYSCALE);
                }

                if (img.empty()) {
                    continue;
                }
            }

            const auto tp_1 = std::chrono::steady_clock::now();

            if (i % frame_skip == 0) {
                if (stereo_mode) {
                    // input the current frame and estimate the camera pose
                    SLAM.feed_stereo_frame(left_img_rect, right_img_rect, euroc_frame.timestamp_);
                }
                else {
                    // input the current frame and estimate the camera pose
                    SLAM.feed_monocular_frame(img, euroc_frame.timestamp_);
                }
            }

            const auto tp_2 = std::chrono::steady_clock::now();

            const auto track_time = std::chrono::duration_cast<std::chrono::duration<double>>(tp_2 - tp_1).count();
            if (i % frame_skip == 0) {
                track_times.push_back(track_time);
            }

            // wait until the timestamp of the next frame
            if (!no_sleep && i < static_cast<int>(euroc_frames.size()) - 1) {
                const auto wait_time = euroc_frames.at(i + 1).timestamp_ - (euroc_frame.timestamp_ + track_time);
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

    // run tracking
    if (cfg->camera_->setup_type_ != openvslam::camera::setup_type_t::Monocular && cfg->camera_->setup_type_ != openvslam::camera::setup_type_t::Stereo) {
        throw std::runtime_error("Invalid setup type: " + cfg->camera_->get_setup_type_string());
    }

    tracking(cfg, vocab_file_path->value(), data_dir_path->value(),
             frame_skip->value(), no_sleep->is_set(), auto_term->is_set(),
             eval_log->is_set(), equal_hist->is_set(), output->value(),
             start->value(), end_time->value());

#ifdef USE_GOOGLE_PERFTOOLS
    ProfilerStop();
#endif

    return EXIT_SUCCESS;
}
