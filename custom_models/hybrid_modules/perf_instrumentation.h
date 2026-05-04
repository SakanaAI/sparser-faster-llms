/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#pragma once

#include <cuda_runtime.h>
#include <string>
#include <unordered_map>
#include <vector>
#include <cstdio>
#include <cmath>
#include <algorithm>
#include <mutex>

// Compile-time flag to enable/disable profiling
#ifndef ENABLE_PERF_PROFILING
#define ENABLE_PERF_PROFILING 0
#endif

// Performance profiling infrastructure for CUDA kernels
// Uses CUDA events for accurate GPU timing with minimal overhead

namespace perf {

struct TimingStats {
    std::vector<float> samples;  // All timing samples in milliseconds
    float total_time = 0.0f;
    int count = 0;

    void add_sample(float ms) {
        samples.push_back(ms);
        total_time += ms;
        count++;
    }

    float mean() const {
        return count > 0 ? total_time / count : 0.0f;
    }

    float min() const {
        return samples.empty() ? 0.0f : *std::min_element(samples.begin(), samples.end());
    }

    float max() const {
        return samples.empty() ? 0.0f : *std::max_element(samples.begin(), samples.end());
    }

    float percentile(float p) const {
        if (samples.empty()) return 0.0f;
        std::vector<float> sorted = samples;
        std::sort(sorted.begin(), sorted.end());
        int idx = static_cast<int>(p * (sorted.size() - 1));
        return sorted[idx];
    }

    float stddev() const {
        if (count < 2) return 0.0f;
        float m = mean();
        float sum_sq_diff = 0.0f;
        for (float s : samples) {
            float diff = s - m;
            sum_sq_diff += diff * diff;
        }
        return std::sqrt(sum_sq_diff / count);
    }
};

class PerformanceProfiler {
private:
    std::unordered_map<std::string, TimingStats> stats_;
    std::unordered_map<std::string, cudaEvent_t> start_events_;
    std::unordered_map<std::string, cudaEvent_t> stop_events_;
    std::unordered_map<std::string, cudaStream_t> event_streams_;
    std::mutex mutex_;
    bool enabled_;

public:
    PerformanceProfiler() : enabled_(ENABLE_PERF_PROFILING) {
        // Check environment variable override
        const char* env_val = std::getenv("SPARSE_ENABLE_PERF_PROFILING");
        if (env_val != nullptr) {
            enabled_ = (std::atoi(env_val) != 0);
        }
    }

    ~PerformanceProfiler() {
        // Skip cleanup - CUDA events will be automatically cleaned up at process exit
        // Explicitly destroying them can cause issues with shared library unloading order
    }

    bool is_enabled() const { return enabled_; }

    void start_timing(const std::string& name, cudaStream_t stream = 0) {
        if (!enabled_) return;

        std::lock_guard<std::mutex> lock(mutex_);

        // Create or reuse events
        if (start_events_.find(name) == start_events_.end()) {
            cudaEvent_t start, stop;
            cudaError_t err = cudaEventCreate(&start);
            TORCH_CHECK(err == cudaSuccess, "cudaEventCreate failed: ", cudaGetErrorString(err));
            err = cudaEventCreate(&stop);
            TORCH_CHECK(err == cudaSuccess, "cudaEventCreate failed: ", cudaGetErrorString(err));
            start_events_[name] = start;
            stop_events_[name] = stop;
        }

        event_streams_[name] = stream;
        cudaError_t err = cudaEventRecord(start_events_[name], stream);
        TORCH_CHECK(err == cudaSuccess, "cudaEventRecord failed: ", cudaGetErrorString(err));
    }

    void stop_timing(const std::string& name) {
        if (!enabled_) return;

        std::lock_guard<std::mutex> lock(mutex_);

        if (start_events_.find(name) == start_events_.end()) {
            fprintf(stderr, "[PERF] Warning: stop_timing called for '%s' without start_timing\n", name.c_str());
            return;
        }

        cudaStream_t stream = event_streams_[name];
        cudaError_t err = cudaEventRecord(stop_events_[name], stream);
        TORCH_CHECK(err == cudaSuccess, "cudaEventRecord failed: ", cudaGetErrorString(err));
        err = cudaEventSynchronize(stop_events_[name]);
        TORCH_CHECK(err == cudaSuccess, "cudaEventSynchronize failed: ", cudaGetErrorString(err));

        float elapsed_ms = 0.0f;
        err = cudaEventElapsedTime(&elapsed_ms, start_events_[name], stop_events_[name]);
        TORCH_CHECK(err == cudaSuccess, "cudaEventElapsedTime failed: ", cudaGetErrorString(err));

        stats_[name].add_sample(elapsed_ms);
    }

    void print_report(FILE* out = stderr) const {
        if (!enabled_) return;

        fprintf(out, "\n");
        fprintf(out, "================================================================================\n");
        fprintf(out, "                    PERFORMANCE PROFILING REPORT                               \n");
        fprintf(out, "================================================================================\n");
        fprintf(out, "%-40s %8s %10s %10s %10s %10s %10s %10s\n",
                "Operation", "Count", "Total(ms)", "Mean(ms)", "Min(ms)", "Max(ms)", "P50(ms)", "Stddev");
        fprintf(out, "--------------------------------------------------------------------------------\n");

        // Calculate total time across all operations
        float total_overall = 0.0f;
        for (const auto& pair : stats_) {
            total_overall += pair.second.total_time;
        }

        // Sort by total time (descending)
        std::vector<std::pair<std::string, TimingStats>> sorted_stats(stats_.begin(), stats_.end());
        std::sort(sorted_stats.begin(), sorted_stats.end(),
                  [](const auto& a, const auto& b) { return a.second.total_time > b.second.total_time; });

        for (const auto& pair : sorted_stats) {
            const std::string& name = pair.first;
            const TimingStats& stat = pair.second;
            float percent = (total_overall > 0) ? (stat.total_time / total_overall * 100.0f) : 0.0f;

            fprintf(out, "%-40s %8d %10.3f %10.3f %10.3f %10.3f %10.3f %10.3f (%5.1f%%)\n",
                    name.c_str(), stat.count, stat.total_time, stat.mean(),
                    stat.min(), stat.max(), stat.percentile(0.5f), stat.stddev(), percent);
        }

        fprintf(out, "--------------------------------------------------------------------------------\n");
        fprintf(out, "Total profiled time: %.3f ms\n", total_overall);
        fprintf(out, "================================================================================\n");
    }

    void write_report_to_file(const std::string& filename) const {
        if (!enabled_) return;

        FILE* f = fopen(filename.c_str(), "w");
        if (!f) {
            fprintf(stderr, "[PERF] Error: cannot open file '%s' for writing\n", filename.c_str());
            return;
        }
        print_report(f);
        fclose(f);
    }

    void reset() {
        if (!enabled_) return;
        std::lock_guard<std::mutex> lock(mutex_);
        stats_.clear();
    }

    const std::unordered_map<std::string, TimingStats>& get_stats() const {
        return stats_;
    }
};

// Global profiler instance
inline PerformanceProfiler& get_global_profiler() {
    static PerformanceProfiler profiler;
    return profiler;
}

// RAII timer for automatic start/stop
class ScopedTimer {
private:
    std::string name_;
    cudaStream_t stream_;
    bool active_;

public:
    ScopedTimer(const std::string& name, cudaStream_t stream = 0)
        : name_(name), stream_(stream), active_(get_global_profiler().is_enabled()) {
        if (active_) {
            get_global_profiler().start_timing(name_, stream_);
        }
    }

    ~ScopedTimer() {
        if (active_) {
            get_global_profiler().stop_timing(name_);
        }
    }

    // Prevent copying
    ScopedTimer(const ScopedTimer&) = delete;
    ScopedTimer& operator=(const ScopedTimer&) = delete;
};

// Convenience macros for easy instrumentation
#if ENABLE_PERF_PROFILING
    #define PERF_TIMER(name, stream) perf::ScopedTimer _perf_timer_##__LINE__(name, stream)
    #define PERF_START(name, stream) perf::get_global_profiler().start_timing(name, stream)
    #define PERF_STOP(name) perf::get_global_profiler().stop_timing(name)
    #define PERF_REPORT() perf::get_global_profiler().print_report()
    #define PERF_REPORT_FILE(filename) perf::get_global_profiler().write_report_to_file(filename)
    #define PERF_RESET() perf::get_global_profiler().reset()
#else
    #define PERF_TIMER(name, stream) ((void)0)
    #define PERF_START(name, stream) ((void)0)
    #define PERF_STOP(name) ((void)0)
    #define PERF_REPORT() ((void)0)
    #define PERF_REPORT_FILE(filename) ((void)0)
    #define PERF_RESET() ((void)0)
#endif

} // namespace perf
