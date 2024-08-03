#pragma once

struct Timer
{
    std::chrono::steady_clock::time_point start;
    std::chrono::steady_clock::time_point end;

    inline Timer() { start = std::chrono::steady_clock::now(); }
    inline Timer& Start() { start = std::chrono::steady_clock::now(); return *this; }
    inline Timer& End() { end = std::chrono::steady_clock::now(); return *this; }
    inline double DurationInMilliseconds() { return std::chrono::duration<double>(end - start).count() * 1000; }
};

struct ScopeTimer
{
    Timer timer;
    std::function<void(double durationInMilliseconds)> callback;

    inline ScopeTimer(std::function<void(double durationInMilliseconds)> callback) : callback(callback)
    {
        timer.Start();
    }
    
    inline ~ScopeTimer() 
    { 
        callback(timer.End().DurationInMilliseconds());
    }
};

struct Timings
{
    std::vector<double> rawSamples;

    struct Stats
    {
        size_t count;
        double sum;
        double average;
        double median;
        double min;
        double max;
    };

    struct SampleStats
    {
        Stats cold;
        Stats hot;
    };

    static Stats ComputeStats(gsl::span<const double> sampleSpan)
    {
        Stats stats = {};

        if (!sampleSpan.empty())
        {
            std::vector<double> samples(sampleSpan.size());
            std::copy(sampleSpan.begin(), sampleSpan.end(), samples.begin());
            std::sort(samples.begin(), samples.end());

            stats.count = sampleSpan.size();
            stats.sum = std::accumulate(samples.begin(), samples.end(), 0.0);
            stats.average = stats.sum / samples.size();
            stats.median = samples[samples.size() / 2];
            stats.min = samples[0];
            stats.max = samples[samples.size() - 1];
        }

        return stats;
    }

    SampleStats ComputeStats(size_t maxWarmupSampleCount) const
    {
        SampleStats stats = {};
        if (rawSamples.empty())
        {
            return stats;
        }

        // The first samples may be from "warmup" runs that skew the results because of cold caches.
        // We call the first few samples "cold" and the later samples "hot". We always want at least 
        // 1 hot sample. Example:
        //
        // Raw Samples | maxWarmup | cold | hot
        // ------------|-----------|------|----
        //           0 |         2 |    0 |   0
        //           1 |         2 |    0 |   1
        //           2 |         2 |    1 |   1
        //           3 |         2 |    2 |   1
        //           4 |         2 |    2 |   2
        //           5 |         2 |    2 |   3

        size_t coldSampleCount = std::min(std::max<size_t>(rawSamples.size(), 1) - 1, maxWarmupSampleCount);
        size_t hotSampleCount = rawSamples.size() - coldSampleCount;
        assert(coldSampleCount + hotSampleCount == rawSamples.size());

        stats.cold = ComputeStats(gsl::make_span<const double>(rawSamples.data(), coldSampleCount));
        stats.hot = ComputeStats(gsl::make_span<const double>(rawSamples.data() + coldSampleCount, hotSampleCount));

        return stats;
    }
};