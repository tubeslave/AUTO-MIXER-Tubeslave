/**
 * @file coreaudio_capture.h
 * @brief CoreAudio input capture for macOS
 * 
 * Provides real-time multi-channel audio capture from system audio devices
 * (e.g. Dante Virtual Soundcard, USB audio from Wing, aggregate devices).
 * Used as fallback when direct Dante SDK is not available.
 */

#pragma once

#include <functional>
#include <string>
#include <vector>
#include <atomic>
#include <cstdint>

#ifdef __APPLE__
#include <AudioToolbox/AudioToolbox.h>
#include <CoreAudio/CoreAudio.h>
#endif

namespace AutoFader {
namespace Audio {

/// Information about an available audio device
struct AudioDeviceInfo {
    uint32_t device_id;
    std::string name;
    std::string uid;
    int max_input_channels;
    int max_output_channels;
    double sample_rate;
};

/// Audio callback: (channel_ptrs, num_channels, num_samples, user_data)
using AudioCallback = std::function<void(const float* const*, int, int, void*)>;

/**
 * @class CoreAudioCapture
 * @brief Captures audio from a macOS CoreAudio input device
 * 
 * Usage:
 *   1. Call listDevices() to enumerate available input devices
 *   2. Call initialize() with device_id (or 0 for default)
 *   3. Call start() with your audio callback
 *   4. Call stop() when done
 */
class CoreAudioCapture {
public:
    CoreAudioCapture();
    ~CoreAudioCapture();
    
    // Non-copyable
    CoreAudioCapture(const CoreAudioCapture&) = delete;
    CoreAudioCapture& operator=(const CoreAudioCapture&) = delete;
    
    /// Enumerate all available input devices
    static std::vector<AudioDeviceInfo> listDevices();
    
    /// Find a device by partial name match (case-insensitive)
    static uint32_t findDeviceByName(const std::string& partial_name);
    
    /**
     * @brief Initialize capture from a specific device
     * @param device_id  CoreAudio device ID (0 = system default input)
     * @param num_channels  Number of channels to capture (will be clamped to device max)
     * @param sample_rate  Desired sample rate (0 = use device default)
     * @param block_size  Preferred buffer size in samples
     * @return true if initialization succeeded
     */
    bool initialize(uint32_t device_id, int num_channels, double sample_rate = 0, int block_size = 256);
    
    /// Start audio capture with callback
    bool start(AudioCallback callback, void* user_data = nullptr);
    
    /// Stop capture
    void stop();
    
    /// Check if currently capturing
    bool isRunning() const { return running_.load(); }
    
    /// Get last error message
    const std::string& getLastError() const { return last_error_; }
    
    /// Get actual sample rate (may differ from requested)
    double getActualSampleRate() const { return actual_sample_rate_; }
    
    /// Get actual number of channels
    int getActualChannelCount() const { return actual_channels_; }
    
private:
#ifdef __APPLE__
    /// CoreAudio render callback (called from audio thread)
    static OSStatus renderCallback(
        void* inRefCon,
        AudioUnitRenderActionFlags* ioActionFlags,
        const AudioTimeStamp* inTimeStamp,
        UInt32 inBusNumber,
        UInt32 inNumberFrames,
        AudioBufferList* ioData
    );
    
    AudioComponentInstance audio_unit_ = nullptr;
    AudioBufferList* buffer_list_ = nullptr;
#endif
    
    AudioCallback callback_;
    void* user_data_ = nullptr;
    
    std::atomic<bool> running_{false};
    std::string last_error_;
    
    double actual_sample_rate_ = 0;
    int actual_channels_ = 0;
    int block_size_ = 256;
    
    // Deinterleaved channel pointers for callback
    std::vector<std::vector<float>> channel_buffers_;
    std::vector<const float*> channel_ptrs_;
};

} // namespace Audio
} // namespace AutoFader
