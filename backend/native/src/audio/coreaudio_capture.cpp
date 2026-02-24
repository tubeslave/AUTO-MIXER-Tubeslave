/**
 * @file coreaudio_capture.cpp
 * @brief CoreAudio input capture implementation for macOS
 */

#include "coreaudio_capture.h"
#include <iostream>
#include <algorithm>
#include <cstring>

#ifdef __APPLE__
#include <CoreFoundation/CoreFoundation.h>
#endif

namespace AutoFader {
namespace Audio {

CoreAudioCapture::CoreAudioCapture() = default;

CoreAudioCapture::~CoreAudioCapture() {
    stop();
#ifdef __APPLE__
    if (audio_unit_) {
        AudioComponentInstanceDispose(audio_unit_);
        audio_unit_ = nullptr;
    }
    if (buffer_list_) {
        for (UInt32 i = 0; i < buffer_list_->mNumberBuffers; ++i) {
            free(buffer_list_->mBuffers[i].mData);
        }
        free(buffer_list_);
        buffer_list_ = nullptr;
    }
#endif
}

std::vector<AudioDeviceInfo> CoreAudioCapture::listDevices() {
    std::vector<AudioDeviceInfo> devices;
    
#ifdef __APPLE__
    // Get all audio devices
    AudioObjectPropertyAddress prop;
    prop.mSelector = kAudioHardwarePropertyDevices;
    prop.mScope = kAudioObjectPropertyScopeGlobal;
    prop.mElement = kAudioObjectPropertyElementMain;
    
    UInt32 data_size = 0;
    OSStatus status = AudioObjectGetPropertyDataSize(
        kAudioObjectSystemObject, &prop, 0, nullptr, &data_size);
    
    if (status != noErr || data_size == 0) {
        return devices;
    }
    
    int device_count = data_size / sizeof(AudioDeviceID);
    std::vector<AudioDeviceID> device_ids(device_count);
    
    status = AudioObjectGetPropertyData(
        kAudioObjectSystemObject, &prop, 0, nullptr, &data_size, device_ids.data());
    
    if (status != noErr) {
        return devices;
    }
    
    for (AudioDeviceID dev_id : device_ids) {
        AudioDeviceInfo info;
        info.device_id = dev_id;
        
        // Get device name
        prop.mSelector = kAudioObjectPropertyName;
        prop.mScope = kAudioObjectPropertyScopeGlobal;
        CFStringRef name_ref = nullptr;
        data_size = sizeof(CFStringRef);
        status = AudioObjectGetPropertyData(dev_id, &prop, 0, nullptr, &data_size, &name_ref);
        if (status == noErr && name_ref) {
            char name_buf[256];
            if (CFStringGetCString(name_ref, name_buf, sizeof(name_buf), kCFStringEncodingUTF8)) {
                info.name = name_buf;
            }
            CFRelease(name_ref);
        }
        
        // Get device UID
        prop.mSelector = kAudioDevicePropertyDeviceUID;
        CFStringRef uid_ref = nullptr;
        data_size = sizeof(CFStringRef);
        status = AudioObjectGetPropertyData(dev_id, &prop, 0, nullptr, &data_size, &uid_ref);
        if (status == noErr && uid_ref) {
            char uid_buf[256];
            if (CFStringGetCString(uid_ref, uid_buf, sizeof(uid_buf), kCFStringEncodingUTF8)) {
                info.uid = uid_buf;
            }
            CFRelease(uid_ref);
        }
        
        // Get input channel count
        prop.mSelector = kAudioDevicePropertyStreamConfiguration;
        prop.mScope = kAudioDevicePropertyScopeInput;
        data_size = 0;
        status = AudioObjectGetPropertyDataSize(dev_id, &prop, 0, nullptr, &data_size);
        if (status == noErr && data_size > 0) {
            std::vector<uint8_t> buf(data_size);
            auto* abl = reinterpret_cast<AudioBufferList*>(buf.data());
            status = AudioObjectGetPropertyData(dev_id, &prop, 0, nullptr, &data_size, abl);
            if (status == noErr) {
                int total_channels = 0;
                for (UInt32 i = 0; i < abl->mNumberBuffers; ++i) {
                    total_channels += abl->mBuffers[i].mNumberChannels;
                }
                info.max_input_channels = total_channels;
            }
        }
        
        // Get output channel count
        prop.mScope = kAudioDevicePropertyScopeOutput;
        data_size = 0;
        status = AudioObjectGetPropertyDataSize(dev_id, &prop, 0, nullptr, &data_size);
        if (status == noErr && data_size > 0) {
            std::vector<uint8_t> buf(data_size);
            auto* abl = reinterpret_cast<AudioBufferList*>(buf.data());
            status = AudioObjectGetPropertyData(dev_id, &prop, 0, nullptr, &data_size, abl);
            if (status == noErr) {
                int total_channels = 0;
                for (UInt32 i = 0; i < abl->mNumberBuffers; ++i) {
                    total_channels += abl->mBuffers[i].mNumberChannels;
                }
                info.max_output_channels = total_channels;
            }
        }
        
        // Get sample rate
        prop.mSelector = kAudioDevicePropertyNominalSampleRate;
        prop.mScope = kAudioObjectPropertyScopeGlobal;
        Float64 sr = 0;
        data_size = sizeof(Float64);
        status = AudioObjectGetPropertyData(dev_id, &prop, 0, nullptr, &data_size, &sr);
        if (status == noErr) {
            info.sample_rate = sr;
        }
        
        // Only include devices with input channels
        if (info.max_input_channels > 0) {
            devices.push_back(info);
        }
    }
#endif
    
    return devices;
}

uint32_t CoreAudioCapture::findDeviceByName(const std::string& partial_name) {
    auto devices = listDevices();
    
    // Case-insensitive search
    std::string search = partial_name;
    std::transform(search.begin(), search.end(), search.begin(), ::tolower);
    
    for (const auto& dev : devices) {
        std::string dev_name = dev.name;
        std::transform(dev_name.begin(), dev_name.end(), dev_name.begin(), ::tolower);
        if (dev_name.find(search) != std::string::npos) {
            return dev.device_id;
        }
    }
    
    return 0;  // Not found
}

bool CoreAudioCapture::initialize(uint32_t device_id, int num_channels, double sample_rate, int block_size) {
#ifndef __APPLE__
    last_error_ = "CoreAudio is only available on macOS";
    return false;
#else
    block_size_ = block_size;
    
    // Get default input device if device_id is 0
    if (device_id == 0) {
        AudioObjectPropertyAddress prop;
        prop.mSelector = kAudioHardwarePropertyDefaultInputDevice;
        prop.mScope = kAudioObjectPropertyScopeGlobal;
        prop.mElement = kAudioObjectPropertyElementMain;
        
        AudioDeviceID default_id = 0;
        UInt32 size = sizeof(AudioDeviceID);
        OSStatus status = AudioObjectGetPropertyData(
            kAudioObjectSystemObject, &prop, 0, nullptr, &size, &default_id);
        
        if (status != noErr || default_id == kAudioObjectUnknown) {
            last_error_ = "No default input device found";
            return false;
        }
        device_id = default_id;
    }
    
    // Get device info
    auto devices = listDevices();
    const AudioDeviceInfo* dev_info = nullptr;
    for (const auto& d : devices) {
        if (d.device_id == device_id) {
            dev_info = &d;
            break;
        }
    }
    
    if (!dev_info) {
        last_error_ = "Device ID " + std::to_string(device_id) + " not found";
        return false;
    }
    
    // Clamp channels to device maximum
    actual_channels_ = std::min(num_channels, dev_info->max_input_channels);
    if (actual_channels_ <= 0) {
        last_error_ = "Device has no input channels";
        return false;
    }
    
    std::cout << "CoreAudio: Using device '" << dev_info->name << "' (ID: " << device_id
              << ", " << actual_channels_ << "/" << dev_info->max_input_channels << " channels)" << std::endl;
    
    // Use device sample rate if not specified
    actual_sample_rate_ = (sample_rate > 0) ? sample_rate : dev_info->sample_rate;
    
    // Set device sample rate if needed
    if (sample_rate > 0 && sample_rate != dev_info->sample_rate) {
        AudioObjectPropertyAddress prop;
        prop.mSelector = kAudioDevicePropertyNominalSampleRate;
        prop.mScope = kAudioObjectPropertyScopeGlobal;
        prop.mElement = kAudioObjectPropertyElementMain;
        
        Float64 new_sr = sample_rate;
        OSStatus status = AudioObjectSetPropertyData(
            device_id, &prop, 0, nullptr, sizeof(Float64), &new_sr);
        
        if (status != noErr) {
            std::cerr << "CoreAudio: Could not set sample rate to " << sample_rate
                      << ", using device default " << dev_info->sample_rate << std::endl;
            actual_sample_rate_ = dev_info->sample_rate;
        }
    }
    
    // Set buffer size
    {
        AudioObjectPropertyAddress prop;
        prop.mSelector = kAudioDevicePropertyBufferFrameSize;
        prop.mScope = kAudioObjectPropertyScopeGlobal;
        prop.mElement = kAudioObjectPropertyElementMain;
        
        UInt32 buf_size = block_size;
        OSStatus status = AudioObjectSetPropertyData(
            device_id, &prop, 0, nullptr, sizeof(UInt32), &buf_size);
        
        if (status != noErr) {
            // Read actual buffer size
            UInt32 size = sizeof(UInt32);
            AudioObjectGetPropertyData(device_id, &prop, 0, nullptr, &size, &buf_size);
            std::cerr << "CoreAudio: Could not set buffer size to " << block_size
                      << ", using " << buf_size << std::endl;
            block_size_ = buf_size;
        }
    }
    
    // Create AudioUnit (AUHAL - Audio Unit Hardware Abstraction Layer)
    AudioComponentDescription desc;
    desc.componentType = kAudioUnitType_Output;
    desc.componentSubType = kAudioUnitSubType_HALOutput;
    desc.componentManufacturer = kAudioUnitManufacturer_Apple;
    desc.componentFlags = 0;
    desc.componentFlagsMask = 0;
    
    AudioComponent component = AudioComponentFindNext(nullptr, &desc);
    if (!component) {
        last_error_ = "Could not find HAL AudioUnit component";
        return false;
    }
    
    OSStatus status = AudioComponentInstanceNew(component, &audio_unit_);
    if (status != noErr) {
        last_error_ = "Could not create AudioUnit instance (error " + std::to_string(status) + ")";
        return false;
    }
    
    // Enable input on bus 1
    UInt32 enable_io = 1;
    status = AudioUnitSetProperty(audio_unit_,
        kAudioOutputUnitProperty_EnableIO,
        kAudioUnitScope_Input,
        1,  // Input bus
        &enable_io, sizeof(enable_io));
    
    if (status != noErr) {
        last_error_ = "Could not enable input (error " + std::to_string(status) + ")";
        return false;
    }
    
    // Disable output on bus 0
    UInt32 disable_io = 0;
    status = AudioUnitSetProperty(audio_unit_,
        kAudioOutputUnitProperty_EnableIO,
        kAudioUnitScope_Output,
        0,  // Output bus
        &disable_io, sizeof(disable_io));
    
    if (status != noErr) {
        last_error_ = "Could not disable output (error " + std::to_string(status) + ")";
        return false;
    }
    
    // Set input device
    status = AudioUnitSetProperty(audio_unit_,
        kAudioOutputUnitProperty_CurrentDevice,
        kAudioUnitScope_Global,
        0,
        &device_id, sizeof(AudioDeviceID));
    
    if (status != noErr) {
        last_error_ = "Could not set input device (error " + std::to_string(status) + ")";
        return false;
    }
    
    // Set stream format: 32-bit float, non-interleaved (one buffer per channel)
    AudioStreamBasicDescription format;
    std::memset(&format, 0, sizeof(format));
    format.mSampleRate = actual_sample_rate_;
    format.mFormatID = kAudioFormatLinearPCM;
    format.mFormatFlags = kAudioFormatFlagIsFloat | kAudioFormatFlagIsPacked | kAudioFormatFlagIsNonInterleaved;
    format.mBitsPerChannel = 32;
    format.mChannelsPerFrame = actual_channels_;
    format.mFramesPerPacket = 1;
    format.mBytesPerFrame = sizeof(float);
    format.mBytesPerPacket = sizeof(float);
    
    status = AudioUnitSetProperty(audio_unit_,
        kAudioUnitProperty_StreamFormat,
        kAudioUnitScope_Output,  // Output scope of input bus = what we receive
        1,  // Input bus
        &format, sizeof(format));
    
    if (status != noErr) {
        last_error_ = "Could not set stream format (error " + std::to_string(status) + ")";
        return false;
    }
    
    // Set render callback
    AURenderCallbackStruct callback_struct;
    callback_struct.inputProc = renderCallback;
    callback_struct.inputProcRefCon = this;
    
    status = AudioUnitSetProperty(audio_unit_,
        kAudioOutputUnitProperty_SetInputCallback,
        kAudioUnitScope_Global,
        0,
        &callback_struct, sizeof(callback_struct));
    
    if (status != noErr) {
        last_error_ = "Could not set input callback (error " + std::to_string(status) + ")";
        return false;
    }
    
    // Allocate buffer list for render
    buffer_list_ = static_cast<AudioBufferList*>(
        calloc(1, sizeof(AudioBufferList) + sizeof(AudioBuffer) * (actual_channels_ - 1)));
    buffer_list_->mNumberBuffers = actual_channels_;
    
    UInt32 max_frames = block_size_ * 2;  // Extra room for variable buffer sizes
    for (int i = 0; i < actual_channels_; ++i) {
        buffer_list_->mBuffers[i].mNumberChannels = 1;
        buffer_list_->mBuffers[i].mDataByteSize = max_frames * sizeof(float);
        buffer_list_->mBuffers[i].mData = calloc(max_frames, sizeof(float));
    }
    
    // Allocate channel buffers for deinterleaving
    channel_buffers_.resize(actual_channels_);
    channel_ptrs_.resize(actual_channels_);
    for (int i = 0; i < actual_channels_; ++i) {
        channel_buffers_[i].resize(max_frames, 0.0f);
        channel_ptrs_[i] = channel_buffers_[i].data();
    }
    
    // Initialize AudioUnit
    status = AudioUnitInitialize(audio_unit_);
    if (status != noErr) {
        last_error_ = "Could not initialize AudioUnit (error " + std::to_string(status) + ")";
        return false;
    }
    
    std::cout << "CoreAudio: Initialized - " << actual_channels_ << " channels @ "
              << actual_sample_rate_ << " Hz, buffer " << block_size_ << " samples" << std::endl;
    
    return true;
#endif
}

bool CoreAudioCapture::start(AudioCallback callback, void* user_data) {
#ifndef __APPLE__
    last_error_ = "CoreAudio is only available on macOS";
    return false;
#else
    if (!audio_unit_) {
        last_error_ = "Not initialized";
        return false;
    }
    
    callback_ = callback;
    user_data_ = user_data;
    running_.store(true);
    
    OSStatus status = AudioOutputUnitStart(audio_unit_);
    if (status != noErr) {
        running_.store(false);
        last_error_ = "Could not start AudioUnit (error " + std::to_string(status) + ")";
        return false;
    }
    
    std::cout << "CoreAudio: Capture started" << std::endl;
    return true;
#endif
}

void CoreAudioCapture::stop() {
#ifdef __APPLE__
    if (audio_unit_ && running_.load()) {
        AudioOutputUnitStop(audio_unit_);
        running_.store(false);
        std::cout << "CoreAudio: Capture stopped" << std::endl;
    }
#endif
}

#ifdef __APPLE__
OSStatus CoreAudioCapture::renderCallback(
    void* inRefCon,
    AudioUnitRenderActionFlags* ioActionFlags,
    const AudioTimeStamp* inTimeStamp,
    UInt32 inBusNumber,
    UInt32 inNumberFrames,
    AudioBufferList* /* ioData */)
{
    auto* self = static_cast<CoreAudioCapture*>(inRefCon);
    
    if (!self->running_.load() || !self->callback_) {
        return noErr;
    }
    
    // Update buffer sizes for this callback
    for (UInt32 i = 0; i < self->buffer_list_->mNumberBuffers; ++i) {
        self->buffer_list_->mBuffers[i].mDataByteSize = inNumberFrames * sizeof(float);
    }
    
    // Render (pull) input audio into our buffer
    OSStatus status = AudioUnitRender(
        self->audio_unit_,
        ioActionFlags,
        inTimeStamp,
        1,  // Input bus
        inNumberFrames,
        self->buffer_list_);
    
    if (status != noErr) {
        return status;
    }
    
    // Build channel pointer array (already non-interleaved from format settings)
    int num_ch = self->actual_channels_;
    for (int ch = 0; ch < num_ch; ++ch) {
        self->channel_ptrs_[ch] = static_cast<const float*>(self->buffer_list_->mBuffers[ch].mData);
    }
    
    // Call user callback with deinterleaved channel pointers
    self->callback_(self->channel_ptrs_.data(), num_ch, 
                    static_cast<int>(inNumberFrames), self->user_data_);
    
    return noErr;
}
#endif

} // namespace Audio
} // namespace AutoFader
