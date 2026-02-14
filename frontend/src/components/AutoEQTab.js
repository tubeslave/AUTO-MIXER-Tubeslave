import React, { useState, useEffect, useRef, useCallback } from 'react';
import websocketService from '../services/websocket';
import './AutoEQTab.css';

const EQ_PROFILES = [
  // Drums
  { id: 'kick', name: 'Kick', category: 'Drums' },
  { id: 'kick_rock', name: 'Kick (Rock)', category: 'Drums' },
  { id: 'kick_metal', name: 'Kick (Metal)', category: 'Drums' },
  { id: 'kick_jazz', name: 'Kick (Jazz)', category: 'Drums' },
  { id: 'kick_pop', name: 'Kick (Pop)', category: 'Drums' },
  { id: 'snare', name: 'Snare', category: 'Drums' },
  { id: 'snare_rock', name: 'Snare (Rock)', category: 'Drums' },
  { id: 'snare_metal', name: 'Snare (Metal)', category: 'Drums' },
  { id: 'snare_jazz', name: 'Snare (Jazz)', category: 'Drums' },
  { id: 'tom', name: 'Tom', category: 'Drums' },
  { id: 'ftom', name: 'Floor Tom', category: 'Drums' },
  { id: 'hihat', name: 'Hi-Hat', category: 'Drums' },
  { id: 'ride', name: 'Ride', category: 'Drums' },
  { id: 'cymbals', name: 'Cymbals', category: 'Drums' },
  { id: 'overheads', name: 'Overheads', category: 'Drums' },
  // Vocals
  { id: 'leadvocal', name: 'Lead Vocal', category: 'Vocals' },
  { id: 'vocal_rock', name: 'Vocal (Rock)', category: 'Vocals' },
  { id: 'vocal_pop', name: 'Vocal (Pop)', category: 'Vocals' },
  { id: 'vocal_jazz', name: 'Vocal (Jazz)', category: 'Vocals' },
  { id: 'vocal_metal', name: 'Vocal (Metal)', category: 'Vocals' },
  { id: 'vocal_rnb', name: 'Vocal (R&B)', category: 'Vocals' },
  { id: 'vocal_classical', name: 'Vocal (Classical)', category: 'Vocals' },
  { id: 'backvocal', name: 'Back Vocal', category: 'Vocals' },
  // Guitars
  { id: 'guitar', name: 'Electric Guitar', category: 'Guitars' },
  { id: 'guitar_rock', name: 'Guitar (Rock)', category: 'Guitars' },
  { id: 'guitar_metal', name: 'Guitar (Metal)', category: 'Guitars' },
  { id: 'guitar_jazz', name: 'Guitar (Jazz)', category: 'Guitars' },
  { id: 'guitar_clean', name: 'Guitar (Clean)', category: 'Guitars' },
  { id: 'guitar_lead', name: 'Guitar (Lead)', category: 'Guitars' },
  { id: 'acousticguitar', name: 'Acoustic Guitar', category: 'Guitars' },
  // Bass
  { id: 'bass', name: 'Bass', category: 'Bass' },
  { id: 'bass_rock', name: 'Bass (Rock)', category: 'Bass' },
  { id: 'bass_metal', name: 'Bass (Metal)', category: 'Bass' },
  { id: 'bass_jazz', name: 'Bass (Jazz)', category: 'Bass' },
  { id: 'bass_funk', name: 'Bass (Funk)', category: 'Bass' },
  // Keys
  { id: 'keys', name: 'Keys', category: 'Keys' },
  { id: 'piano', name: 'Piano', category: 'Keys' },
  { id: 'synth', name: 'Synth', category: 'Keys' },
  // Orchestral
  { id: 'violin', name: 'Violin', category: 'Orchestral' },
  { id: 'cello', name: 'Cello', category: 'Orchestral' },
  { id: 'trumpet', name: 'Trumpet', category: 'Orchestral' },
  { id: 'saxophone', name: 'Saxophone', category: 'Orchestral' },
  { id: 'flute', name: 'Flute', category: 'Orchestral' },
  // Other
  { id: 'accordion', name: 'Accordion', category: 'Other' },
  { id: 'playback', name: 'Playback', category: 'Other' },
  { id: 'custom', name: 'Custom (Flat)', category: 'Other' }
];

const FREQ_LABELS = ['31', '63', '125', '250', '500', '1k', '2k', '4k', '8k', '16k'];

// Function to auto-detect profile from channel name
function detectProfileFromChannelName(channelName) {
  if (!channelName) return 'custom';
  
  const name = channelName.toUpperCase();
  
  // Specific names first (exact matches)
  if (name === 'KATYA' || name === 'SERGEY' || name === 'SLAVA') {
    return 'leadvocal';
  }
  
  // Hi-Hat variations
  if (name.includes('HI HAT') || name.includes('HI-HAT') || name.includes('HIHAT') || name.includes('HH ')) {
    return 'hihat';
  }
  
  // Playback variations
  if (name === 'PB' || name.includes('PLAYBACK') || name.startsWith('PB ')) {
    return 'playback';
  }
  
  // Accordion variations
  if (name.includes('ACCORD') || name.includes('ACCORDION')) {
    return 'accordion';
  }
  
  // Drums
  if (name.includes('KICK') || name.includes('BD ') || name === 'BD' || name.includes('BASS DRUM')) {
    return 'kick';
  }
  if (name.includes('SNARE') || name.includes('SD ') || name === 'SD') {
    return 'snare';
  }
  if (name.includes('TOM') || name.includes('TOM-') || name.includes('TOM ')) {
    if (name.includes('FLOOR') || name.includes('FTOM') || name.includes('F TOM')) {
      return 'ftom';
    }
    return 'tom';
  }
  if (name.includes('RIDE') || name.includes('RD ')) {
    return 'ride';
  }
  if (name.includes('OVERHEAD') || name.includes('OH ') || name.includes('OHL') || name.includes('OHR')) {
    return 'overheads';
  }
  if (name.includes('CYMBAL') || name.includes('CRASH')) {
    return 'cymbals';
  }
  
  // Vocals
  if (name.includes('VOCAL') || name.includes('VOC ')) {
    if (name.includes('BACK') || name.includes('BG')) {
      return 'backvocal';
    }
    return 'leadvocal';
  }
  
  // Guitars
  if (name.includes('GUITAR') || name.includes('GT ') || name.includes('GTR')) {
    if (name.includes('ACOUSTIC') || name.includes('AC ')) {
      return 'acousticguitar';
    }
    return 'guitar';
  }
  
  // Bass
  if (name.includes('BASS') || name.includes('BS ') || name === 'BS') {
    return 'bass';
  }
  
  // Keys
  if (name.includes('PIANO') || name.includes('PN ')) {
    return 'piano';
  }
  if (name.includes('KEY') || name.includes('KY ')) {
    return 'keys';
  }
  if (name.includes('SYNTH') || name.includes('SY ')) {
    return 'synth';
  }
  
  // Orchestral
  if (name.includes('VIOLIN') || name.includes('VLN')) {
    return 'violin';
  }
  if (name.includes('CELLO') || name.includes('VLC')) {
    return 'cello';
  }
  if (name.includes('TRUMPET') || name.includes('TPT')) {
    return 'trumpet';
  }
  if (name.includes('SAX') || name.includes('SAXOPHONE')) {
    return 'saxophone';
  }
  if (name.includes('FLUTE') || name.includes('FLT')) {
    return 'flute';
  }
  
  return 'custom';
}

function AutoEQTab({ selectedChannels, availableChannels, selectedDevice, audioDevices }) {
  // State
  const [isActive, setIsActive] = useState(false);
  const [selectedChannel, setSelectedChannel] = useState(selectedChannels?.[0] || 1);
  const [selectedProfile, setSelectedProfile] = useState('custom');
  const [autoApply, setAutoApply] = useState(false);
  const [showTargetCurve, setShowTargetCurve] = useState(true);
  const [statusMessage, setStatusMessage] = useState('');
  
  // Batch mode state (always enabled)
  const [channelProfiles, setChannelProfiles] = useState({});
  const [channelAutoApply, setChannelAutoApply] = useState({});
  const [channelStatus, setChannelStatus] = useState({});
  const [channelCorrections, setChannelCorrections] = useState({});
  const [selectedChannelForView, setSelectedChannelForView] = useState(null);
  
  // Initialize channel profiles when selectedChannels change
  useEffect(() => {
    if (selectedChannels && selectedChannels.length > 0 && availableChannels) {
      const newProfiles = {};
      
      // selectedChannels is array of IDs (numbers), not objects
      selectedChannels.forEach(channelId => {
        // Find channel name from availableChannels
        const channelObj = availableChannels.find(c => c.id === channelId);
        const name = channelObj ? channelObj.name : null;
        
        // Only set if not already set (preserve user changes)
        if (!channelProfiles[channelId]) {
          const detectedProfile = detectProfileFromChannelName(name);
          newProfiles[channelId] = detectedProfile;
        }
      });
      
      if (Object.keys(newProfiles).length > 0) {
        setChannelProfiles(prev => ({ ...prev, ...newProfiles }));
      }
    }
  }, [selectedChannels, availableChannels, channelProfiles]); // Include availableChannels for name lookup
  
  // Spectrum data
  const [spectrumData, setSpectrumData] = useState(new Array(32).fill(-60));
  const [targetCurve, setTargetCurve] = useState(null);
  const [spectralInfo, setSpectralInfo] = useState({
    peak_freq: 0,
    centroid: 0,
    rolloff: 0,
    flatness: 0
  });
  
  // EQ corrections
  const [corrections, setCorrections] = useState([]);
  const [eqBands, setEqBands] = useState({
    low: { gain: 0, freq: 100, q: 1.0 },
    band1: { gain: 0, freq: 250, q: 2.0 },
    band2: { gain: 0, freq: 1000, q: 2.0 },
    band3: { gain: 0, freq: 3000, q: 2.0 },
    band4: { gain: 0, freq: 8000, q: 2.0 },
    high: { gain: 0, freq: 10000, q: 1.0 }
  });
  
  // Canvas ref for spectrum visualization
  const canvasRef = useRef(null);
  const animationRef = useRef(null);
  
  // WebSocket event handlers
  useEffect(() => {
    const handleAutoEQStatus = (data) => {
      console.log('Auto-EQ status update:', data);
      
      if (data.active !== undefined) {
        setIsActive(data.active);
      }
      
      if (data.message) {
        setStatusMessage(data.message);
      }
      
      if (data.error) {
        setStatusMessage(`Error: ${data.error}`);
        setIsActive(false);
      }
      
      if (data.profile) {
        setSelectedProfile(data.profile);
      }
    };
    
    const handleAutoEQSpectrum = (data) => {
      if (data.spectrum && Array.isArray(data.spectrum)) {
        setSpectrumData(data.spectrum);
      }
      
      if (data.target_curve && Array.isArray(data.target_curve)) {
        setTargetCurve(data.target_curve);
      }
      
      setSpectralInfo({
        peak_freq: data.peak_freq || 0,
        centroid: data.centroid || 0,
        rolloff: data.rolloff || 0,
        flatness: data.flatness || 0
      });
    };
    
    const handleAutoEQCorrections = (data) => {
      if (data.corrections && Array.isArray(data.corrections)) {
        setCorrections(data.corrections);
        
        // Update EQ bands from corrections
        const newBands = { ...eqBands };
        data.corrections.forEach((corr, idx) => {
          if (idx === 0 && corr.band_type === 'low_shelf') {
            newBands.low = { gain: corr.gain, freq: corr.frequency, q: corr.q };
          } else if (corr.band_type === 'high_shelf') {
            newBands.high = { gain: corr.gain, freq: corr.frequency, q: corr.q };
          } else if (idx <= 4) {
            const bandKey = `band${idx}`;
            if (newBands[bandKey]) {
              newBands[bandKey] = { gain: corr.gain, freq: corr.frequency, q: corr.q };
            }
          }
        });
        setEqBands(newBands);
      }
    };
    
    const handleAutoEQApplyResult = (data) => {
      if (data.success) {
        setStatusMessage('EQ corrections applied to mixer');
      } else {
        setStatusMessage(`Failed to apply EQ: ${data.error || 'Unknown error'}`);
      }
    };
    
    const handleAutoEQResetResult = (data) => {
      if (data.success) {
        setStatusMessage('EQ reset to flat');
        // Reset local state
        setEqBands({
          low: { gain: 0, freq: 100, q: 1.0 },
          band1: { gain: 0, freq: 250, q: 2.0 },
          band2: { gain: 0, freq: 1000, q: 2.0 },
          band3: { gain: 0, freq: 3000, q: 2.0 },
          band4: { gain: 0, freq: 8000, q: 2.0 },
          high: { gain: 0, freq: 10000, q: 1.0 }
        });
        setCorrections([]);
      } else {
        setStatusMessage(`Failed to reset EQ: ${data.error || 'Unknown error'}`);
      }
    };
    
    // Multi-channel handlers
    const handleMultiChannelSpectrum = (data) => {
      const channel = data.channel;
      setChannelStatus(prev => ({ ...prev, [channel]: 'Analyzing' }));
      
      // Update spectrum if this is the selected channel for viewing
      const firstChannelId = selectedChannels && selectedChannels.length > 0 
        ? selectedChannels[0]  // selectedChannels is array of IDs
        : null;
      
      if (channel === selectedChannelForView || (!selectedChannelForView && channel === firstChannelId)) {
        if (data.spectrum && Array.isArray(data.spectrum)) {
          setSpectrumData(data.spectrum);
        }
        if (data.target_curve && Array.isArray(data.target_curve)) {
          setTargetCurve(data.target_curve);
        }
        setSpectralInfo({
          peak_freq: data.peak_freq || 0,
          centroid: data.centroid || 0,
          rolloff: data.rolloff || 0,
          flatness: data.flatness || 0
        });
      }
    };
    
    const handleMultiChannelCorrections = (data) => {
      const channel = data.channel;
      setChannelCorrections(prev => ({ ...prev, [channel]: data.corrections || [] }));
      setChannelStatus(prev => ({ ...prev, [channel]: 'Ready' }));
    };
    
    const handleMultiChannelStatus = (data) => {
      if (data.active !== undefined) {
        setIsActive(data.active);
      }
      if (data.message) {
        setStatusMessage(data.message);
      }
      if (data.channel && data.profile) {
        setChannelProfiles(prev => ({ ...prev, [data.channel]: data.profile }));
      }
    };
    
    const handleMultiChannelApplyResult = (data) => {
      if (data.success) {
        if (data.channel) {
          setChannelStatus(prev => ({ ...prev, [data.channel]: 'Applied' }));
        } else {
          setStatusMessage(data.message || 'Corrections applied');
        }
      } else {
        setStatusMessage(`Failed: ${data.error || 'Unknown error'}`);
      }
    };
    
    websocketService.on('auto_eq_status', handleAutoEQStatus);
    websocketService.on('auto_eq_spectrum', handleAutoEQSpectrum);
    websocketService.on('auto_eq_corrections', handleAutoEQCorrections);
    websocketService.on('auto_eq_apply_result', handleAutoEQApplyResult);
    websocketService.on('auto_eq_reset_result', handleAutoEQResetResult);
    
    const handleResetAllEQResult = (data) => {
      if (data.success) {
        setStatusMessage(data.message || `EQ reset for ${data.reset_count || 0} channels`);
      } else {
        setStatusMessage(`Failed to reset EQ: ${data.error || 'Unknown error'}`);
      }
    };
    websocketService.on('reset_all_eq_result', handleResetAllEQResult);
    
    // Multi-channel handlers
    websocketService.on('multi_channel_spectrum', handleMultiChannelSpectrum);
    websocketService.on('multi_channel_corrections', handleMultiChannelCorrections);
    websocketService.on('multi_channel_status', handleMultiChannelStatus);
    websocketService.on('multi_channel_auto_eq_status', handleMultiChannelStatus);
    websocketService.on('multi_channel_apply_result', handleMultiChannelApplyResult);
    
    return () => {
      websocketService.off('auto_eq_status', handleAutoEQStatus);
      websocketService.off('auto_eq_spectrum', handleAutoEQSpectrum);
      websocketService.off('auto_eq_corrections', handleAutoEQCorrections);
      websocketService.off('auto_eq_apply_result', handleAutoEQApplyResult);
      websocketService.off('auto_eq_reset_result', handleAutoEQResetResult);
      websocketService.off('reset_all_eq_result', handleResetAllEQResult);
      websocketService.off('multi_channel_spectrum', handleMultiChannelSpectrum);
      websocketService.off('multi_channel_corrections', handleMultiChannelCorrections);
      websocketService.off('multi_channel_status', handleMultiChannelStatus);
      websocketService.off('multi_channel_auto_eq_status', handleMultiChannelStatus);
      websocketService.off('multi_channel_apply_result', handleMultiChannelApplyResult);
    };
  }, [eqBands, selectedChannelForView, selectedChannels]);
  
  // Draw spectrum visualization
  const drawSpectrum = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;
    
    // Clear canvas
    ctx.fillStyle = '#0a0a1a';
    ctx.fillRect(0, 0, width, height);
    
    // Draw grid
    ctx.strokeStyle = 'rgba(0, 212, 255, 0.1)';
    ctx.lineWidth = 1;
    
    // Horizontal grid lines (dB levels)
    for (let i = 0; i <= 6; i++) {
      const y = (i / 6) * height;
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(width, y);
      ctx.stroke();
    }
    
    // Vertical grid lines (frequency bands)
    const numBands = spectrumData.length;
    for (let i = 0; i <= numBands; i += 4) {
      const x = (i / numBands) * width;
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, height);
      ctx.stroke();
    }
    
    // Draw spectrum bars
    const barWidth = (width / numBands) - 2;
    const gradient = ctx.createLinearGradient(0, height, 0, 0);
    gradient.addColorStop(0, '#00d4ff');
    gradient.addColorStop(0.5, '#00c851');
    gradient.addColorStop(1, '#ff9800');
    
    spectrumData.forEach((db, i) => {
      // Normalize dB to 0-1 range (-60 to 0 dB)
      const normalizedHeight = Math.max(0, Math.min(1, (db + 60) / 60));
      const barHeight = normalizedHeight * height;
      const x = (i / numBands) * width + 1;
      
      ctx.fillStyle = gradient;
      ctx.fillRect(x, height - barHeight, barWidth, barHeight);
    });
    
    // Draw target curve if enabled
    if (showTargetCurve && selectedProfile !== 'custom' && targetCurve && targetCurve.length > 0) {
      ctx.strokeStyle = 'rgba(255, 152, 0, 0.8)';
      ctx.lineWidth = 2;
      ctx.setLineDash([5, 5]);
      ctx.beginPath();
      
      // Normalize target curve values (assuming they're in dB, range roughly -10 to +10)
      const normalizeDb = (db) => {
        // Map dB range (-10 to +10) to canvas height (0 to height)
        // Center at middle of canvas
        const normalized = (db + 10) / 20; // 0 to 1
        return height - (normalized * height * 0.6 + height * 0.2); // Use middle 60% of canvas
      };
      
      const numTargetPoints = Math.min(targetCurve.length, numBands);
      for (let i = 0; i < numTargetPoints; i++) {
        const x = (i / numBands) * width;
        const y = normalizeDb(targetCurve[i] || 0);
        if (i === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
      }
      ctx.stroke();
      ctx.setLineDash([]);
    }
    
    // Continue animation
    if (isActive) {
      animationRef.current = requestAnimationFrame(drawSpectrum);
    }
  }, [spectrumData, showTargetCurve, selectedProfile, targetCurve, isActive]);
  
  // Start/stop animation when active state changes
  useEffect(() => {
    if (isActive) {
      animationRef.current = requestAnimationFrame(drawSpectrum);
    } else {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    }
    
    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [isActive, drawSpectrum]);
  
  // Resize canvas
  useEffect(() => {
    const canvas = canvasRef.current;
    if (canvas) {
      const container = canvas.parentElement;
      canvas.width = container.clientWidth;
      canvas.height = 200;
      drawSpectrum();
    }
  }, [drawSpectrum]);
  
  // Control handlers
  const handleStartAnalysis = () => {
    if (!selectedDevice) {
      setStatusMessage('Please select an audio device first');
      return;
    }
    
    websocketService.send({
      type: 'start_auto_eq',
      device_id: selectedDevice,
      channel: selectedChannel,
      profile: selectedProfile,
      auto_apply: autoApply
    });
    
    setStatusMessage('Starting analysis...');
  };
  
  const handleStopAnalysis = () => {
    websocketService.send({ type: 'stop_auto_eq' });
    setStatusMessage('Stopping analysis...');
  };
  
  const handleProfileChange = (e) => {
    const newProfile = e.target.value;
    setSelectedProfile(newProfile);
    
    if (isActive) {
      websocketService.send({
        type: 'set_eq_profile',
        profile: newProfile
      });
    }
  };
  
  const handleChannelChange = (e) => {
    setSelectedChannel(parseInt(e.target.value));
  };
  
  const handleApplyCorrection = () => {
    websocketService.send({ type: 'apply_eq_correction' });
    setStatusMessage('Applying corrections...');
  };
  
  const handleResetEQ = () => {
    websocketService.send({ 
      type: 'reset_eq',
      channel: selectedChannel
    });
    setStatusMessage('Resetting EQ...');
  };
  
  const handleResetAllEQ = () => {
    if (!selectedChannels || selectedChannels.length === 0) {
      setStatusMessage('No channels selected');
      return;
    }
    
    websocketService.send({ 
      type: 'reset_all_eq',
      channels: selectedChannels
    });
    setStatusMessage(`Resetting EQ for ${selectedChannels.length} channels...`);
  };
  
  // Batch mode handlers
  const handleStartBatchAnalysis = () => {
    if (!selectedDevice || !selectedChannels || selectedChannels.length === 0) {
      setStatusMessage('Please select an audio device and channels first');
      return;
    }
    
    // Ensure all channels have profiles (auto-detect if missing)
    const updatedProfiles = { ...channelProfiles };
    
    // Build channels config - selectedChannels is array of IDs
    const channelsConfig = selectedChannels.map(channelId => {
      // Find channel name from availableChannels
      const channelObj = availableChannels ? availableChannels.find(c => c.id === channelId) : null;
      const name = channelObj ? channelObj.name : null;
      
      // Auto-detect profile if not set
      if (!updatedProfiles[channelId]) {
        updatedProfiles[channelId] = detectProfileFromChannelName(name);
      }
      
      return {
        channel: channelId,
        profile: updatedProfiles[channelId] || 'custom',
        auto_apply: channelAutoApply[channelId] || false
      };
    });
    
    setChannelProfiles(updatedProfiles);
    
    console.log('Starting batch analysis with config:', channelsConfig);
    console.log('Selected device:', selectedDevice);
    console.log('Selected channels:', selectedChannels);
    
    // Convert selectedDevice to device index if it's an object
    let deviceId = selectedDevice;
    if (typeof selectedDevice === 'object' && selectedDevice.index !== undefined) {
      deviceId = selectedDevice.index;
    } else if (typeof selectedDevice === 'string') {
      // Try to find device index from audioDevices
      const device = audioDevices && audioDevices.find(d => 
        d.id === selectedDevice || 
        d.name === selectedDevice ||
        String(d.index) === String(selectedDevice)
      );
      if (device && device.index !== undefined) {
        deviceId = device.index;
      } else {
        // Try to parse as number
        const parsed = parseInt(selectedDevice, 10);
        if (!isNaN(parsed)) {
          deviceId = parsed;
        }
      }
    }
    
    console.log('Using device ID:', deviceId);
    
    websocketService.startMultiChannelAutoEQ(deviceId, channelsConfig);
    setStatusMessage(`Starting batch analysis for ${channelsConfig.length} channels...`);
    setIsActive(true);
  };
  
  const handleStopBatchAnalysis = () => {
    websocketService.stopMultiChannelAutoEQ();
    setStatusMessage('Stopping batch analysis...');
    setIsActive(false);
    setChannelStatus({});
  };
  
  const handleChannelProfileChange = (channelId, profile, event) => {
    // Stop event propagation to prevent row click
    if (event) {
      event.stopPropagation();
    }
    
    // Update only this specific channel's profile
    setChannelProfiles(prev => {
      const updated = { ...prev };
      updated[channelId] = profile;
      return updated;
    });
    
    // If analysis is active, update profile on backend
    if (isActive) {
      websocketService.setChannelProfile(channelId, profile);
    }
  };
  
  const handleChannelAutoApplyChange = (channelId, enabled) => {
    setChannelAutoApply(prev => ({ ...prev, [channelId]: enabled }));
  };
  
  const handleApplyChannelCorrection = (channelId) => {
    websocketService.applyChannelCorrection(channelId);
  };
  
  const handleApplyAllCorrections = () => {
    websocketService.applyAllCorrections();
    setStatusMessage('Applying corrections to all channels...');
  };
  
  const handleSelectChannelForView = (channelId) => {
    setSelectedChannelForView(channelId);
  };
  
  const handleBandChange = (bandKey, param, value) => {
    setEqBands(prev => ({
      ...prev,
      [bandKey]: {
        ...prev[bandKey],
        [param]: parseFloat(value)
      }
    }));
  };
  
  // Format frequency for display
  const formatFreq = (freq) => {
    if (freq >= 1000) {
      return `${(freq / 1000).toFixed(1)}k`;
    }
    return `${Math.round(freq)}`;
  };
  
  return (
    <div className="auto-eq-tab">
      <div className="auto-eq-section">
        <h2>Auto EQ - Spectrum Analysis</h2>
        
        {/* Auto EQ UI */}
        <div className="batch-mode-section">
            <h3>Process Selected Channels</h3>
            
            {/* Channels Table */}
            {selectedChannels && selectedChannels.length > 0 ? (
              <div className="batch-mode-table">
                <table>
                  <thead>
                    <tr>
                      <th>Channel</th>
                      <th>Profile</th>
                      <th>Auto-Apply</th>
                      <th>Status</th>
                      <th>Actions</th>
                    </tr>
                  </thead>
                  <tbody>
                    {selectedChannels.map(channelId => {
                      // selectedChannels is array of IDs, find channel object
                      const channelObj = availableChannels ? availableChannels.find(c => c.id === channelId) : null;
                      const name = channelObj ? channelObj.name : null;
                      
                      const displayName = name || `Ch ${channelId}`;
                      const currentProfile = channelProfiles[channelId] || detectProfileFromChannelName(name);
                      
                      return (
                        <tr 
                          key={channelId}
                          className={selectedChannelForView === channelId ? 'selected-row' : ''}
                          onClick={() => handleSelectChannelForView(channelId)}
                        >
                          <td>{displayName}</td>
                          <td>
                            <select 
                              value={currentProfile}
                              onChange={(e) => handleChannelProfileChange(channelId, e.target.value, e)}
                              onClick={(e) => e.stopPropagation()}
                              disabled={isActive}
                            >
                              {EQ_PROFILES.map(p => (
                                <option key={p.id} value={p.id}>
                                  {p.category ? `${p.category}: ${p.name}` : p.name}
                                </option>
                              ))}
                            </select>
                          </td>
                          <td>
                            <input 
                              type="checkbox"
                              checked={channelAutoApply[channelId] || false}
                              onChange={(e) => {
                                e.stopPropagation();
                                handleChannelAutoApplyChange(channelId, e.target.checked);
                              }}
                              disabled={isActive}
                            />
                          </td>
                          <td>{channelStatus[channelId] || 'Ready'}</td>
                          <td>
                            <button 
                              onClick={(e) => {
                                e.stopPropagation();
                                handleApplyChannelCorrection(channelId);
                              }}
                              disabled={!channelCorrections[channelId] || channelCorrections[channelId].length === 0}
                              className="btn-small"
                            >
                              Apply
                            </button>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            ) : (
              <div className="status-message">
                No channels selected. Select channels in "Select Channels to Process" section above.
                {availableChannels && availableChannels.length > 0 && (
                  <div style={{ marginTop: '10px', fontSize: '0.9em', color: '#888' }}>
                    Available channels: {availableChannels.length}
                  </div>
                )}
              </div>
            )}
            
            {/* Batch Mode Controls */}
            <div className="action-buttons">
              {!isActive ? (
                <button 
                  className="btn-auto-eq start"
                  onClick={handleStartBatchAnalysis}
                  disabled={!selectedDevice || !selectedChannels || selectedChannels.length === 0}
                  title={!selectedDevice ? 'Select audio device first' : (!selectedChannels || selectedChannels.length === 0) ? 'Select channels to process first' : 'Start analysis'}
                >
                  Start Analysis
                </button>
              ) : (
                <button 
                  className="btn-auto-eq stop"
                  onClick={handleStopBatchAnalysis}
                >
                  Stop Analysis
                </button>
              )}
              
              <button 
                className="btn-auto-eq apply"
                onClick={handleApplyAllCorrections}
                disabled={Object.keys(channelCorrections).length === 0}
                title={Object.keys(channelCorrections).length === 0 ? 'No corrections available' : 'Apply all calculated corrections to mixer'}
              >
                Apply All Corrections
              </button>
              
              <button 
                className="btn-auto-eq reset"
                onClick={handleResetAllEQ}
                disabled={!selectedChannels || selectedChannels.length === 0}
                title={!selectedChannels || selectedChannels.length === 0 ? 'Select channels first' : 'Reset EQ for all selected channels'}
              >
                Reset All EQ
              </button>
            </div>
          </div>
      </div>
    </div>
  );
}

export default AutoEQTab;
