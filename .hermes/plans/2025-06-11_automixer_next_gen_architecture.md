# Next-Generation Automixer Architecture Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Advance the AUTO-MIXER-Tubeslave project towards a fully autonomous AI Mixing Engineer capable of complete soundcheck, mixing, and live concert control with professional quality.

**Architecture:** Multi-agent system with four automation modes (Analysis → Assisted → Supervised → Full Autonomous), built on proven safety gates, offline replay validation, and incremental promotion.

**Tech Stack:** Python 3.10+, asyncio, PyTorch, OSC (Behringer WING), MIDI/TCP (dLive), WebSocket frontend, ChromaDB/RAG, virtual_mixer for testing.

---

## Phase 1: Core Infrastructure & Safety Foundation (Weeks 1-3)

### Task 1.1: Consolidate Automation Mode Implementation
**Objective:** Ensure all four modes (Analysis, Assisted, Supervised, Full Autonomous) have working runtime state machines with proper transitions.

**Files:**
- Modify: `backend/operator_mode_policy.py` - Add mode state machine with persistence
- Modify: `backend/autonomy_mode_transition_gate.py` - Add real-time mode transition logic
- Create: `backend/automation_mode_manager.py` - Centralized mode controller with audit trail

**Step 1: Write failing test**
```python
def test_automation_mode_state_machine():
    manager = AutomationModeManager()
    assert manager.current_mode == "analysis"
    manager.request_transition("assisted")  # Requires operator confirmation
    assert manager.pending_transition == "assisted"
    manager.confirm_transition()
    assert manager.current_mode == "assisted"
    # Supervised requires proof evidence
    manager.request_transition("supervised")
    assert manager.current_mode == "assisted"  # Blocked until proof
```

**Step 2: Run test to verify failure**

**Step 3: Write minimal implementation**

**Step 4: Run test to verify pass**

**Step 5: Commit**

---

### Task 1.2: Harden Safety Gates & Real-time Constraints
**Objective:** Make safety guarantees provable and auditable with sub-millisecond decision latency.

**Files:**
- Modify: `backend/autofoh_safety.py` - Add formal verification hooks
- Modify: `backend/thread_safety.py` - Add lock-free ring buffer for audio path
- Create: `backend/safety_audit_logger.py` - Immutable audit trail for every console write

**Step 1: Write failing test for latency budget (< 10ms decision loop)**
**Step 2: Implement lock-free audio path with `asyncio.Event` instead of `while RUNNING`**
**Step 3: Add safety invariant checks that cannot be bypassed**
**Step 4: Commit**

---

### Task 1.3: Unified Console State Model
**Objective:** Single source of truth for all mixer parameters with readback verification.

**Files:**
- Modify: `backend/mixer_state.py` - Add change detection, diff engine, snapshot versioning
- Modify: `backend/operator_product_state.py` - Integrate with physical readback contracts
- Create: `backend/console_state_synchronizer.py` - Bidirectional sync with conflict resolution

**Step 1: Write test for state convergence after network partition**
**Step 2: Implement CRDT-based state merging for multi-client scenarios**
**Step 3: Add automatic drift detection and reconciliation**
**Step 4: Commit**

---

## Phase 2: Audio Intelligence Pipeline (Weeks 4-6)

### Task 2.1: Multi-Channel Source Classification & Identity Tracking
**Objective:** Robust 25-class instrument classification with persistent identity across sessions.

**Files:**
- Modify: `backend/ml/channel_classifier.py` - Add online learning, confidence calibration
- Modify: `backend/channel_recognizer.py` - Integrate with source identity memory
- Create: `backend/source_identity_tracker.py` - Long-term identity persistence with bleed handling

**Step 1: Add test for classification stability across gain changes**
**Step 2: Implement spectral+embedding fusion classifier (MERT + traditional features)**
**Step 3: Add cross-session identity matching with confidence decay**
**Step 4: Commit**

---

### Task 2.2: Arrangement Role Detection & Musical Context
**Objective:** Understand song structure (verse/chorus/bridge) and instrument function dynamically.

**Files:**
- Modify: `backend/arrangement_role_proposal.py` - Add temporal modeling (HMM/Transformer)
- Create: `backend/musical_context_analyzer.py` - Section detection, energy contour, density analysis
- Create: `backend/genre_style_classifier.py` - Real-time genre/style recognition

**Step 1: Test verse/chorus detection on multitrack stems**
**Step 2: Implement section-aware processing parameter adaptation**
**Step 3: Add genre-specific target curves and processing defaults**
**Step 4: Commit**

---

### Task 2.3: Psychoacoustic & Perceptual Evaluation Engine
**Objective:** Human-aligned quality metrics replacing proxy measurements.

**Files:**
- Modify: `backend/perceptual/` - Integrate MERT/MuQ for perceptual similarity
- Create: `backend/perceptual/reference_matching.py` - Match against genre reference mixes
- Create: `backend/perceptual/quality_gate.py` - Block degrading changes automatically

**Step 1: Test perceptual metrics correlation with engineer ratings**
**Step 2: Implement A/B evaluation framework with statistical significance**
**Step 3: Add online perceptual learning from engineer accept/reject**
**Step 4: Commit**

---

## Phase 3: Processing Automation Families (Weeks 7-12)

### Task 3.1: Gain Staging & Level Management (Priority 1)
**Objective:** Complete supervised gain/fader automation with Ayaic Mix Monolith planes.

**Files:**
- Modify: `backend/lufs_gain_staging.py` - Add multi-bus coordination, stem-level balancing
- Modify: `backend/auto_fader.py` - Unify v1 and hybrid, add genre-level planes
- Create: `backend/gain_staging_coordinator.py` - Cross-channel gain staging with headroom optimization

**Target:** All 4 operating modes working for gain/fader with proof evidence.

**Step 1: Test gain staging on 64-channel virtual mixer with known references**
**Step 2: Implement Ayaic plane compliance checker**
**Step 3: Add stem/bus-level balancing alongside channels**
**Step 4: Commit**

---

### Task 3.2: Equalization & Spectral Balance (Priority 2)
**Objective:** Auto-EQ with unmasking, phase awareness, and feedback safety.

**Files:**
- Modify: `backend/auto_eq.py` - Add cross-channel unmasking, reference matching
- Modify: `backend/cross_adaptive_eq.py` - Make production-ready with readback
- Modify: `backend/phase_aware_eq_guard.py` - Integrate with EQ decisions
- Create: `backend/eq_decision_orchestrator.py` - Unified EQ strategy across all bands

**Target:** Assisted mode for EQ with operator proposals; Supervised for notch/HPF only.

**Step 1: Test EQ proposals against reference mixes across genres**
**Step 2: Implement unmasking with causal attribution (which channel masks which)**
**Step 3: Add phase coherence penalty to EQ cost function**
**Step 4: Commit**

---

### Task 3.3: Dynamics Processing (Priority 3)
**Objective:** Compression, gating, de-essing with musical intelligence.

**Files:**
- Modify: `backend/auto_compressor.py` / `auto_compressor_cf.py` - Unified compressor AI
- Modify: `backend/auto_gate_caig.py` - Production gate with bleed-aware thresholds
- Modify: `backend/deesser_vocal_sibilance.py` - Real-time sibilance tracking
- Create: `backend/dynamics_orchestrator.py` - Coordinated compression/gating/de-essing

**Target:** Proposal-only for full dynamics; Supervised for gain reduction < 3dB only.

**Step 1: Test compressor detection of instrument-type appropriate settings**
**Step 2: Implement adaptive attack/release based on musical role**
**Step 3: Add makeup gain with true-peak safety**
**Step 4: Commit**

---

### Task 3.4: Spatial Processing (Priority 4)
**Objective:** Panning, stereo width, phase alignment with mono compatibility.

**Files:**
- Modify: `backend/auto_panner.py` / `auto_panner_adaptive.py` - Unified spatial engine
- Modify: `backend/phase_alignment.py` - Real-time phase correction with mono check
- Create: `backend/spatial_orchestrator.py` - Coordinated pan/width/phase

**Target:** Proposal-only with visual frontend preview; Supervised for phase-only.

**Step 1: Test panning proposals against genre spatial templates**
**Step 2: Implement mono-compatibility constraint in cost function**
**Step 3: Add phase correlation monitoring with automatic correction guard**
**Step 4: Commit**

---

### Task 3.5: Effects & Send Management (Priority 5)
**Objective:** Reverb/delay/FX with musical timing and intelligent sends.

**Files:**
- Modify: `backend/auto_fx.py` / `auto_reverb.py` - AI-driven FX parameter selection
- Modify: `backend/fx_send_proposal.py` - Send level optimization with feedback safety
- Create: `backend/fx_orchestrator.py` - Global FX architecture management

**Target:** Proposal-only; Assisted for send levels on known sources.

**Step 1: Test FX type selection per instrument/genre**
**Step 2: Implement tempo-sync delay/reverb with musical timing**
**Step 3: Add FX return processing (EQ, compression on returns)**
**Step 4: Commit**

---

### Task 3.6: Master Bus Optimization (Priority 6)
**Objective:** Final polish with loudness compliance and translation checking.

**Files:**
- Modify: `backend/auto_mastering.py` - Master bus EQ/compression/limiting
- Modify: `backend/master_bus_optimization_proposal.py` - Proposal generation
- Create: `backend/master_quality_gate.py` - LUFS/TP/PEAK compliance checker

**Target:** Proposal-only with reference comparison; Assisted for limiting only.

**Step 1: Test master chain against streaming platform specs**
**Step 2: Implement multi-reference A/B (streaming, CD, vinyl targets)**
**Step 3: Add translation check (mono, small speakers, headphones)**
**Step 4: Commit**

---

## Phase 4: Live Performance & Adaptation (Weeks 13-16)

### Task 4.1: Real-time Musician Level Adaptation
**Objective:** Track and compensate for performer dynamics changes during show.

**Files:**
- Modify: `backend/performance_adaptation_proposal.py` - Add trend detection, rate limiting
- Create: `backend/live_adaptation_engine.py` - Sub-second response with safety envelope
- Create: `backend/musician_profile_tracker.py` - Per-artist behavior models

**Step 1: Simulate level drift scenarios in virtual mixer**
**Step 2: Implement adaptive gain riding with cooldown/hysteresis**
**Step 3: Add "set and forget" vs "continuous" modes per channel**
**Step 4: Commit**

---

### Task 4.2: Concert Flow & Section Management
**Objective:** Automatic song/section recognition and snapshot transitions.

**Files:**
- Modify: `backend/scene_preset_management_proposal.py` - Add timeline/trigger system
- Create: `backend/concert_timeline_manager.py` - Setlist, cues, auto-advance
- Create: `backend/section_transition_handler.py` - Smooth parameter interpolation

**Step 1: Test setlist-driven snapshot recall with crossfades**
**Step 2: Implement MIDI/OSC trigger integration**
**Step 3: Add manual override with graceful re-sync**
**Step 4: Commit**

---

### Task 4.3: Feedback & Emergency Handling
**Objective:** Sub-millisecond feedback suppression with graceful degradation.

**Files:**
- Modify: `backend/feedback_detector.py` - Multi-algorithm fusion (GCC-PHAT + ML)
- Create: `backend/emergency_response_coordinator.py` - Tiered response: notch → fader → mute
- Create: `backend/feedback_prevention.py` - Predictive gain/eq adjustment

**Step 1: Test feedback detection latency < 5ms on synthetic + real signals**
**Step 2: Implement automatic notch filter with musical frequency preservation**
**Step 3: Add preemptive gain reduction on feedback-prone channels**
**Step 4: Commit**

---

## Phase 5: Learning, Memory & Long-term Improvement (Weeks 17-20)

### Task 5.1: Long-term Memory & Decision Learning
**Objective:** Persistent learning from every session with operator feedback integration.

**Files:**
- Modify: `backend/long_term_memory_learning_proposal.py` - Add episodic + semantic memory
- Modify: `backend/decision_memory.py` - Vector similarity search for precedent
- Create: `backend/experience_replay_buffer.py` - Prioritized replay for offline training
- Create: `backend/memory_consolidation_scheduler.py` - Periodic consolidation to knowledge base

**Step 1: Test memory retrieval improves first-session decisions**
**Step 2: Implement operator correction → memory update pipeline**
**Step 3: Add cross-venue/artist transfer learning**
**Step 4: Commit**

---

### Task 5.2: Automated Soundcheck Pipeline
**Objective:** Fully autonomous soundcheck from line check to show-ready mix.

**Files:**
- Modify: `backend/auto_soundcheck_engine.py` - End-to-end orchestration
- Create: `backend/soundcheck_phase_orchestrator.py` - Line check → gain → EQ → dynamics → balance → FX → master
- Create: `backend/soundcheck_quality_gate.py` - Pass/fail criteria per phase

**Step 1: Test full soundcheck on virtual mixer with 32 channels**
**Step 2: Implement per-instrument soundcheck workflows**
**Step 3: Add "warm start" from previous venue/artist memory**
**Step 4: Commit**

---

### Task 5.3: Offline Validation & Regression Testing
**Objective:** Continuous integration with multitrack test corpus.

**Files:**
- Create: `backend/offline_validation_harness.py` - Expand to full regression suite
- Create: `backend/multitrack_test_corpus.py` - Curated test tracks with ground truth
- Create: `backend/regression_detector.py` - Statistical comparison of processing decisions

**Step 1: Build test corpus: 50+ multitracks across genres**
**Step 2: Implement automated A/B against engineer mixes**
**Step 3: Add CI pipeline with perceptual quality gates**
**Step 4: Commit**

---

## Phase 6: Multi-Console & Ecosystem Integration (Weeks 21-24)

### Task 6.1: dLive Full Parity
**Objective:** Equal capability on Allen & Heath dLive via MIDI/TCP.

**Files:**
- Modify: `backend/dlive_client.py` - Complete parameter mapping
- Create: `backend/dlive_parameter_map.py` - Bidirectional OSC↔MIDI translation
- Modify: All processing modules - Add dLive-specific address schemes

**Step 1: Map all WING processing surfaces to dLive equivalents**
**Step 2: Implement dLive scene/preset/snapshot control**
**Step 3: Test with dLive Director software (virtual console)**
**Step 4: Commit**

---

### Task 6.2: Dante/AES67/SoundGrid Audio Routing
**Objective:** Network audio integration for channel patching and recording.

**Files:**
- Modify: `backend/dante_routing_config.py` - Real-time routing control
- Create: `backend/network_audio_manager.py` - Discovery, subscription, redundancy
- Create: `backend/virtual_soundcheck_recorder.py` - Multitrack capture to DAW

**Step 1: Implement Dante controller with device discovery**
**Step 2: Add automatic failover for critical channels**
**Step 3: Integrate with virtual soundcheck workflow**
**Step 4: Commit**

---

### Task 6.3: Frontend & Operator Experience
**Objective:** Professional UI matching or exceeding vendor consoles.

**Files:**
- Modify: `frontend/src/components/` - Real-time parameter visualization
- Create: `frontend/src/automation_dashboard.tsx` - Mode status, proposals, evidence
- Create: `frontend/src/soundcheck_wizard.tsx` - Guided autonomous soundcheck

**Step 1: Implement real-time meter bridge with LUFS/TP**
**Step 2: Add proposal review UI with evidence packets**
**Step 3: Create mobile/tablet responsive layout for FOH position**
**Step 4: Commit**

---

## Phase 7: Production Hardening & Documentation (Weeks 25-28)

### Task 7.1: Reliability Engineering
**Objective:** 99.9% uptime during concert with graceful degradation.

**Files:**
- Create: `backend/health_monitor.py` - Component health, auto-restart, failover
- Create: `backend/watchdog_supervisor.py` - External process supervision
- Modify: `backend/server.py` - Graceful degradation modes

**Step 1: Chaos testing: network partitions, mixer disconnects, audio dropouts**
**Step 2: Implement automatic safe-mode (Analysis only) on critical failures**
**Step 3: Add concert-mode lock (prevents config changes during show)**
**Step 4: Commit**

---

### Task 7.2: Comprehensive Documentation
**Objective:** Complete technical and operator documentation.

**Files:**
- Create: `Docs/OPERATOR_MANUAL.md` - Mode workflows, safety procedures
- Create: `Docs/DEVELOPER_GUIDE.md` - Architecture, extension points, testing
- Create: `Docs/API_REFERENCE.md` - All public APIs, WebSocket messages
- Create: `Docs/SAFETY_MANUAL.md` - Risk analysis, emergency procedures

---

## Acceptance Criteria for Full Autonomous Mode

| Criterion | Target | Verification |
|-----------|--------|--------------|
| Soundcheck autonomy | 90%+ channels zero-touch | Virtual mixer + 10 real sessions |
| Gain staging accuracy | ±1.5 LUFS vs Ayaic planes | Multitrack corpus (50 tracks) |
| EQ match to reference | >0.85 perceptual similarity | MERT/MuQ evaluation |
| Feedback response | <10ms detection + suppression | Synthetic feedback injection |
| Level adaptation latency | <100ms end-to-end | Simulated drift scenarios |
| Decision explainability | 100% decisions have evidence packet | Audit log review |
| Safety incidents | 0 in 100 concert-hours | Production deployment |
| Mode transition safety | Zero unintended console writes | Transition gate tests |
| Long-term learning | Measurable improvement over 20 shows | A/B vs static config |

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Console firmware incompatibility | Abstract mixer client, test matrix in CI |
| Audio driver instability | Dedicated audio thread, watchdog restart |
| ML model drift | Online evaluation, automatic rollback |
| Operator trust erosion | Transparent evidence, gradual autonomy increase |
| Regulatory/liability | Comprehensive audit trail, insurance-grade docs |

---

## Next Immediate Steps (This Sprint)

1. **Implement AutomationModeManager** - Central state machine for 4 modes
2. **Add lock-free audio path** - Replace `while RUNNING` with `asyncio.Event.wait()`
3. **Complete source identity tracker** - Persistent cross-session identity
4. **Unify auto_fader v1/hybrid/v2** - Single production fader engine
5. **Build multitrack test corpus** - Foundation for all validation

---

*Plan created: 2025-06-11*
*Project: AUTO-MIXER-Tubeslave-main*
*Target: Next-Generation Autonomous AI Mixing Engineer*
