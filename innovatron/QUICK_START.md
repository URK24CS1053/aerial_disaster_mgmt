# Quick Reference - Running Your SAR System

## Three Ways to Run

### Way 1: Interactive Menu (Choose What You Want)
```powershell
cd decision_making
python main.py
# Prompts you to select mode 1 (test) or 2 (webcam)
```

### Way 2: Test Scenarios Only (Fast, No Hardware)
```powershell
cd decision_making
python run_test_scenarios.py
# Runs 6 victim scenarios in 2-3 seconds
# No webcam required
```

### Way 3: Live Webcam Only (Real Perception)
```powershell
cd decision_making
python run_live_webcam.py
# Opens webcam
# Press Q to quit, P to pause, S to save snapshot
```

---

## Data Flow

### Test Mode (run_test_scenarios.py)
```
Hardcoded Victim Data
    ↓
PerceptionOutput (14-element feature vectors)
    ↓
ML Inference (RandomForest)
    ↓
VictimState
    ↓
Urgency Scoring
    ↓
Results: CRITICAL/HIGH/MODERATE/NONE
```

### Live Mode (run_live_webcam.py)
```
Webcam Frame
    ↓
YOLOv8 Person Detection
    ↓
PerceptionOutput (14-element feature vectors)
    ↓
ML Inference (RandomForest)
    ↓
VictimState
    ↓
Urgency Scoring
    ↓
Results: CRITICAL/HIGH/MODERATE/NONE
    ↓
Display on Video + Logging
```

---

## Live Webcam Controls

| Key | Action |
|-----|--------|
| **Q** | Quit |
| **P** | Pause/Resume |
| **S** | Save snapshot |

---

## Expected Output

### Test Mode
```
[OK] 6/6 scenarios passed
    [PASS] Responsive Healthy Victim
    [PASS] Critical - Unresponsive with Weak Vitals
    [PASS] Moderate Injury + High Environmental Risk
    [PASS] Weak Response with Uncertain Vitals
    [PASS] False Alarm - No Human Detected
    [PASS] Building Collapse Survivor - Critical
```

### Live Mode
```
Frame: 123 | Persons: 1 | Status: RUNNING

Detection Results:
  - Person detected (confidence: 0.92)
  - Distance: 1.5m
  - Responsiveness: RESPONSIVE
  - Vital Signs: STABLE_SIGNS
  - Urgency: MODERATE
```

---

## System Status

✅ Environment: Configured
✅ Dependencies: Installed (numpy, scikit-learn, opencv-python)
✅ ML Model: Trained (297 samples, 14 features)
✅ Architecture: Refactored (strict layer separation)
✅ Logging: Active

---

## What's New

✅ **Live Webcam Support** - Real human detection from camera
✅ **Dual Modes** - Test (hardcoded) + Live (camera)
✅ **Interactive Menu** - Easy mode selection
✅ **Quick Scripts** - run_test_scenarios.py and run_live_webcam.py
✅ **Full Pipeline** - Perception → Fusion → Urgency → Logging
✅ **Real Features** - 14-element vectors from actual detections

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "Cannot open webcam" | Check camera is connected; try test mode |
| No persons detected | Ensure face/body in frame, good lighting |
| Slow performance | First run slower (model loading); try HOG mode |
| Import errors | Check venv is activated and dependencies installed |

---

## Complete Commands (Copy-Paste Ready)

**Test Mode (from project root):**
```powershell
cd c:\Users\Manendra\Desktop\innovatron\aerial_disaster_mngmt\decision_making
& "C:\Users\Manendra\Desktop\innovatron\.venv\Scripts\python.exe" run_test_scenarios.py
```

**Live Mode (from project root):**
```powershell
cd c:\Users\Manendra\Desktop\innovatron\aerial_disaster_mngmt\decision_making
& "C:\Users\Manendra\Desktop\innovatron\.venv\Scripts\python.exe" run_live_webcam.py
```

**With Activation Script:**
```powershell
cd c:\Users\Manendra\Desktop\innovatron
.\.venv\Scripts\Activate.ps1
cd aerial_disaster_mngmt\decision_making
python run_test_scenarios.py
# or
python run_live_webcam.py
```

---

## Documentation Files

- `RUNNING_MODES.md` - Complete guide to both modes
- `LIVE_PERCEPTION_READY.md` - What changed
- `REFACTORING_ARCHITECTURE.md` - System architecture
- `SETUP_GUIDE_WINDOWS.md` - Environment setup
- `EXECUTION_AUDIT_COMPLETE.md` - Full audit report

---

**Last Updated:** January 29, 2026
**Status:** ✅ Production Ready
