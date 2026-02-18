# H.E.R.O. System - Main Package

## Structure

### `cognitive_tests/`
Cognitive assessment tests (memory, reaction time, drawing, etc.)
- Tests cognitive function and decline
- All follow standard test interface pattern

### `biosensors/`
Biosensor integration modules
- `eeg/` - EEG data acquisition and processing
- `eye_tracking/` - Eye tracking and gaze analysis  
- `wearable/` - Heart rate, oximeter, accelerometer

### `affective_computing/`
Emotion recognition and facial analysis
- Uses MediaPipe for facial landmarks
- ML models for emotion classification

### `consultation/`
User interface and consultation orchestration
- Avatar system
- Display management
- Test sequencing

### `data/`
Data handling and storage
- Database access
- Data loading/saving
- Data processing

### `utils/`
Shared utility functions
```

---

### **Step 6: What Your Structure Looks Like Now**
```
hero-monitor/
├── hero/                           ✅ NEW!
│   ├── __init__.py
│   ├── README.md
│   ├── cognitive_tests/            ✅ Your converted tests
│   │   ├── __init__.py
│   │   ├── speed_test.py           ✅ Done
│   │   └── memory_test.py          ✅ Done
│   ├── biosensors/                 ✅ Ready for YOUR additions
│   │   ├── __init__.py
│   │   ├── eeg/
│   │   ├── eye_tracking/
│   │   └── wearable/
│   ├── affective_computing/        ⏳ To migrate
│   ├── consultation/               ⏳ To migrate
│   ├── data/                       ⏳ To migrate
│   └── utils/                      ⏳ To migrate
│
├── consultation/                   📦 OLD (will migrate from here)
├── games/                          📦 OLD (will convert from here)
├── affective_computing/            📦 OLD (will migrate)
└── ...