# Posture and Emotions Dectection

A single, on-device Python pipeline that uses one camera feed (your laptop webcam) to
classify:
1. Posture:
○ Binary: Upright vs. Hunched
2. Emotion:
○ Multi‐Class (3+ classes, e.g., relaxed, stressed, angry, etc.)

For pose estimation [Yolov11](https://docs.ultralytics.com/tasks/pose/) was used, and posture is derived using key-points. For emotions classifcation - [emotion](https://github.com/George-Ogden/emotion) was used, which uses repVGG for faster and accurate predictions.

## Setup
```
conda create -n <name> python=3.11.11
conda activate <name>
pip install -r requirements.txt
```

## Models
1. Pose estimation: Yolov11 (tiny) provides a fast and acurate pose estimation models, very easy to intergrate.
2. Emotions classification: RepVGG model is used to classify face crops into 8 catgories of emotions, namely:
- Anger
- Contempt
- Disgust
- Fear
- Happy
- Relax
- Sad
- Surprise

## Running
```
python main.py
```

### options:
- -v: display version 
- -d: display camera feeds with key-points

_with display model (-d) press `esc` to exit, otherwise press `ctrl+c`._

## Performance
- Machine: Intel(R) Core(TM) i5-6300U CPU @ 2.40GHz
- FPS: 7-8 FPS
- CPU: 60-70%
- MEM: 10%

_running it on a gpu enabled system would easily achive 15 FPS._

## Future Improvements
- Posture detection is done using very naieve thresholding. Which can be improved using a CNN based predictor for different postures - “Upright,” “Slight Lean,” “Severe Hunch” etc.
- Emotion classification model can made faster using mobile-net variants of architecture.
- Models can be converted to hardware accelated formats for faster inference.