# Kalman Navigation

An autonomous flight control, pose estimation, and vision system designed for micro unmanned aerial vehicles (UAVs). The system combines real-time ArUco marker detection with a Kalman Filter to maintain robust target tracking, continuous state prediction during visual occlusions, and precise trajectory control.

---

## Key Features

- **Predictive Target Tracking:** Uses a Kalman Filter to maintain pose estimation and target state vectors when visual line-of-sight is temporarily lost.
- **Computer Vision Pipeline:** Real-time marker identification, spatial error calculation, and vector overlay via OpenCV.
- **Closed-Loop PID Control:** Dynamic feedback loop driving flight commands for stable positioning and trajectory alignment.

---

## Repository Structure

```text
Kalman-Navigation/
├── README.md                 Project documentation
├── src/                      Core source code
│   ├── MissionControl.py     Main flight state machine & execution interface
│   ├── Control_Aruco.py      PID controller & Kalman state estimation engine
│   ├── coordinatesAruco.py   ArUco detection module & visual error calculation
│   └── __pycache__/
└── tests/                    Testing & diagnostic suite
    ├── ArucoTest.py          Standalone ArUco detector evaluation
    ├── camera.py             Generic camera utilities
    ├── CheckVel.py           Velocity command verification
    ├── telloCamera.py        Tello camera stream wrapper
    └── vision.py             Vision pipeline integration tests
