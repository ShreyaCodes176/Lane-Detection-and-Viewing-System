# Real-Time Curved Lane Detection System
This system detects road lanes from video footage and accurately highlights both straight and curved lanes using advanced Computer Vision, Image Processing, and Machine Learning-based smoothing techniques.

It processes each frame in real time, converts the road into a Bird’s Eye View, detects lane pixels using histogram and sliding window techniques, fits curved lane boundaries using polynomial regression, and overlays the detected lane area back onto the original video.

The project simulates core concepts used in:

* Autonomous Driving Systems
* Advanced Driver Assistance Systems (ADAS)
* Intelligent Transportation Systems

## ⚙️ Workflow

### 1. Perspective Transformation

The road frame is converted into a Bird’s Eye View using perspective warping to simplify lane detection.

### 2. Lane Extraction

The frame is processed using:
* HLS color masking
* Adaptive thresholding
* Binary segmentation
This isolates lane markings from the road surface.
### 3. Histogram-Based Lane Detection

A histogram of the lower half of the image identifies the approximate lane positions.
### 4. Sliding Window Search

A sliding window algorithm scans upward to collect lane pixels across the frame.
### 5. Polynomial Curve Fitting

Detected lane points are fitted using second-order polynomial regression.
This helps detect:
* Curved lanes
* Partial lane markings
* Continuous lane paths
This enables accurate curved lane visualization.

### 6. Temporal Smoothing

Previous lane fits are averaged to reduce flickering and stabilize detection across frames.

### 7. Lane Rendering

The detected lane area is projected back onto the original road frame.
The final output displays:
* Left lane boundary (Blue)
* Right lane boundary (Red)
* Highlighted drivable region (Green)

## ✨ Features

* Real-Time Lane Detection
* Curved Lane Tracking
* Bird’s Eye View Transformation
* Perspective Warping & Unwarping
* Sliding Window Lane Search
* Histogram-Based Lane Detection
* Polynomial Curve Fitting
* Temporal Smoothing for Stable Tracking
* Adaptive Thresholding
* Video Frame Processing
* Lane Overlay Visualization
* Supports Curved Roads

## 🛠️ Tech Stack

| Category              | Technologies                  |
| --------------------- | ----------------------------- |
| Programming Language  | Python                        |
| Computer Vision       | OpenCV                        |
| Numerical Computing   | NumPy                         |
| Video Processing      | OpenCV VideoWriter            |
| Image Processing      | HLS Color Space, Thresholding |
| Mathematical Modeling | Polynomial Regression         |
| Lane Tracking         | Sliding Window Algorithm      |
| Visualization         | OpenCV Drawing Functions      |


## 🧠 Concepts Used

### 📷 Computer Vision
* Perspective Transformation
* Bird’s Eye View Generation
* Lane Segmentation
* Image Masking

  
### 🖼️ Image Processing
* Grayscale Conversion
* Adaptive Thresholding
* Histogram Analysis
* Edge and Feature Extraction
### 📊 Data Science & Machine Learning Concepts
* Polynomial Curve Fitting
* Temporal Smoothing
* Feature Detection
* Real-Time Data Processing Pipelines
* Pattern Recognition
### 📐 Mathematics
* Perspective Geometry
* Coordinate Transformations
* Curve Estimation
* Matrix Transformations

## 🚀 Skills Demonstrated
* Computer Vision
* Real-Time Video Analytics
* OpenCV Development
* Perspective Geometry
* Image Processing
* Mathematical Modeling
* Polynomial Regression
* Data Processing Pipelines
