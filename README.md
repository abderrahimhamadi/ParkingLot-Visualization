# Parking Space Counter

This Python project uses machine learning and computer vision to count available parking spaces in real-time from a video feed. It consists of two main components: model training and real-time parking space visualization.

## Project Structure

```

.
├── clf-data/                     # Data for classifier training
│   ├── empty/                    # Images of empty parking spaces
│   └── not_empty/                # Images of occupied parking spaces
├── data/                         # Data used by the visualization script
│   ├── parking_1920_1080_loop.mp4  # Demo video for visualization
│   └── mask_1920_1080.png          # Mask image for parking spot delineation
├── [main.py](http://main.py/)                       # Main script to run both preparation and visualization
├── preparing_model.py            # Script for training the parking space classifier
├── [README.md](http://readme.md/)                     # Project documentation
├── requirements.txt              # Python dependencies for the project
├── [util.py](http://util.py/)                       # Utility functions for the project
└── visualizing_parking.py        # Script for visualizing parking space occupancy

```

## Setup

### Requirements

- Python 3.8.19
- Dependencies: `numpy`, `opencv-python`, `scikit-image`, `scikit-learn`, `optuna`, `pickle`

### Installation

Clone the repository and install the required Python packages:

```bash
git clone [repository-url]
cd [repository-directory]
pip install -r requirements.txt

```

Link to data used for training and testing the model that Achieved **100 % Accuracy score** :  https://drive.google.com/file/d/13lWZFPajKRvD4LD1HDQi8KdpH2djsCgp/view?usp=sharing

## Usage

### Model Preparation

Run `preparing_model.py` to train a classifier to distinguish between empty and occupied parking spaces:

```bash
python preparing_model.py

```

This script performs the following operations:

- Loads images from the `clf-data` directory.
- Resizes images and flattens them into a feature vector.
- Splits data into training and testing sets.
- Uses Optuna to optimize hyperparameters for an SVM classifier.
- Trains the SVM classifier with the best found hyperparameters.
- Saves the trained model to `model.p` for later use.

### Visualizing Parking

Run `visualizing_parking.py` to start the real-time parking space counter:

```bash
python visualizing_parking.py

```

This script processes a video feed to detect and count available parking spots:

- Applies a predefined mask to identify parking spot regions.
- Uses the previously trained model to determine if spots are empty or occupied.
- Displays the video with rectangles around parking spots (green for empty, red for occupied).
- Updates the count of available spots in real-time on the video feed.

Press 'q' to quit the video feed.

### Performance Optimization:

#### Selective Frame Processing 
To enhance performance and reduce computational load, the script captures and processes one frame every 60 frames. This frequency is based on the observation that most parking spots do not change status frequently, thus eliminating the need for constant checking.
#### Change Detection
Integrated a function to detect significant differences between the current state and the last recorded state of each parking spot. This method ensures that the system only updates spot statuses when a noticeable change is detected, further optimizing performance.

## Contributing

Contributions are welcome! Please fork the repository, make your changes, and submit a pull request.
