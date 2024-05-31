# Facial Emotion Recognition with Music Recommendation
**Overview**
This project integrates facial emotion recognition using a Convolutional Neural Network (CNN) with a music recommendation system powered by Spotify. The application captures real-time video feed from a webcam, detects the user's facial emotion, and recommends music based on the detected emotion. It leverages deep learning and Spotify's API to provide a dynamic and interactive user experience.

* needs Spotify premium

### Key Features

Real-time Emotion Detection: Uses a CNN to classify emotions from webcam feed.

Music Recommendation: Recommends and plays songs based on detected emotions using Spotify's API.

User Controls: Includes playback controls for play, pause, next, and previous tracks.

### Prerequisites

Python 3.x

Required Python libraries: pandas, numpy, opencv-python, tensorflow, keras, Pillow, dotenv, spotipy, scikit-learn
Spotify developer account for API credentials

### Setup Instructions
1. Clone the Repository
code: git clone https://github.com/yourusername/emotion-music-recommendation.git
cd emotion-music-recommendation

2. Install Required Packages
code
pip install pandas numpy opencv-python tensorflow keras Pillow python-dotenv spotipy scikit-learn

3. Setup Spotify API Credentials
Create a .env file in the project directory with the following contents:
makefile

SPOTIPY_CLIENT_ID=your_spotify_client_id

SPOTIPY_CLIENT_SECRET=your_spotify_client_secret

SPOTIPY_REDIRECT_URI=your_spotify_redirect_uri

4. Prepare Dataset: kaggle datasets download -d msambare/fer2013

## Conclusion
This project successfully demonstrates how deep learning can be integrated with music recommendation systems to enhance user experience. 
By recognizing facial emotions in real-time and recommending appropriate music, it creates an engaging and interactive application. 
Future improvements could focus on increasing model accuracy, expanding the dataset, and adding more robust user controls. 

## Summary
The Facial Emotion Recognition with Music Recommendation project leverages deep learning and Spotify API to deliver a seamless and interactive user experience. 
The application detects emotions in real-time from webcam feed and recommends music accordingly, making it a novel approach to personalized music recommendations.
