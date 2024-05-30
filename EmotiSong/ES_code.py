import pandas as pd
import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input, Dropout
import PIL 
import os
import dotenv
import tkinter as tk
from tkinter import messagebox
import threading
import csv
from datetime import datetime

# Load env var from .env file
dotenv.load_dotenv(r"  ")

# Debug Print environment variables 
print("SPOTIPY_CLIENT_ID:", os.getenv('SPOTIPY_CLIENT_ID'))
print("SPOTIPY_CLIENT_SECRET:", os.getenv('SPOTIPY_CLIENT_SECRET'))
print("SPOTIPY_REDIRECT_URI:", os.getenv('SPOTIPY_REDIRECT_URI'))

# Verifing Spotipy install
try:
    import spotipy
    from spotipy.oauth2 import SpotifyOAuth
except ImportError:
    print("Spotipy not installed. Please install")
    raise

# Load data
train_dir = r"train"
test_dir = r"test"

# Img para
img_width, img_height = 48, 48

# Hyperpara
batch_size = 64
epochs =  255

# Creating data gene
datagen = ImageDataGenerator(rescale=1./255)

train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    color_mode='grayscale',
    class_mode='categorical'
)

test_generator = datagen.flow_from_directory(
    test_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    color_mode='grayscale',
    class_mode='categorical'
)

# Define model with Input(shape)
input_shape = (img_width, img_height, 1)
model = Sequential([
    Input(shape=input_shape),
    Conv2D(32, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    
    Conv2D(128, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(7, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
model.fit(
    train_generator,
    epochs=epochs,
    validation_data=test_generator
)

# Spotify authentication using env var
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
    client_id=os.getenv('SPOTIPY_CLIENT_ID'),
    client_secret=os.getenv('SPOTIPY_CLIENT_SECRET'),
    redirect_uri=os.getenv('SPOTIPY_REDIRECT_URI'),
    scope='playlist-read-private playlist-modify-public user-library-read user-modify-playback-state user-read-playback-state',
    cache_path=".spotify_cache"  # Specify cache path
))

# Func: get song recommendation based on detected emotion
def get_song_recommendation(emotion):
    emotion_playlists = {
        'angry': 'spotify:playlist:37i9dQZF1DXdPec7aLTmlC',  # Updated valid playlist ID
        'disgust': 'spotify:playlist:37i9dQZF1DWXIcbzpLauPS',  # Updated valid playlist ID
        'fear': 'spotify:playlist:37i9dQZF1DWZwtERXCS82H',     # Updated valid playlist ID
        'happy': 'spotify:playlist:37i9dQZF1DX9XIFQuFvzM4',    # Updated valid playlist ID
        'sad': 'spotify:playlist:37i9dQZF1DX3YSRoSdA634',      # Updated valid playlist ID
        'surprise': 'spotify:playlist:37i9dQZF1DX4SBhb3fqCJd', # Updated valid playlist ID
        'neutral': 'spotify:playlist:37i9dQZF1DWYMroOc5KTTh'   # Updated valid playlist ID
    }
    
    playlist_id = emotion_playlists.get(emotion, 'spotify:playlist:37i9dQZF1DX3rxVfibe1L0')
    
    try:
        results = sp.playlist_tracks(playlist_id)
        return results['items']
    except spotipy.exceptions.SpotifyException as e:
        print(f"Error fetching playlist: {e}")
        return []

# Global var for controlling playback
current_playlist = []
current_track_index = 0
is_playing = False

# Emotion labels
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# Func: play a song using Spotify's Web Playback SDK
def play_song():
    global current_playlist, current_track_index, is_playing
    
    if not current_playlist:
        return
    
    # Ensure playback starts from the beginning of the playlist if shuffling is disabled
    track_uri = current_playlist[current_track_index]['track']['uri']
    
    # Get available devices and set the first one as active
    devices = sp.devices()
    if not devices['devices']:
        print("No active device found. Please open Spotify on a device.")
        return
    
    active_device_id = devices['devices'][0]['id']
    sp.transfer_playback(active_device_id)
    
    sp.start_playback(device_id=active_device_id, uris=[track_uri])
    is_playing = True

# Func: handle playback control (next, previous, pause, resume)
def control_playback(action):
    global current_track_index, is_playing
    
    if action == 'next':
        current_track_index = (current_track_index + 1) % len(current_playlist)
    elif action == 'previous':
        current_track_index = (current_track_index - 1) % len(current_playlist)
    elif action == 'pause':
        sp.pause_playback()
        is_playing = False
    elif action == 'resume':
        sp.start_playback()
        is_playing = True

# Func: predict emotion from webcam feed
def predict_emotion(image):
    processed_image = image.reshape(1, img_width, img_height, 1)
    prediction = model.predict(processed_image)
    emotion = emotion_labels[np.argmax(prediction)]
    return emotion

# Real-time emotion detect with music recommendation
def start_webcam():
    global current_playlist, current_track_index, is_playing
    
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)
        
        for (x, y, w, h) in faces:
            roi_gray = gray_frame[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray, (img_width, img_height))
            emotion = predict_emotion(roi_gray)
            
            # Get playlist based on detected emotion
            current_playlist = get_song_recommendation(emotion)
            current_track_index = 0  # Reset to the first track
            
            # Play the first track in the playlist
            if not is_playing:
                play_song()
            
            cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        cv2.imshow('Facial Emotion Recognition with Music Recommendation', frame)
        
        # Handle user input to control playback
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('n'):  # Play next track
            control_playback('next')
        elif key == ord('p'):  # Play previous track
            control_playback('previous')
        elif key == ord('s'):  # Stop playback
            control_playback('pause')
        elif key == ord('r'):  # Resume playback
            control_playback('resume')
    
    cap.release()
    cv2.destroyAllWindows()

start_webcam()
