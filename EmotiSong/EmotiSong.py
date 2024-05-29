import pandas as pd
import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input, Dropout
import PIL 


# Verifing Spotipy install
try:
    import spotipy
    from spotipy.oauth2 import SpotifyOAuth
except ImportError:
    print("Spotipy not installed. Please install")
    raise

# Load data
train_dir = r"D:\Zip extracts\train"
test_dir = r"D:\Zip extracts\test"

# Img para
img_width, img_height = 48, 48

# Hyperpara
batch_size = 64
epochs = 105

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

# Retrieve sensitive info from enviro var
client_id = os.getenv('SPOTIPY_CLIENT_ID')
client_secret = os.getenv('SPOTIPY_CLIENT_SECRET')
redirect_uri = os.getenv('SPOTIPY_REDIRECT_URI')

# Validate when enviro var are set
if not all([client_id, client_secret, redirect_uri]):
    raise ValueError("Missing environment variables. Please set SPOTIPY_CLIENT_ID, SPOTIPY_CLIENT_SECRET, and SPOTIPY_REDIRECT_URI.")

# Spotify auth using enviro var
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id=client_id,
                                               client_secret=client_secret,
                                               redirect_uri=redirect_uri,
                                               scope='playlist-read-private playlist-modify-public user-library-read user-modify-playback-state user-read-playback-state'))

# Func: get song recommendation based on emo
def get_song_recommendation(emotion):
    emotion_playlists = {
        'angry': 'spotify:playlist:37i9dQZF1DWXIcbzpLauPS',
        'disgust': 'spotify:playlist:37i9dQZF1DXa2PsvJSPnPf',
        'fear': 'spotify:playlist:37i9dQZF1DX7XfIh3w0THN',
        'happy': 'spotify:playlist:37i9dQZF1DXdPec7aLTmlC',
        'sad': 'spotify:playlist:37i9dQZF1DX7qK8ma5wgG1',
        'surprise': 'spotify:playlist:37i9dQZF1DX4UtSsGT1Sbe',
        'neutral': 'spotify:playlist:37i9dQZF1DWYBO1MoTDhZI'
    }
    playlist_id = emotion_playlists.get(emotion, 'spotify:playlist:37i9dQZF1DX3rxVfibe1L0')
    results = sp.playlist_tracks(playlist_id)
    track = results['items'][0]['track']['name']
    artist = results['items'][0]['track']['artists'][0]['name']
    track_uri = results['items'][0]['track']['uri']
    return track, artist, track_uri

# Func: play song using Spotify web playback SDK
def play_song(track_uri):
    sp.start_playback(uris=[track_uri])

# Emo labels
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# Func: predict emo
def predict_emotion(image):
    processed_image = image.reshape(1, img_width, img_height, 1)
    prediction = model.predict(processed_image)
    emotion = emotion_labels[np.argmax(prediction)]
    track, artist, track_uri = get_song_recommendation(emotion)
    print(f"Playing {track} by {artist} for {emotion} mood")
    play_song(track_uri)
    return emotion

# Real-time emo detect with music recommend
def start_webcam():
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
            cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.imshow('Facial Emotion Recognition with Music Recommendation', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

start_webcam()
