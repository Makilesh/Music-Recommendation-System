import streamlit as st
import random
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import torch
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

# Spotify API credentials (replace these with your actual credentials)
CLIENT_ID = "bb0b045fd43e42d4a7b8efef6b1af290"
CLIENT_SECRET = "6d06b9f8bb2b400ea8b73b395d7b1b22"

# Authenticate with Spotify
client_credentials_manager = SpotifyClientCredentials(client_id=CLIENT_ID, client_secret=CLIENT_SECRET)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

# Load YOLO emotion detection model
model_path = r"best (1).pt"  # Update with your model path
model = YOLO(model_path)

# Emotion labels from the model (YOLO custom labels)
emotion_labels = {0: 'anger', 1: 'fear', 2: 'happy', 3: 'neutral', 4: 'sad'}

# Confidence threshold for detection
CONFIDENCE_THRESHOLD = 0.3  # Lowering it to allow more detections

# Map detected emotion to song search term
def emotion_to_song_query(emotion):
    if emotion == "happy":
        return "happy music"
    elif emotion == "sad":
        return "sad music"
    elif emotion == "angry":
        return "angry music"
    elif emotion == "fear":
        return "fear music"
    else:
        return "calm music"  # Default for neutral or unrecognized emotion

# Function to search for a song
def search_song(query):
    results = sp.search(q=query, limit=10, type='track')
    tracks = results['tracks']['items']
    if tracks:
        return random.choice(tracks)
    else:
        return None

# Function to get multiple song recommendations
def get_multiple_song_recommendations(query):
    results = sp.search(q=query, limit=5, type='track')
    tracks = results['tracks']['items']
    return tracks

# Display a friendly greeting based on the time of day
def get_greeting():
    import datetime
    current_hour = datetime.datetime.now().hour
    if current_hour < 12:
        return "Good Morning! 🌅"
    elif 12 <= current_hour < 18:
        return "Good Afternoon! 🌞"
    else:
        return "Good Evening! 🌙"

# Fun facts based on detected emotions
def fun_fact_for_emotion(emotion):
    facts = {
        "happy": "Did you know? Smiling can reduce stress and boost your mood!",
        "sad": "Sadness is just temporary! Tomorrow will be a better day.",
        "angry": "Anger is a natural emotion, but learning to control it can improve your well-being.",
        "fear": "Facing your fears can make you stronger and more resilient.",
        "neutral": "Sometimes, being neutral is the best state of mind to make clear decisions!"
    }
    return facts.get(emotion, "Every emotion has its own importance. Embrace them!")

# Set up a simple header and greeting
st.title("🎶 Emotion Detection & Music Recommendation 🎶")
st.markdown("""
This app detects your facial emotion using the camera, and recommends a song based on your mood.
Let's discover a song that suits your mood! 😊
""")

# Add a personalized greeting
st.subheader(get_greeting())

# Emotion History: Store previous emotions detected (temporary session-based storage)
if 'emotion_history' not in st.session_state:
    st.session_state.emotion_history = []

# Start Button to initiate emotion detection
if st.button("Start Emotion Detection"):
    st.spinner("Detecting emotion... Please look at the camera! 👀")
    
    # Capture frame from webcam
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()

    if ret:
        # Convert frame from BGR (OpenCV format) to RGB (YOLO format)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame for emotion detection
        results = model(frame_rgb)

        detected_emotion = None
        max_confidence = 0  # To track the highest confidence

        # Check if the results contain boxes and process the detections
        if len(results[0].boxes.data) > 0:
            # Loop through the detected objects (the results.xyxy[0] contains detections)
            for detection in results[0].boxes.data.tolist():
                class_id = int(detection[5])  # Class id for the detected object
                confidence = detection[4]  # Confidence score

                # If this detection has the highest confidence so far and above the threshold
                if confidence > CONFIDENCE_THRESHOLD and confidence > max_confidence:
                    detected_emotion = emotion_labels.get(class_id, None)
                    max_confidence = confidence

        if detected_emotion:
            # Display detected emotion and suggest a song
            st.subheader(f"🙂 Detected Emotion: {detected_emotion.capitalize()}")

            # Change the background color based on the emotion
            color_theme = {
                "happy": "#f7f9c5",
                "sad": "#a8a8d1",
                "angry": "#f57c00",
                "fear": "#99b3ff",
                "neutral": "#d3d3d3"
            }

            st.markdown(f"<style>body{{background-color: {color_theme.get(detected_emotion, '#d3d3d3')}}}</style>", unsafe_allow_html=True)

            # Fun fact for the detected emotion
            fun_fact = fun_fact_for_emotion(detected_emotion)
            st.write(f"💡 Fun Fact: {fun_fact}")

            # Convert emotion to song query and search for songs
            song_query = emotion_to_song_query(detected_emotion)
            suggested_songs = get_multiple_song_recommendations(song_query)

            if suggested_songs:
                st.write("Here are some song recommendations for you!")

                for song in suggested_songs:
                    song_name = song['name']
                    song_artist = song['artists'][0]['name']
                    song_url = song['external_urls']['spotify']
                    st.write(f"- {song_name} by {song_artist}")
                    st.markdown(f"[Listen on Spotify]({song_url})")
                    st.markdown(f"""
                    <iframe src="https://open.spotify.com/embed/track/{song['id']}" width="300" height="80" frameBorder="0" allowtransparency="true" allow="encrypted-media"></iframe>
                    """, unsafe_allow_html=True)

            else:
                st.write("Sorry, no songs could be found for the detected emotion.")
            
            # Store the detected emotion in history
            st.session_state.emotion_history.append(detected_emotion)
        else:
            st.write("No emotion detected in the frame.")
    else:
        st.write("Failed to capture image from the camera.")

# Display emotion history
st.subheader("Emotion History 🗂")
if st.session_state.emotion_history:
    st.write("Detected emotions in this session:")
    for emotion in st.session_state.emotion_history:
        st.write(f"- {emotion.capitalize()}")
else:
    st.write("No emotions detected yet.")

# Additional section for user feedback
st.markdown("""
---
### 🎤 Give Us Your Feedback!
Please let us know how we can improve the app or share your thoughts! 😊
""")
user_feedback = st.text_area("Your feedback:", height=100)

# Allow users to submit feedback
if st.button("Submit Feedback"):
    print(f"User Feedback: {user_feedback}")
    if user_feedback:
        st.write("Thank you for your feedback! 🙏")
    else:
        st.write("Please enter some feedback before submitting.")

# Add some extra space and footer
st.markdown("""
---
Made with ❤ using Streamlit and YOLO model for emotion detection.
""")