import pandas as pd

# Updated dataset with banking intents
data = {
    'text': [
        # Check Account Balance
        "What's my account balance?",
        "Can you show me my balance?",
        "Check my current balance.",
        "I want to see my bank balance.",
        "Show me the available balance.",

        # Transfer Money
        "Transfer money to John.",
        "Send $500 to my savings account.",
        "I need to transfer funds to my checking account.",
        "Move $1000 to account number 123456.",
        "I want to transfer money.",

        # Get Last Five Transactions
        "Show my last five transactions.",
        "What are my recent transactions?",
        "Can I see my previous transactions?",
        "Display the last five transactions.",
        "I want to check my transaction history."
    ],
    'intent': [
        # Check Account Balance
        "CheckBalance", "CheckBalance", "CheckBalance", "CheckBalance", "CheckBalance",

        # Transfer Money
        "TransferMoney", "TransferMoney", "TransferMoney", "TransferMoney", "TransferMoney",

        # Get Last Five Transactions
        "GetLastTransactions", "GetLastTransactions", "GetLastTransactions", "GetLastTransactions", "GetLastTransactions"
    ]
}

df = pd.DataFrame(data)

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# Splitting the data
X = df['text']
y = df['intent']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorizing text data
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)
from sklearn.linear_model import LogisticRegression

# Training the model
model1 = LogisticRegression()
model1.fit(X_train_vec, y_train)
from sklearn.metrics import classification_report

# Making predictions
y_pred = model1.predict(X_test_vec)

# Evaluating the model
print(classification_report(y_test, y_pred))
def predict_intent(text):
    text_vec = vectorizer.transform([text])
    prediction = model1.predict(text_vec)
    return prediction[0]

# Example usage
user_input = "I want to fly to London"
predicted_intent = predict_intent(user_input)
print(f"Predicted Intent: {predicted_intent}")

import pyaudio
import numpy as np
import wave
from faster_whisper import WhisperModel
import threading
import speech_recognition as sr

# Set up parameters for real-time audio recording
FORMAT = pyaudio.paInt16  # 16-bit resolution
CHANNELS = 1  # Mono audio
RATE = 16000  # Sampling rate (Whisper models typically use 16000 Hz)
CHUNK = 1024  # Number of audio samples per buffer
RECORD_SECONDS = 5  # Duration for each recording chunk (adjust as necessary)

# Initialize Faster Whisper model (using CPU)
model = WhisperModel("tiny", device="cpu")  # You can use "tiny" for even faster performance on low-end CPUs


def recognize_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Say something:")
        audio = recognizer.listen(source)
        try:
            text = recognizer.recognize_google(audio)
            print(f"You said: {text}")

            return text
        except sr.UnknownValueError:
            print("Sorry, I could not understand the audio.")
            return None
        except sr.RequestError as e:
            print(f"Could not request results; {e}")
            return None


def transcribe_audio(audio_data):
    """Function to run Whisper on captured audio"""
    # print("Transcribing audio...")
    # Transcribe the real-time audio input
    segments, _ = model.transcribe(audio_data, language="en")  # Change language code if needed
    s = ""
    # Print the transcription results

    for segment in segments:
        # print(f"[{segment.start:.2f}s - {segment.end:.2f}s]: {segment.text}")
        # s+=segment
        if "Start" in segment.text:
            spoken_text = recognize_speech()
            if spoken_text:
                intent = predict_intent(spoken_text)
                print(f"Predicted Intent: {intent}")

    print(s)



def record_audio(stream, wave_file):
    """Capture audio in real time"""
    # print("Recording...")
    frames = []

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    # print("Recording finished.")

    # Save the recorded audio to a WAV file (optional)
    wf = wave.open(wave_file, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

    # Convert audio frames to NumPy array and normalize for Whisper input
    audio_data = np.frombuffer(b''.join(frames), dtype=np.int16).astype(np.float32) / 32768.0
    transcribe_audio(audio_data)


# Initialize PyAudio for real-time audio capture
p = pyaudio.PyAudio()

stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

# Start real-time recording and transcription in a loop
try:
    while True:
        # Run in a separate thread to avoid blocking the main thread
        t = threading.Thread(target=record_audio, args=(stream, "output.wav"))
        t.start()
        t.join()

except KeyboardInterrupt:
    # print(s)
    print("Stopping...")

finally:
    # Close the audio stream
    stream.stop_stream()
    stream.close()
    p.terminate()

