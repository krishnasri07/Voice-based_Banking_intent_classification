# Voice-Based Banking Intent Recognition

This project is a voice-enabled banking intent recognition system. It utilizes Natural Language Processing (NLP) and Speech Recognition to identify user intents, such as checking account balance, transferring money, and retrieving recent transactions.

## Features
- **Text-Based Intent Recognition:** Uses TF-IDF and Logistic Regression to classify user intents.
- **Speech-to-Text:** Integrates Google's Speech Recognition API for converting voice input to text.
- **Real-Time Audio Processing:** Utilizes PyAudio for capturing live voice input.
- **Whisper Model:** Implements Faster Whisper for advanced speech-to-text transcription.
- **Multithreading:** Handles real-time voice recognition efficiently.

## Technologies Used
- **Python** (for scripting and ML processing)
- **scikit-learn** (for NLP and intent classification)
- **pyaudio** (for real-time audio capture)
- **speech_recognition** (Google Speech-to-Text API integration)
- **faster-whisper** (for better speech recognition)
- **NumPy, Pandas** (for data processing)

## Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/Voice-Based-Banking-Intent-Recognition.git
   cd Voice-Based-Banking-Intent-Recognition
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

## Usage
1. Run the script:
   ```sh
   python source_code.py
   ```
2. Speak a banking-related command (e.g., *"Check my balance"*).
3. The system predicts the intent and displays the output.

## Example Output
```sh
Say something:
You said: "Transfer $500 to my savings account."
Predicted Intent: TransferMoney
```

## Future Enhancements
- Add support for more banking intents.
- Integrate with real banking APIs.
- Improve speech recognition accuracy.

## License
This project is licensed under the MIT License.
