import os
import whisper
from google.colab import drive

def mount_google_drive():
    """Mount Google Drive to access files."""
    drive.mount('/content/drive')
    print("Google Drive mounted successfully.")

def list_audio_files(folder_path):
    """List all audio/video files in the specified folder."""
    return [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.endswith(('.mp3', '.mp4', '.wav'))
    ]

def transcribe_files(files, model):
    """Transcribe a list of audio files using Whisper."""
    for file_path in files:
        print(f"Processing: {file_path}")
        result = model.transcribe(file_path)
        
        # Save transcription to a text file
        output_file = file_path.replace('.mp3', '.txt').replace('.mp4', '.txt').replace('.wav', '.txt')
        with open(output_file, "w") as f:
            f.write(result["text"])
        print(f"Saved transcription to: {output_file}")

def main():
    # Mount Google Drive
    mount_google_drive()

    # Define the folder path in Google Drive
    target_folder = "/content/drive/My Drive/TranscriptionFolder"

    # List audio/video files
    audio_files = list_audio_files(target_folder)
    if not audio_files:
        print("No audio/video files found in the specified folder.")
        return

    # Load the Whisper model
    model = whisper.load_model("medium")  # Change model size if needed

    # Transcribe the files
    transcribe_files(audio_files, model)

if __name__ == "__main__":
    main()
