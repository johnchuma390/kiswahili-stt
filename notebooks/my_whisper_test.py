import torch
import numpy as np
import gradio as gr
import librosa
from transformers import WhisperProcessor, WhisperForConditionalGeneration

WHISPER_RATE = 16000

print("Loading Whisper-Small model...")
processor = WhisperProcessor.from_pretrained("openai/whisper-small")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
model.eval()
print("Model ready.\n")

def transcribe(audio):
    if audio is None:
        return "No audio received. Please record or upload a file."

    sampling_rate, audio_array = audio
    audio_array = audio_array.astype(np.float32)

    if audio_array.ndim > 1:
        audio_array = audio_array.mean(axis=1)

    if audio_array.max() > 1.0:
        audio_array = audio_array / 32768.0

    if sampling_rate != WHISPER_RATE:
        audio_array = librosa.resample(
            audio_array,
            orig_sr=sampling_rate,
            target_sr=WHISPER_RATE
        )

    inputs = processor(
        audio_array,
        sampling_rate=WHISPER_RATE,
        return_tensors="pt"
    )

    with torch.no_grad():
        predicted_ids = model.generate(
            inputs["input_features"],
            language="sw",
            task="transcribe"
        )

    result = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return result

app = gr.Interface(
    fn=transcribe,
    inputs=gr.Audio(
        sources=["microphone", "upload"],
        type="numpy",
        label="Speak Swahili or upload an audio file"
    ),
    outputs=gr.Textbox(
        label="Kiswahili Transcription",
        lines=4
    ),
    title="Kiswahili Speech-to-Text (Whisper Baseline)",
    description="Record your voice or upload a .wav/.mp3 file. Whisper will transcribe it to Kiswahili text.",
)

print("Starting Gradio app...")
print("Open your browser and go to the URL shown below.\n")
app.launch(share=False)
