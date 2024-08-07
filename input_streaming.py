import torch
import torchaudio
import pyaudio
import numpy as np
import cv2
import librosa
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from cnn import CNNNetwork
from urbansounddataset import UrbanSoundDataset
from train import AUDIO_DIR, ANNOTATIONS_FILE, SAMPLE_RATE, NUM_SAMPLES

class_mapping = [
    "air_conditioner",
    "car_horn",
    "children_playing",
    "dog_bark",
    "drilling",
    "engine_idling",
    "gun_shot",
    "jackhammer",
    "siren",
    "street_music"
]

SAMPLE_RATE = 22050

FORMAT = pyaudio.paInt16
CHANNELS = 1

CHUNK = int(SAMPLE_RATE / 1)


def plot_waveform(waveform, sample_rate, title="Waveform", xlim=None, ylim=None):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].plot(time_axis, waveform[c], linewidth=1)
        axes[c].grid(True)
        if num_channels > 1:
            axes[c].set_ylabel(f'Channel {c + 1}')
        if xlim:
            axes[c].set_xlim(xlim)
        if ylim:
            axes[c].set_ylim(ylim)
    figure.suptitle(title)

    # redraw the canvas
    figure.canvas.draw()
    img = np.fromstring(figure.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    img = img.reshape(figure.canvas.get_width_height()[::-1] + (3,))

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imshow("wave", img)


def plot_specgram(waveform, sample_rate, title="Spectrogram", xlim=None):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].specgram(waveform[c], Fs=sample_rate)  # , xextent=[0,1])
        if num_channels > 1:
            axes[c].set_ylabel(f'Channel {c + 1}')
        if xlim:
            axes[c].set_xlim(xlim)
    figure.suptitle(title)

    # redraw the canvas
    figure.canvas.draw()
    img = np.fromstring(figure.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    img = img.reshape(figure.canvas.get_width_height()[::-1] + (3,))

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imshow("spectrogram", img)


def plot_mel_spectrogram(specgram, title=None, ylabel="freq_bin", ax=None):

    figure, axes = plt.subplots(1, 1)
    if title is not None:
        axes.set_title(title)
    axes.set_ylabel(ylabel)
    axes.imshow(librosa.power_to_db(specgram), origin="lower", aspect="auto", interpolation="nearest")

    figure.suptitle(title)
    figure.canvas.draw()

    img = np.fromstring(figure.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    img = img.reshape(figure.canvas.get_width_height()[::-1] + (3,))

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imshow("mel_spectrogram", img)


def int2float(sound):
    abs_max = np.abs(sound).max()
    sound = sound.astype('float32')
    if abs_max > 0:
        sound *= 1/32768
    sound = sound.squeeze()
    return sound


def predict(model, input, target, class_mapping):
    model.eval()
    with torch.no_grad():
        predictions = model(input)
        # Tensor (1, 10) -> [ [0.1, 0.01, ..., 0.6] ]
        predicted_index = predictions[0].argmax(0)
        predicted = class_mapping[predicted_index]
        expected = class_mapping[target]
    return predicted, expected


def main():
    print(torch.__version__)
    print(torchaudio.__version__)

    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=SAMPLE_RATE,
                        input=True,
                        frames_per_buffer=CHUNK)

    cnn = CNNNetwork()
    state_dict = torch.load("best_feedforwardnet.pth")
    cnn.load_state_dict(state_dict)

    usd = UrbanSoundDataset(ANNOTATIONS_FILE,
                                 AUDIO_DIR,
                                 None,
                                 SAMPLE_RATE,
                                 NUM_SAMPLES,
                                 "cpu")

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )


    data = []
    num_samples = SAMPLE_RATE

    print("Started Recording")
    while True:
        audio_chunk = stream.read(num_samples)

        data.append(audio_chunk)

        audio_int16 = np.frombuffer(audio_chunk, np.int16);

        audio_float32 = int2float(audio_int16)

        segment = torch.from_numpy(audio_float32)

        segment.unsqueeze_(0)
        segment = usd._mix_down_if_necessary(segment)
        segment = usd._cut_if_necessary(segment)
        segment = usd._right_pad_if_necessary(segment)

        plot_waveform(segment, SAMPLE_RATE, ylim=[-0.5,0.5])
        mel_signal = mel_spectrogram(segment.squeeze(0))
        plot_mel_spectrogram(mel_signal, "mel_spec")

        if cv2.waitKey(33) == ord('q'):
            break

        silent = usd._detect_silence(segment)
        if silent:
            print("waiting for sound...")
            continue

        segment.unsqueeze_(0)
        # make an inference
        predicted, expected = predict(cnn, mel_spectrogram(segment), 0, class_mapping)
        print(f"Predicted: '{predicted}'")

print("Stopped the recording")

if __name__ == "__main__":
    main()