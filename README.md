# Sound recognition
Sound recognition example. Trained on UbanSounds8K using CNNs to classify Mel spectrograms.

You can train your CNN using `train.py` after setting up your `ANNOTATIONS_FILE` and `AUDIO_DIR` variables. You can also use the pretrained checkpoint `best_feedforwardnet.pth` to classify the UrbanSounds8K categories 
\["air_conditioner", "car_horn", "children_playing", "dog_bark", "drilling", "engine_idling", "gun_shot", "jackhammer", "siren", "street_music"].

The `input_streaming.py` script captures an audio stream from your input device and performs classification. An audio signal and Mel spectrograms will be shown along with the inference category for the input sound:

![Input signal.](https://github.com/cjvargasc/sound_recognition/blob/main/imgs/signal.png)

![Mel spectrogram.](https://github.com/cjvargasc/sound_recognition/blob/main/imgs/spectrogram.png)

![Category inference.](https://github.com/cjvargasc/sound_recognition/blob/main/imgs/output.png)

Inspired by https://www.youtube.com/@ValerioVelardoTheSoundofAI. :+1:
