# sound_recognition
Sound recognition example trained on UbanSounds8K using CNNs to classify Mel spectrograms.

You can train your CNN using `train.py` after setting up your `ANNOTATIONS_FILE` and `AUDIO_DIR` variables.

You can use the `input_streaming.py` to capture a audio stream from you input device and perform classification. An audio signal and Mel spectrograms will be shown along with the inference category for the input sound:



inspired by https://www.youtube.com/@ValerioVelardoTheSoundofAI
