# EmotionAI

### Project description
Face emotion recognition on images/videos/audios.

### Models description
This multimodal emotion detection model predicts a speaker's emotion using audio and image sequences from videos.
The repository contains two primary models: an audio tone recognition model with a CNN for audio-based emotion prediction, and a facial emotion recognition model using a CNN and optional mediapipe face landmarks for facial emotion prediction.
The third model combines a video clip's audio and image sequences, processed through an LSTM for speaker emotion prediction.
Hyperparameters such as landmark usage, CNN model selection, LSTM units, and dense layers are tuned for optimal accuracy using included modules.
For new datasets, follow the instructions below to retune the hyperparameters.

### Contributing:
1. Create issue: https://github.com/Shurik12/EmotionAI/issues

### Additional info
1. How to run in dev
```bash
gunicorn --bind 0.0.0.0:8000 wsgi:app
```
2. How to run in production
```bash
sudo systemctl restart emotion_detection.service
```
3. Production settings
```bash
# Nginx configuration
/etc/nginx/sites-available/emotion_detection
# Service configuration
/etc/systemd/system/emotion_detection.service
```
4. Install redis server
```bash
sudo apt update
sudo apt install redis-server
sudo vim /etc/redis/redis.conf
```
5. Build c++ project
```bash
git submodule update --init --recursive
bash install_deps.sh
mkdir build && cd build
cmake .. -G Ninja
cmake --build .
```
6. For production
```bash
sudo vim /etc/nginx/sites-available/your-cpp-service
sudo vim /etc/systemd/system/your-cpp-service.service
# Create symlink to enable your service using command
sudo systemctl enable your-cpp-service.service 
# Create symlink from sites-available to sites-enabled
sudo ln -s /etc/nginx/sites-available/your-cpp-service /etc/nginx/sites-enabled/
```