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
1. How to run in dev python
```bash
gunicorn --bind 0.0.0.0:8000 wsgi:app
```
2. Install redis server
```bash
sudo apt update
sudo apt install redis-server
sudo vim /etc/redis/redis.conf
```
3. Build c++ project
```bash
# build
git submodule update --init --recursive
bash install_deps.sh
patch -p1 < emotiefflib.patch
mkdir build && cd build
cmake .. -G Ninja -DBUILD_TESTS=ON
cmake --build .
# run
./EmotionAI
# run tests
./tests/EmotionAI_UnitTests
./tests/EmotionAI_IntegrationTests
```
4. For production
```bash
sudo vim /etc/nginx/sites-available/your-cpp-service
sudo vim /etc/systemd/system/your-cpp-service.service

# Create symlink to enable your service using command
sudo systemctl enable your-cpp-service.service 
# Create symlink from sites-available to sites-enabled
sudo ln -s /etc/nginx/sites-available/your-cpp-service /etc/nginx/sites-enabled/

sudo systemctl daemon-reload
sudo systemctl restart your-cpp-service
sudo systemctl reload nginx
```