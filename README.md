# EmotionAI

### Project description
Face emotion recognition on images/videos/audios.

### Models description
This multimodal emotion detection model predicts a speaker's emotion using audio and image sequences from videos.
The repository contains two primary models: an audio tone recognition model with a CNN for audio-based emotion prediction, and a facial emotion recognition model using a CNN and optional mediapipe face landmarks for facial emotion prediction.
The third model combines a video clip's audio and image sequences, processed through an LSTM for speaker emotion prediction.
Hyperparameters such as landmark usage, CNN model selection, LSTM units, and dense layers are tuned for optimal accuracy using included modules.
For new datasets, follow the instructions below to retune the hyperparameters.

### How to use project
1. make install
2. make build
3. make models
4. Set up config.yaml, copy to build/config.yaml
5. Change field `requirepass` in /etc/redis/redis.conf
6. make run

### Additional info
1. `gunicorn --bind 0.0.0.0:8000 wsgi:app` - run python version in dev
2. /etc/redis/redis.conf - redis configuration file
3. For production
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
4. Dragonfly
```bash
# Run in foreground
sudo docker run -p 6379:6379 --name dragonfly docker.dragonflydb.io/dragonflydb/dragonfly
# Run in background
sudo docker run -d -p 6379:6379 --name dragonfly docker.dragonflydb.io/dragonflydb/dragonfly
sudo docker start dragonfly
```

### Contributing:
1. Create issue: https://github.com/Shurik12/EmotionAI/issues
