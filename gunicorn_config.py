# gunicorn_config.py
bind = "unix:/var/run/emotionai/gunicorn.sock"
workers = 4
worker_class = "gevent"
timeout = 120
keepalive = 5
accesslog = "-"
errorlog = "-"
loglevel = "info"
