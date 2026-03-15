### Config guide line

This folder contains config yaml files for different types of running.
Copy content one of them to ../config.yaml
```bash
cp config_simple.yml ../config.yaml
```

### Update certificates
```bash
sudo systemctl stop nginx
sudo certbot renew
sudo systemctl start nginx
sudo certbot certificates
```
