### Update certificates
```bash
sudo systemctl stop nginx
sudo certbot renew
sudo systemctl start nginx
sudo certbot certificates
```
