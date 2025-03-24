#Go to http://xxx.xxx.xxx.xxx:6007/ ()

# sudo ufw allow 6007/tcp
tensorboard --logdir lightning_logs --host 0.0.0.0 --port 6007
# sudo netstat -tulnp | grep -E "6006|6007"