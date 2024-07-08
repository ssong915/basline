pkill python3
pkill nvidia-smi
kill -9 `ps -ef|grep tgn|awk '{print $2}'`