#! /bin/bash

docker run -t -i -p "127.0.0.1:6379:6379" --net edna -v "${PWD}/redis-data/:/data" --name redis-container -d redis redis-server --requirepass edna12 --loglevel warning