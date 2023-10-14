#! /bin/bash

# Just in case: kill -9 $(lsof -t -i tcp:6379)

docker run -t -i -p "127.0.0.1:6379:6379" --net falcon_backend -v "${PWD}/redis-data/:/data" --name redis-container -d redis redis-server --requirepass votefalcon12 --loglevel warning