#! /bin/bash

# WARN='\033[1;33m'
ERR='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

CONTAINER_NAME=WLC
OSRM_CONTAINER_NAME=osrm_container

if docker ps -a | grep $CONTAINER_NAME
then
    echo "${ERR}A container with name $CONTAINER_NAME is already running!${NC}"
    docker stop ${CONTAINER_NAME} > /dev/null
    docker rm ${CONTAINER_NAME} > /dev/null
fi

destdir=$(basename "$(pwd)")

arch="$(arch)"
tag=
platform=
while test "$#" -ne 0; do
    if test "$1" = "-a" -o "$1" = "--arm" -o "$1" = "--arm64"; then
        if test "$arch" = "arm64" -o "$arch" = "aarch64"; then
            platform=linux/arm64
            shift
        else
            echo "\`run-docker --arm\` only works on ARM64 hosts" 1>&2
            exit 1
        fi
    elif test "$1" = "-x" -o "$1" = "--x86-64" -o "$1" = "--x86_64" -o "$1" = "--amd64"; then
        platform=linux/amd64
        shift
    else
        echo "Usage: run-docker" 1>&2
        exit 1
    fi
done
if test -z "$platform" -a \( "$arch" = "arm64" -o "$arch" = "aarch64" \); then
    platform=linux/arm64
elif test -z "$platform"; then
    platform=linux/amd64
fi
if test -z "$tag" -a "$platform" = linux/arm64; then
    tag=arm64
elif test -z "$tag"; then
    tag=latest
fi

# If the OSRM container is running, print out its IP address for convenience
if docker ps -a | grep osrm_container
then
    IP=$(docker inspect -f '{{range.NetworkSettings.Networks}}{{.IPAddress}}{{end}}' ${OSRM_CONTAINER_NAME})
    echo -e "${GREEN}IP Address of the OSRM container is ${IP}${NC}"
else
    echo -e "${ERR}Could not find OSRM container at ${OSRM_CONTAINER_NAME}. Is it running?${NC}"
fi

echo "Starting container with name $tag"
# docker run -it --platform $platform --rm --privileged --net=bridge --name=$CONTAINER_NAME --security-opt seccomp=unconfined -v "$(pwd):/home/user/$destdir" -w "/home/user/$destdir" $tag
docker run -it --platform $platform --rm --privileged --net falcon_backend --name=$CONTAINER_NAME --security-opt seccomp=unconfined -v /var/run/docker.sock:/var/run/docker.sock -v "$(pwd):/home/user/$destdir" -w "/home/user/$destdir" $tag
