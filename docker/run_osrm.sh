#! /bin/bash

WARN='\033[1;33m'
ERR='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

CONTAINER_NAME=osrm_container

if [[ $(docker ps -a | grep ${CONTAINER_NAME}) ]]; then
    echo -e "${WARN}Found an old ${CONTAINER_NAME}. Removing it to make room.${NC}"
    docker stop ${CONTAINER_NAME} > /dev/null
    docker rm ${CONTAINER_NAME} > /dev/null
fi

echo -e "${GREEN}Updating docker container for OSRM${NC}"
docker pull osrm/osrm-backend

# Check to see if an OSRM file is already existent
if [[ $(find . -name '*.osrm' -type f -maxdepth 1) ]]; then
    filename=$(find . -name '*.osrm' -type f -maxdepth 1 -exec basename {} .osrm \;)
    echo -e "${GREEN}Found Pre-processed OSRM data named ${filename}${NC}"
else
    echo -e "${GREEN}No OSRM file found. Downloading northeast and pre-processing${NC}"
    echo -e "${WARN}THIS WILL TAKE many mins. If you have an OSRM file already, exit and move it here${NC}"
    filename=pennsylania
    wget -O $filename.osm.pbf "http://download.geofabrik.de/north-america/us/pennsylvania-latest.osm.pbf"

    # Pre-process the extract with the foot profile and start a routing engine HTTP server on port 5000
    echo -e "${GREEN}Pre-processing the area. This will take a while${NC}"
    docker run -t -v "${PWD}:/data" osrm/osrm-backend osrm-extract -p /opt/foot.lua /data/$filename.osm.pbf

    if [[ $(find . -name '*.osrm.ebg' -type f -maxdepth 1) ]]; then
        docker run -t -v "${PWD}:/data" osrm/osrm-backend osrm-partition /data/$filename.osrm
        docker run -t -v "${PWD}:/data" osrm/osrm-backend osrm-customize /data/$filename.osrm
    else
        echo -e "${ERR}Pre-processing failed. It could be that the file was too big so there wasn't enough RAM${NC}"
    fi
fi

# Run the final container
echo -e "${GREEN}Running the final OSRM container server${NC}"

# Map port 5001 on local to 5000 on docker (5000 is sometimes busy on local hosts)
docker run -t -i -p 5001:5000 -v "${PWD}:/data" --name ${CONTAINER_NAME} osrm/osrm-backend osrm-routed --algorithm mld /data/$filename.osrm