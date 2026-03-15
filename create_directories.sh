#!/bin/bash

sudo rm -r logs
sudo rm -r data
mkdir data logs
cd logs
mkdir server1 server2
cd ../data
mkdir minio prometheus grafana dragonfly_master dragonfly_replica1 dragonfly_replica2
