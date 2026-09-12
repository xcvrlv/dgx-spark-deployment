#!/bin/bash
# 10 minutes of idle on the head: MemAvailable every 30 s
for i in $(seq 0 20); do echo "$(date +%H:%M:%S) $(awk '/MemAvailable/{printf "%.2f GB", $2/1024/1024}' /proc/meminfo)"; sleep 30; done
