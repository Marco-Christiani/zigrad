#!/bin/bash
DATA_DIR=$(dirname "$(echo "$2" | cut -d' ' -f1)")
export DATA_DIR
exec "$1"
