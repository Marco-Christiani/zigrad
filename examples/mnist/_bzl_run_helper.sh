#!/bin/bash
ZG_DATA_DIR=$(dirname "$(echo "$2" | cut -d' ' -f1)")
export ZG_DATA_DIR
exec "$1"
