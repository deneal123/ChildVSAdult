#!/bin/sh
set -eu

mkdir -p /app/models
chown prom:prom /app/models
exec su -s /bin/sh prom -c 'exec prom-model-init'
