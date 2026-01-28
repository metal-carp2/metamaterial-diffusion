#!/bin/bash
docker run -v $(pwd):/root/shared dolfinx/dolfinx:stable python3 /root/shared/$1