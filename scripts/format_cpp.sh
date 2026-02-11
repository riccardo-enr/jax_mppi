#!/bin/bash
find third_party/cuda-mppi \
  -regex '.*\.(cpp|hpp|c|h|cu|cuh)$' \
  -not -path '*/build/*' \
  -exec uncrustify \
    -c /opt/ros/jazzy/lib/python3.12/site-packages/ament_uncrustify/configuration/ament_code_style_0_78.cfg \
    -l CPP \
    --replace \
    --no-backup {} \;
