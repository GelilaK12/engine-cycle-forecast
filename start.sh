#!/bin/bash
cd ~/engine-cycle-forecast
docker start -ai rul-container || docker run -it --name rul-container rul-project

