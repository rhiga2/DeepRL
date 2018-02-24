#!/bin/bash
set -eux
for e in RoboschoolHopper-v1 RoboschoolAnt-v1 RoboschoolHalfCheetah-v1 RoboschoolHumanoid-v1 RoboschoolReacher-v1 RoboschoolWalker2d-v1
do
    python run_expert.py experts/$e.pkl $e --render --num_rollouts=1
done
