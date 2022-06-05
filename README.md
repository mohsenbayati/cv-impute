# cv-impute
This repository contains codes for replicating simulations of the following paper:

N. Hamidi and M. Bayati, On Low-rank Trace Regression under General Sampling Distribution, https://arxiv.org/abs/1904.08576.

To recreate all simulations, use the following commands:

**Synthetic data simulations**

50x50 matrix of rank 2:
-----------------------

python -m experiments --run 100 --n-fold 10 --sd 0.5 --mnr 50 50 2 --step 100 --max-size 1000 --name run-100-n-fold-10-sd-p5-mnr-50-50-2-step-100-max-size-1000

python -m experiments --run 100 --n-fold 10 --sd 1. --mnr 50 50 2 --step 100 --max-size 1000 --name run-100-n-fold-10-sd-1-mnr-50-50-2-step-100-max-size-1000

python -m experiments --run 100 --n-fold 10 --sd 2. --mnr 50 50 2 --step 100 --max-size 1000 --name run-100-n-fold-10-sd-2-mnr-50-50-2-step-100-max-size-1000


100x100 matrix of rank 3
------------------------

python -m experiments --run 100 --n-fold 10 --sd 0.5 --mnr 100 100 3 --step 100 --max-size 2500 --name run-100-n-fold-10-sd-p5-mnr-100-100-3-step-100-max-size-2500

python -m experiments --run 100 --n-fold 10 --sd 1. --mnr 100 100 3 --step 100 --max-size 2500 --name run-100-n-fold-10-sd-1-mnr-100-100-3-step-100-max-size-2500

python -m experiments --run 100 --n-fold 10 --sd 2. --mnr 100 100 3 --step 100 --max-size 2500 --name run-100-n-fold-10-sd-2-mnr-100-100-3-step-100-max-size-2500


**Real data simulations**

50x50 matrix
------------

python -m experiments --run 100 --sd 0.5 --n-fold 10 --mnr 50 50 2 --step 100 --max-size 1000 --real-data 1 --name run-real-100-n-sd-p5-fold-10-50-step-100-max-size-1000-centered

python -m experiments --run 100 --sd 1 --n-fold 10 --mnr 50 50 2 --step 100 --max-size 1000 --real-data 1 --name run-real-100-n-sd-1-fold-10-50-step-100-max-size-1000-centered
(seed 10 maybe necessary)

python -m experiments --run 100 --sd 2 --n-fold 10 --mnr 50 50 2 --step 100 --max-size 1000 --real-data 1 --name run-real-100-n-sd-2-fold-10-50-step-100-max-size-1000-centered

100x100 matrix
------------

python -m experiments --run 100 --sd 0.5 --n-fold 10 --mnr 100 100 3 --step 100 --max-size 2500 --real-data 1 --name run-real-100-n-sd-p5-fold-10-100-step-100-max-size-2500-centered

python -m experiments --run 100 --sd 1 --n-fold 10 --mnr 100 100 3 --step 100 --max-size 2500 --real-data 1 --name run-real-100-n-sd-1-fold-10-100-step-100-max-size-2500-centered

python -m experiments --run 100 --sd 2 --n-fold 10 --mnr 100 100 3 --step 100 --max-size 2500 --real-data 1 --name run-real-100-n-sd-2-fold-10-100-step-100-max-size-2500-centered

