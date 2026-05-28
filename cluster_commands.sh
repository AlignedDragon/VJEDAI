#!/usr/bin/env bash

CLUSTER_USER="ragerber"
CLUSTER_HOST="student-cluster.inf.ethz.ch"
CLUSTER_DIR="/home/ragerber"

copytocluster() {
	scp "$1" "$CLUSTER_USER@$CLUSTER_HOST:$CLUSTER_DIR/"
}

copyfromcluster() {
	scp "$CLUSTER_USER@$CLUSTER_HOST:$CLUSTER_DIR/$1" .
}

sshintocluster() {
	ssh "$CLUSTER_USER@$CLUSTER_HOST"
}

getjobid() {
	ssh "$CLUSTER_USER@$CLUSTER_HOST" "squeue | grep '$CLUSTER_USER'"
}

trackjob() {
	ssh "$CLUSTER_USER@$CLUSTER_HOST" "tail -f '$CLUSTER_DIR/jdepth_train-$1.out'"
}

jobcpu() {
	ssh "$CLUSTER_USER@$CLUSTER_HOST" "srun --jobid='$1' --pty nvidia-smi"
}
startjob() {
	ssh "$CLUSTER_USER@$CLUSTER_HOST" "cd '$CLUSTER_DIR' && sbatch train.sbatch"
}

geterrors() {
	ssh "$CLUSTER_USER@$CLUSTER_HOST" "cat '$CLUSTER_DIR/logs/da3metric_decoder-$1.err'"
}

getloss() {
	ssh "$CLUSTER_USER@$CLUSTER_HOST" "grep '^step ' '$CLUSTER_DIR/logs/da3metric_decoder-$1.out'" > ~/Documents/FS26/CIL/Project/tests/loss.txt
}
