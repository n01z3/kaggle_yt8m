#!/usr/bin/env bash

#!/usr/bin/env bash

PYTHON_BIN=$HOME/anaconda3/bin/python

$PYTHON_BIN n12_pepe_rn.py --nepoh 20 --optim -1 --versn rn-1 --mtype 0 --ngpus 1  --begin 0
$PYTHON_BIN n12_pepe_rn.py --nepoh 10 --optim sgd --lrate 0.3 --versn rn-1 --mtype 0 --ngpus 1  --begin -1
$PYTHON_BIN n12_pepe_rn.py --nepoh 5 --optim sgd --lrate 0.1 --versn rn-1 --mtype 0 --ngpus 1 --begin -1
$PYTHON_BIN n12_pepe_rn.py --nepoh 5 --optim sgd --lrate 0.01 --versn rn-1 --mtype 0 --ngpus 1 --begin -1
$PYTHON_BIN n12_pepe_rn.py --nepoh 5 --optim sgd --lrate 0.1 --batch 1000 --versn rn-1 --mtype 0 --ngpus 1 --begin -1
$PYTHON_BIN n12_pepe_rn.py --nepoh 2 --optim sgd --lrate 0.01 --batch 1000 --versn rn-1 --mtype 0 --ngpus 1 --begin -1
$PYTHON_BIN n12_pepe_rn.py --nepoh 2 --optim sgd --lrate 0.001 --batch 1000 --versn rn-1 --mtype 0 --ngpus 1 --begin -1

$PYTHON_BIN n12_pepe_rn.py --nepoh 20 --optim -1 --versn rn-2 --mtype 1 --ngpus 1  --begin 0
$PYTHON_BIN n12_pepe_rn.py --nepoh 10 --optim sgd --lrate 0.3 --versn rn-2 --mtype 1 --ngpus 1  --begin -1
$PYTHON_BIN n12_pepe_rn.py --nepoh 5 --optim sgd --lrate 0.1 --versn rn-2 --mtype 1 --ngpus 1 --begin -1
$PYTHON_BIN n12_pepe_rn.py --nepoh 5 --optim sgd --lrate 0.01 --versn rn-2 --mtype 1 --ngpus 1 --begin -1
$PYTHON_BIN n12_pepe_rn.py --nepoh 5 --optim sgd --lrate 0.1 --batch 1000 --versn rn-2 --mtype 1 --ngpus 1 --begin -1
$PYTHON_BIN n12_pepe_rn.py --nepoh 2 --optim sgd --lrate 0.01 --batch 1000 --versn rn-2 --mtype 1 --ngpus 1 --begin -1
$PYTHON_BIN n12_pepe_rn.py --nepoh 2 --optim sgd --lrate 0.001 --batch 1000 --versn rn-2 --mtype 1 --ngpus 1 --begin -1

$PYTHON_BIN n12_pepe_rn.py --nepoh 20 --optim -1 --versn rn-3 --mtype 2 --ngpus 1  --begin 0
$PYTHON_BIN n12_pepe_rn.py --nepoh 10 --optim sgd --lrate 0.3 --versn rn-3 --mtype 2 --ngpus 1  --begin -1
$PYTHON_BIN n12_pepe_rn.py --nepoh 5 --optim sgd --lrate 0.1 --versn rn-3 --mtype 2 --ngpus 1 --begin -1
$PYTHON_BIN n12_pepe_rn.py --nepoh 5 --optim sgd --lrate 0.01 --versn rn-3 --mtype 2 --ngpus 1 --begin -1
$PYTHON_BIN n12_pepe_rn.py --nepoh 5 --optim sgd --lrate 0.1 --batch 1000 --versn rn-3 --mtype 2 --ngpus 1 --begin -1
$PYTHON_BIN n12_pepe_rn.py --nepoh 2 --optim sgd --lrate 0.01 --batch 1000 --versn rn-3 --mtype 2 --ngpus 1 --begin -1
$PYTHON_BIN n12_pepe_rn.py --nepoh 2 --optim sgd --lrate 0.001 --batch 1000 --versn rn-3 --mtype 2 --ngpus 1 --begin -1

$PYTHON_BIN n12_pepe_rn.py --nepoh 20 --optim -1 --versn rn-4 --mtype 3 --ngpus 1  --begin 0
$PYTHON_BIN n12_pepe_rn.py --nepoh 10 --optim sgd --lrate 0.3 --versn rn-4 --mtype 3 --ngpus 1  --begin -1
$PYTHON_BIN n12_pepe_rn.py --nepoh 5 --optim sgd --lrate 0.1 --versn rn-4 --mtype 3 --ngpus 1 --begin -1
$PYTHON_BIN n12_pepe_rn.py --nepoh 5 --optim sgd --lrate 0.01 --versn rn-4 --mtype 3 --ngpus 1 --begin -1
$PYTHON_BIN n12_pepe_rn.py --nepoh 5 --optim sgd --lrate 0.1 --batch 1000 --versn rn-4 --mtype 3 --ngpus 1 --begin -1
$PYTHON_BIN n12_pepe_rn.py --nepoh 2 --optim sgd --lrate 0.01 --batch 1000 --versn rn-4 --mtype 3 --ngpus 1 --begin -1
$PYTHON_BIN n12_pepe_rn.py --nepoh 2 --optim sgd --lrate 0.001 --batch 1000 --versn rn-4 --mtype 3 --ngpus 1 --begin -1