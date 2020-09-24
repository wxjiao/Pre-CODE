#!bin/bash
# Var assignment
LR=2e-4
GPU=3
echo ========= lr=$LR ==============
for iter in 1
do
echo --- $Enc - $Dec $iter ---
python LMMain.py \
-lr $LR \
-gpu $GPU \
-d_hidden_low 300 \
-d_hidden_up 300 \
-sentEnc gru2 \
-layers 1 \
-patience 3 \
-data_path OpSub_data.pt \
-vocab_path glob_vocab.pt \
-embedding embedding.pt \
-dataset OpSub
done
