#!bin/bash
# Var assignment
LR=1e-4
GPU=1
du=300
dc=300
echo ========= lr=$LR ==============
for iter in 1 2 3 4 5
do
echo --- $Enc - $Dec $iter ---
python EmoMain.py -load_model -lr $LR -gpu $GPU -d_hidden_low $du -d_hidden_up $dc -patience 6 -report_loss 720 -data_path Friends_data.pt -vocab_path glob_vocab.pt -emodict_path Friends_emodict.pt -tr_emodict_path Friends_tr_emodict.pt -dataset Friends -embedding embedding.pt
#python EmoMain.py -load_model -lr $LR -gpu $GPU -d_hidden_low $du -d_hidden_up $dc -patience 6 -report_loss 720 -data_path Emotionpush_data.pt -vocab_path glob_vocab.pt -emodict_path Emotionpush_emodict.pt -tr_emodict_path Emotionpush_tr_emodict.pt -dataset Emotionpush -embedding embedding.pt
#python EmoMain.py -load_model -lr $LR -gpu $GPU -d_hidden_low $du -d_hidden_up $dc -patience 10 -report_loss 96 -data_path IEMOCAP4v2_data.pt -vocab_path glob_vocab.pt -emodict_path IEMOCAP4v2_emodict.pt -tr_emodict_path IEMOCAP4v2_tr_emodict.pt -dataset IEMOCAP4v2 -embedding embedding.pt
#python EmoMain.py -load_model -lr $LR -gpu $GPU -d_hidden_low $du -d_hidden_up $dc -patience 6 -report_loss 713 -data_path EmoryNLP_data.pt -vocab_path glob_vocab.pt -emodict_path EmoryNLP_emodict.pt -tr_emodict_path EmoryNLP_tr_emodict.pt -dataset EmoryNLP -embedding embedding.pt
#python EmoMain.py -load_model -lr $LR -gpu $GPU -d_hidden_low $du -d_hidden_up $dc -patience 6 -report_loss 2250 -data_path MOSEI_data.pt -vocab_path glob_vocab.pt -emodict_path MOSEI_emodict.pt -tr_emodict_path MOSEI_tr_emodict.pt -dataset MOSEI -embedding embedding.pt
done
