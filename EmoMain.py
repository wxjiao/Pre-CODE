"""
Main function for emtion recognition
date: 2020/09/24
"""
import os
import argparse
import Utils
import Const
from Preprocess import Dictionary # import the object for pickle loading
import Modules
from Modules import *
from EmoTrain import emotrain, emoeval
from datetime import datetime
import time
#str(datetime.now())
#'2011-05-03 17:45:35.177000'


def main():
	'''Main function'''

	parser = argparse.ArgumentParser()

	# learning
	parser.add_argument('-lr', type=float, default=2e-4)
	parser.add_argument('-decay', type=float, default=0.75)
	parser.add_argument('-batch_size', type=int, default=16)
	parser.add_argument('-epochs', type=int, default=60)
	parser.add_argument('-patience', type=int, default=5,
	                    help='patience for early stopping')
	parser.add_argument('-save_dir', type=str, default="snapshot",
	                    help='where to save the models')
	# data
	parser.add_argument('-dataset', type=str, default='Friends',
	                    help='dataset')
	parser.add_argument('-data_path', type=str, required = True,
	                    help='data path')
	parser.add_argument('-vocab_path', type=str, required=True,
	                    help='global vocabulary path')
	parser.add_argument('-emodict_path', type=str, required=True,
	                    help='emotion label dict path')
	parser.add_argument('-tr_emodict_path', type=str, default=None,
	                    help='training set emodict path')
	parser.add_argument('-max_seq_len', type=int, default=60, # 60 for emotion
	                    help='the sequence length')
	# model
	parser.add_argument('-sentEnc', type=str, default='gru2',
	                    help='choose the low encoder')
	parser.add_argument('-contEnc', type=str, default='gru',
	                    help='choose the mid encoder')
	parser.add_argument('-dec', type=str, default='dec',
	                    help='choose the classifier')
	parser.add_argument('-d_word_vec', type=int, default=300,
	                    help='the word embeddings size')
	parser.add_argument('-d_hidden_low', type=int, default=300,
	                    help='the hidden size of rnn1')
	parser.add_argument('-d_hidden_up', type=int, default=300,
	                    help='the hidden size of rnn1')
	parser.add_argument('-layers', type=int, default=1,
	                    help='the num of stacked GRU layers')
	parser.add_argument('-d_fc', type=int, default=100,
	                    help='the size of fc')
	parser.add_argument('-gpu', type=str, default=None,
	                    help='gpu: default 0')
	parser.add_argument('-embedding', type=str, default=None,
	                    help='filename of embedding pickle')
	parser.add_argument('-report_loss', type=int, default=720,
	                    help='how many steps to report loss')
	parser.add_argument('-load_model', action='store_true',
	                    help='load the pretrained model')

	args = parser.parse_args()
	print(args, '\n')

	# load vocabs
	print("Loading vocabulary...")
	glob_vocab = Utils.loadFrPickle(args.vocab_path)
	print("Loading emotion label dict...")
	emodict = Utils.loadFrPickle(args.emodict_path)
	print("Loading review tr_emodict...")
	tr_emodict = Utils.loadFrPickle(args.tr_emodict_path)

	# load field
	print("Loading field...")
	field = Utils.loadFrPickle(args.data_path)
	test_loader = field['test']

	# word embedding
	print("Initializing word embeddings...")
	embedding = nn.Embedding(glob_vocab.n_words, args.d_word_vec, padding_idx=Const.PAD)
	if args.d_word_vec == 300:
		if args.embedding != None and os.path.isfile(args.embedding):
			np_embedding = Utils.loadFrPickle(args.embedding)
		else:
			np_embedding = Utils.load_pretrain(args.d_word_vec, glob_vocab, type='glove')
			Utils.saveToPickle("embedding.pt", np_embedding)
		embedding.weight.data.copy_(torch.from_numpy(np_embedding))
	embedding.max_norm = 1.0
	embedding.norm_type = 2.0
	embedding.weight.requires_grad = False

	# word to vec
	wordenc = Modules.wordEncoder(embedding=embedding)
	# sent to vec
	sentenc = Modules.sentEncoder(d_input=args.d_word_vec, d_output=args.d_hidden_low)
	if args.sentEnc == 'gru2':
		print("Utterance encoder: GRU2")
		sentenc = Modules.sentGRUEncoder(d_input=args.d_word_vec, d_output=args.d_hidden_low)
	if args.layers == 2:
		print("Number of stacked GRU layers: {}".format(args.layers))
		sentenc = Modules.sentGRU2LEncoder(d_input=args.d_word_vec, d_output=args.d_hidden_low)
	# cont
	contenc = Modules.contEncoder(d_input=args.d_hidden_low, d_output=args.d_hidden_up)
	# decoder
	emodec = Modules.mlpDecoder(d_input=args.d_hidden_low + args.d_hidden_up * 2, d_output=args.d_fc, n_class=emodict.n_words)

	if args.load_model:
		print('Load in pretrained model...')
		wordenc = torch.load("snapshot/wordenc_OpSub_"+str(args.d_hidden_low)+"_"+str(args.d_hidden_up)+".pt", map_location='cpu') #
		sentenc = torch.load("snapshot/sentenc_OpSub_"+str(args.d_hidden_low)+"_"+str(args.d_hidden_up)+".pt", map_location='cpu')
		contenc = torch.load("snapshot/contenc_OpSub_"+str(args.d_hidden_low)+"_"+str(args.d_hidden_up)+".pt", map_location='cpu')
		# freeze the pretrained parameters
		for p1 in wordenc.parameters():
			p1.requires_grad = False

	# Choose focused emotions
	focus_emo = Const.four_emo
	args.decay = 0.75
	if args.dataset == 'IEMOCAP4v2':
		focus_emo = Const.four_iem
		args.decay = 0.95
	if args.dataset == 'MELD':
		focus_emo = Const.sev_meld
	if args.dataset == 'EmoryNLP':
		focus_emo = Const.sev_emory
	if args.dataset == 'MOSEI':
		focus_emo = Const.six_mosei
	if args.dataset == 'MOSI':
		focus_emo = Const.two_mosi
	print("Focused emotion labels {}".format(focus_emo))

	emotrain(wordenc=wordenc,
	         sentenc=sentenc,
	         contenc=contenc,
	         dec=emodec,
	         data_loader=field,
	         tr_emodict=tr_emodict,
	         emodict=emodict,
	         args=args,
	         focus_emo=focus_emo)

	# test
	print("Load best models for testing!")

	wordenc = Utils.revmodel_loader(args.save_dir, 'wordenc', args.dataset, args.load_model)
	sentenc = Utils.revmodel_loader(args.save_dir, 'sentenc', args.dataset, args.load_model)
	contenc = Utils.revmodel_loader(args.save_dir, 'contenc', args.dataset, args.load_model)
	emodec = Utils.revmodel_loader(args.save_dir, 'dec', args.dataset, args.load_model)
	pAccs = emoeval(wordenc=wordenc,
	                sentenc=sentenc,
	                contenc=contenc,
	                dec=emodec,
	                data_loader=test_loader,
	                tr_emodict=tr_emodict,
	                emodict=emodict,
	                args=args,
	                focus_emo=focus_emo)
	print("Test: ACCs-F1s-WA-UWA-F1-val {}".format(pAccs))

	# record the test results
	record_file = '{}/{}_{}_finetune?{}.txt'.format(args.save_dir, "record", args.dataset, str(args.load_model))
	if os.path.isfile(record_file):
		f_rec = open(record_file, "a")
	else:
		f_rec = open(record_file, "w")
	f_rec.write("{} - {} - {}\t:\t{}\n".format(datetime.now(), args.d_hidden_low, args.lr, pAccs))
	f_rec.close()


if __name__ == '__main__':
	main()
