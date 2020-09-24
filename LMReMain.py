"""
Main function for loading the pre-trained model to adjust the embedding layer.
Date: 2020/09/24
"""
import numpy as np
import os
import argparse
import torch
import torch.nn as nn
import Utils
import Const
from Preprocess import Dictionary # import the object for pickle loading
import Modules
from LMReTrain import lmtrain, lmeval
from datetime import datetime



def main():
	'''Main function'''

	parser = argparse.ArgumentParser()

	# learning
	parser.add_argument('-lr', type=float, default=2e-4)
	parser.add_argument('-batch_size', type=int, default=64)
	parser.add_argument('-epochs', type=int, default=10)
	parser.add_argument('-patience', type=int, default=5,
	                    help='patience for early stopping')
	parser.add_argument('-save_dir', type=str, default="snapshot",
	                    help='where to save the models')
	# data
	parser.add_argument('-dataset', type=str, default='OpSub',
	                    help='dataset')
	parser.add_argument('-data_path', type=str, required = True,
	                    help='data path')
	parser.add_argument('-vocab_path', type=str, required=True,
	                    help='global vocabulary path')
	parser.add_argument('-max_seq_len', type=int, default=60,
	                    help='the sequence length')
	# model
	parser.add_argument('-sentEnc', type=str, default='gru2',
	                    help='choose the low encoder')
	parser.add_argument('-contEnc', type=str, default='gru',
	                    help='choose the mid encoder')
	parser.add_argument('-dec', type=str, default='revdec',
	                    help='choose the classifier')
	parser.add_argument('-d_word_vec', type=int, default=300,
	                    help='the word embeddings size')
	parser.add_argument('-d_hidden_low', type=int, default=300,
	                    help='the hidden size of rnn')
	parser.add_argument('-d_hidden_up', type=int, default=300,
	                    help='the hidden size of rnn')
	parser.add_argument('-layers', type=int, default=1,
	                    help='the num of stacked RNN layers')
	parser.add_argument('-fix_word_emb', action='store_true',
	                    help='fix the word embeddings')
	parser.add_argument('-gpu', type=str, default=None,
	                    help='gpu: default 0')
	parser.add_argument('-embedding', type=str, default=None,
	                    help='filename of embedding pickle')
	parser.add_argument('-report_loss', type=int, default=5000,
	                    help='how many steps to report loss')

	args = parser.parse_args()
	print(args, '\n')

	# load vocabs
	print("Loading vocabulary...")
	glob_vocab = Utils.loadFrPickle(args.vocab_path)
	# load field
	print("Loading field...")
	field = Utils.loadFrPickle(args.data_path)
	test_loader = field['test']

	print('Load in pretrained model...')
	wordenc = torch.load("snapshot/wordenc_OpSub_" + str(args.d_hidden_low) + "_" + str(args.d_hidden_up) + ".pt", map_location='cpu')  #
	sentenc = torch.load("snapshot/sentenc_OpSub_" + str(args.d_hidden_low) + "_" + str(args.d_hidden_up) + ".pt", map_location='cpu')
	contenc = torch.load("snapshot/contenc_OpSub_" + str(args.d_hidden_low) + "_" + str(args.d_hidden_up) + ".pt", map_location='cpu')
	cmdec = torch.load("snapshot/dec_OpSub_" + str(args.d_hidden_low) + "_" + str(args.d_hidden_up) + ".pt", map_location='cpu')
	# unfreeze the parameters in the embedding layer
	wordenc.embedding.weight.requires_grad = True
	for p1 in wordenc.parameters():
		p1.requires_grad = True

	# loss
	criterion = nn.BCEWithLogitsLoss()

	# train
	lmtrain(wordenc=wordenc, sentenc=sentenc, contenc=contenc, dec=cmdec, criterion=criterion, data_loader=field, args=args)

	# test
	print("Load best models for testing!")

	wordenc = torch.load("snapshot/wordenc_w_"+args.dataset+".pt")
	sentenc = torch.load("snapshot/sentenc_w_"+args.dataset+".pt")
	contenc = torch.load("snapshot/contenc_w_"+args.dataset+".pt")
	cmdec = torch.load("snapshot/dec_w_"+args.dataset+".pt")

	topkns = lmeval(wordenc, sentenc, contenc, cmdec, test_loader, args)
	print("Test: R1@5 R2@5 R1@11 R2@11 {}".format(topkns))

	# record the test results
	record_file = "snapshot/" + "record_" + args.dataset + "_" + args.sentEnc + ".txt"
	if os.path.isfile(record_file):
		f_rec = open(record_file, "a")
	else:
		f_rec = open(record_file, "w")
	f_rec.write(str(datetime.now()) + "\t:\t" + str(topkns) + "\n")
	f_rec.close()

if __name__ == '__main__':
	main()
