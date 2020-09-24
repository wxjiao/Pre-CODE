"""
Train on OpSub dataset.
"""
import os
import time
import numpy as np
import random
import torch
import torch.optim as optim
from torch.autograd import Variable
import Utils
from datetime import datetime

def lmtrain(wordenc, sentenc, contenc, dec, criterion, data_loader, args):
	"""
	:data_loader input the whole field
	"""
	# start time
	time_st = time.time()
	decay_rate = 0.75

	# dataloaders
	train_loader = data_loader['train']
	dev_loader = data_loader['dev']
	scripts, negs, labels = train_loader['script'], train_loader['neg'], train_loader['label']

	lr = args.lr
	wordenc_opt = optim.Adam(wordenc.parameters(), lr=lr)
	sentenc_opt = optim.Adam(sentenc.parameters(), lr=lr)
	contenc_opt = optim.Adam(contenc.parameters(), lr=lr)
	dec_opt = optim.Adam(dec.parameters(), lr=lr)

	wordenc.train()
	sentenc.train()
	contenc.train()
	dec.train()

	over_fitting = 0
	cur_best = 0
	glob_steps = 0
	report_loss = 0
	loss_minbatch = 0
	for epoch in range(1, args.epochs + 1):
		scripts, negs, labels = Utils.shuffle_lists(scripts, negs, labels)
		print("===========Epoch==============")
		print("-{}-{}".format(epoch, datetime.now()))
		for bz in range(len(labels)):
			# tensorize a dialog list
			script, lens = Utils.ToTensor(scripts[bz], is_len=True)
			# negative sampling
			neg_sampled, label_sampled = neg_sample(scripts, bz, num_neg=10)
			neg, lenn = Utils.ToTensor(neg_sampled, is_len=True)
			label = Utils.ToTensor(label_sampled)
			script = Variable(script)
			neg = Variable(neg)
			label = Variable(label).float()

			if args.gpu != None:
				os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
				device = torch.device("cuda: 0")
				wordenc.cuda(device)
				sentenc.cuda(device)
				contenc.cuda(device)
				dec.cuda(device)
				criterion.cuda(device)
				script = script.cuda(device)
				neg = neg.cuda(device)
				label = label.cuda(device)

			word_scr = wordenc(script)
			sent_scr = sentenc(word_scr, lens)[0]
			word_neg = wordenc(neg)
			sent_neg = sentenc(word_neg, lenn)[0]
			cont_scr = contenc(sent_scr)
			prob = dec(sent_scr, cont_scr, sent_neg)
			#print(log_prob, label)
			loss = criterion(prob.view(-1), label.view(-1))
			loss.backward()

			report_loss += loss.item()

			loss_minbatch += loss.item()
			glob_steps += 1

			# gradient clip
			torch.nn.utils.clip_grad_norm_(wordenc.parameters(), max_norm=5)
			torch.nn.utils.clip_grad_norm_(sentenc.parameters(), max_norm=5)
			torch.nn.utils.clip_grad_norm_(contenc.parameters(), max_norm=5)
			torch.nn.utils.clip_grad_norm_(dec.parameters(), max_norm=5)

			wordenc_opt.step()
			sentenc_opt.step()
			contenc_opt.step()
			dec_opt.step()
			wordenc_opt.zero_grad()
			sentenc_opt.zero_grad()
			contenc_opt.zero_grad()
			dec_opt.zero_grad()

			if glob_steps % args.report_loss == 0:
				print("{} Steps: {} Loss: {} LR: {}".format(datetime.now(), glob_steps, report_loss/args.report_loss, sentenc_opt.param_groups[0]['lr']))
				report_loss = 0

		# validate
		topkns = lmeval(wordenc, sentenc, contenc, dec, dev_loader, args)
		print("Time {} Validate: R1@5 R2@5 R1@11 R2@11 {}".format(Utils.timeSince(time_st), topkns))

		last_best = topkns[2]
		if last_best > cur_best:
			Utils.scrmodel_saver(wordenc, args.save_dir, 'wordenc_w', args.dataset)
			Utils.scrmodel_saver(sentenc, args.save_dir, 'sentenc_w', args.dataset)
			Utils.scrmodel_saver(contenc, args.save_dir, 'contenc_w', args.dataset)
			Utils.scrmodel_saver(dec, args.save_dir, 'dec_w', args.dataset)
			cur_best = last_best
			over_fitting = 0
		else:
			over_fitting += 1
			wordenc_opt.param_groups[0]['lr'] *= decay_rate
			sentenc_opt.param_groups[0]['lr'] *= decay_rate
			contenc_opt.param_groups[0]['lr'] *= decay_rate
			dec_opt.param_groups[0]['lr'] *= decay_rate

		if over_fitting >= args.patience:
			break


def neg_sample(scripts, scr_idx, num_neg=10):
	set_len = len(scripts)
	conv_len = len(scripts[scr_idx])
	to_be_avoid = []
	to_be_avoid.append(scr_idx)

	# produce negative samples
	neg = []
	for j in range(num_neg):
		rd1 = random.randrange(0, set_len)
		while rd1 in to_be_avoid:
			rd1 = random.randrange(0, set_len)
		to_be_avoid.append(rd1)
		scr_samp = scripts[rd1]
		num_utt = len(scr_samp)
		rd2 = random.randrange(0, num_utt)
		neg.append(scr_samp[rd2])

	# produce label
	la_idxs = [1] + [0] * num_neg
	laidxs = [la_idxs] * (conv_len - 2)
	return neg, laidxs


def topkn(matrix, k, n, true_idx=0):
	"""
	:param matrix: batch x N, k <= n
	:return:
	"""
	batch = matrix.size()[0]
	topk = matrix[:, :n].topk(k, dim=1)
	topk_sum = torch.sum(topk[1].eq(true_idx))

	return topk_sum, batch


def lmeval(wordenc, sentenc, contenc, dec, data_loader, args):
	""" data_loader only input 'dev' """
	wordenc.eval()
	sentenc.eval()
	contenc.eval()
	dec.eval()

	scripts, negs, labels = data_loader['script'], data_loader['neg'], data_loader['label']

	top15_all = 0
	top25_all = 0
	top111_all = 0
	top211_all = 0
	batch_all = 0
	for bz in range(len(labels)):
		# tensorize a dialog list
		script, lens = Utils.ToTensor(scripts[bz], is_len=True)
		neg, lenn = Utils.ToTensor(negs[bz], is_len=True)
		label = Utils.ToTensor(labels[bz])
		script = Variable(script)
		neg = Variable(neg)
		label = Variable(label)

		if args.gpu != None:
			os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
			device = torch.device("cuda: 0")
			wordenc.cuda(device)
			sentenc.cuda(device)
			contenc.cuda(device)
			dec.cuda(device)
			script = script.cuda(device)
			neg = neg.cuda(device)
			label = label.cuda(device)

		word_scr = wordenc(script)
		sent_scr = sentenc(word_scr, lens)[0]
		word_neg = wordenc(neg)
		sent_neg = sentenc(word_neg, lenn)[0]
		cont_scr = contenc(sent_scr)
		prob0 = dec(sent_scr, cont_scr, sent_neg)
		# L-2 x (1+N)
		# n, k < n
		prob = torch.sigmoid(prob0)
		top15, batch = topkn(prob, 1, 5)
		top25 = topkn(prob, 2, 5)[0]
		top111 = topkn(prob, 1, 11)[0]
		top211 = topkn(prob, 2, 11)[0]
		top15_all += top15.item()
		top25_all += top25.item()
		top111_all += top111.item()
		top211_all += top211.item()
		batch_all += batch

	topkns = [round(float(top15_all)/batch_all, 4),
	          round(float(top25_all)/batch_all, 4),
	          round(float(top111_all)/batch_all, 4),
	          round(float(top211_all)/batch_all, 4)]

	wordenc.train()
	sentenc.train()
	contenc.train()
	dec.train()

	return topkns
