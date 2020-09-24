"""
Build dicts and turn sentences into indices;
Worddict is built and shared between Emotion dataset and OpSub dataset.
date: 2020/09/24
"""

import time
import re
import json
from tqdm import tqdm
import unicodedata
import argparse
from io import open
import Const
from Utils import saveToPickle, loadFrPickle,timeSince


class Dictionary:
	def __init__(self, name):
		self.name = name
		self.pre_word2count = {}
		self.rare = []
		self.word2count = {}
		self.word2index = {}
		self.index2word = {}
		self.n_words = 0
		self.max_length = 0
		self.max_dialog = 0


	def delRare(self, min_count, padunk=True):

		# collect rare words
		for w,c in self.pre_word2count.items():
			if c < min_count:
				self.rare.append(w)

		# add pad and unk
		if padunk:
			self.word2index[Const.PAD_WORD] = Const.PAD
			self.index2word[Const.PAD] = Const.PAD_WORD
			self.word2count[Const.PAD_WORD] = 1
			self.word2index[Const.UNK_WORD] = Const.UNK
			self.index2word[Const.UNK] = Const.UNK_WORD
			self.word2count[Const.UNK_WORD] = 1
			self.n_words += 2

		# index words
		for w,c in self.pre_word2count.items():
			if w not in self.rare:
				self.word2count[w] = c
				self.word2index[w] = self.n_words
				self.index2word[self.n_words] = w
				self.n_words += 1

	def addSentence(self, sentence):
		sentsplit = sentence.split(' ')
		if len(sentsplit) > self.max_length:
			self.max_length = len(sentsplit)
		for word in sentsplit:
			self.addWord(word)

	def addWord(self, word):
		if word not in self.pre_word2count:
			self.pre_word2count[word] = 1
		else:
			self.pre_word2count[word] += 1


# Preprocess of words
def unicodeToAscii(str):
	return ''.join(
		c for c in unicodedata.normalize('NFD', str)
		if unicodedata.category(c) != 'Mn'
	)

def normalizeString(str):
	str = unicodeToAscii(str.lower().strip())
	str = re.sub(r"([!?])", r" \1", str)
	str = re.sub(r"[^a-zA-Z!?]+", r" ", str)
	return str


def readUtterance(filename):
	with open(filename, encoding='utf-8') as data_file:
		data = json.loads(data_file.read())

	diadata = [[normalizeString(utter['utterance']) for utter in dialog] for dialog in data]

	emodata = [[utter['emotion'] for utter in dialog] for dialog in data]
	return diadata, emodata


def buildEmodict(dirt, phaselist, diadict, emodict):
	""" build dicts for words and emotions """
	print("Building dicts for emotion dataset...")

	max_dialog = 0
	for phase in phaselist:
		filename = dirt + phase + '.json'
		diadata, emodata  = readUtterance(filename)
		for dia, emo in zip(diadata, emodata):
			if len(dia) > max_dialog: max_dialog = len(dia)
			for d, e in zip(dia, emo):
				diadict.addSentence(d)
				emodict.addSentence(e)

	diadict.max_dialog = max_dialog

	return diadict, emodict


def indexEmo(dirt, phase, diadict, emodict, max_seq_len=60):

	filename = dirt + phase + '.json'

	diadata, emodata = readUtterance(filename)
	print('Processing file {}, length {}...'.format(filename, len(diadata)))
	diaidxs = []
	emoidxs = []
	for dia, emo in zip(diadata, emodata):
		dia_idxs = []
		emo_idxs = []
		for d, e in zip(dia, emo):
			#d_idxs = [diadict.word2index[w] if w not in diadict.rare else Const.UNK for w in d.split(' ')]
			d_idxs = [diadict.word2index[w] if w in diadict.word2index else Const.UNK for w in d.split(' ')]  # MELD and EmoryNLP not used for building vocab
			e_idxs = [emodict.word2index[e]]
			if len(d_idxs) > max_seq_len:
				dia_idxs.append(d_idxs[:max_seq_len])
			else:
				dia_idxs.append(d_idxs + [Const.PAD] * (max_seq_len - len(d_idxs)))
			emo_idxs.append(e_idxs)
		diaidxs.append(dia_idxs)
		emoidxs.append(emo_idxs)

	diafield = dict()
	diafield['feat'] = diaidxs
	diafield['label'] = emoidxs

	return diafield


def readScr(filename):
	with open(filename, 'r', encoding='utf-8') as f:
		data = json.loads(f.read())

	script = [[normalizeString(si) for si in scr['script']] for scr in data]
	neg = [[normalizeString(si) for si in scr['neg']] for scr in data]

	return script, neg


def buildScrdict(dirt, phaselist, scrwodict):
	""" add words in Script into diadict """
	print("Building dicts for OpSub dataset...")

	for phase in phaselist:
		filename = dirt + phase + '.json'
		scripts, negs = readScr(filename)
		for scr,ne in zip(scripts, negs):
			for si in scr:
				scrwodict.addSentence(si)
				# no need to count words in negs since they come from some other scripts

	return scrwodict


def indexScr(dirt, phase, scrwodict, max_seq_len):
	time_st = time.time()

	filename = dirt + phase + '.json'

	scripts, negs = readScr(filename)
	print('Processing file {}, length {}...'.format(filename, len(scripts)))
	scriptidxs = []
	negidxs = []
	labelidxs = []
	count = 0
	for scr, ne in zip(scripts, negs):
		count += 1
		if count % 1000 == 0:
			print('-->{} dialogs {}'.format(timeSince(time_st), count))

		scridxs = []
		for si in scr:
			si_idxs = [scrwodict.word2index[w] if w not in scrwodict.rare else Const.UNK for w in si.split(' ')]
			if len(si_idxs) > max_seq_len:
				scridxs.append(si_idxs[:max_seq_len])
			else:
				scridxs.append(si_idxs + [Const.PAD] * (max_seq_len - len(si_idxs)))

		scriptidxs.append(scridxs)

		neidxs = []
		for ni in ne:
			ni_idxs = [scrwodict.word2index[w] if w not in scrwodict.rare else Const.UNK for w in ni.split(' ')]
			if len(ni_idxs) > max_seq_len:
				neidxs.append(ni_idxs[:max_seq_len])
			else:
				neidxs.append(ni_idxs + [Const.PAD] * (max_seq_len - len(ni_idxs)))

		negidxs.append(neidxs)

		la_idxs = [1] + [0] * len(neidxs)
		laidxs = [la_idxs] * (len(scridxs) - 2)
		labelidxs.append(laidxs)

	scrfield = dict()
	scrfield['script'] = scriptidxs
	scrfield['neg'] = negidxs
	scrfield['label'] = labelidxs

	return scrfield


def global_vocab(phaselist, min_count, max_seq_len):
	""" Build the global vocabulary for all datasets. """
	fpath = 'Data/Friends/Friends_'
	epath = 'Data/Emotionpush/Emotionpush_'
	ipath = 'Data/IEMOCAP4v2/IEMOCAP4v2_'
	mpath = 'Data/MOSI/MOSI_'
	spath = 'Data/OpSub/OpSub_'
	vocab = Dictionary('globe')
	femo = Dictionary('femo')
	eemo = Dictionary('eemo')
	iemo = Dictionary('iemo')
	memo = Dictionary('memo')

	vocab, femo = buildEmodict(dirt=fpath, phaselist=phaselist, diadict=vocab, emodict=femo)
	vocab, eemo = buildEmodict(dirt=epath, phaselist=phaselist, diadict=vocab, emodict=eemo)
	vocab, iemo = buildEmodict(dirt=ipath, phaselist=phaselist, diadict=vocab, emodict=iemo)
	vocab, memo = buildEmodict(dirt=mpath, phaselist=phaselist, diadict=vocab, emodict=memo)
	vocab = buildScrdict(dirt=spath, phaselist=phaselist, scrwodict=vocab)

	vocab.delRare(min_count=min_count, padunk=True)
	femo.delRare(min_count=0, padunk=False)
	eemo.delRare(min_count=0, padunk=False)
	iemo.delRare(min_count=0, padunk=False)
	memo.delRare(min_count=0, padunk=False)

	saveToPickle('glob_vocab.pt', vocab)
	print('Glabal vocabulary (min_count={}): majority words {} rare words {}\n'.format(
		min_count, vocab.n_words, len(vocab.rare)))

	Scrfield = dict()
	for phase in phaselist:
		scrdata = indexScr(dirt=spath, phase=phase, scrwodict=vocab, max_seq_len=max_seq_len)
		Scrfield[phase] = scrdata

	saveToPickle('OpSub_data.pt', Scrfield)
	print('OpSub data is saved!\n')

	return 1


def proc_emoset(dirt, phaselist, emoset, min_count, max_seq_len):
	""" Build data from emotion sets """

	diadict = Dictionary('dialogue')
	emodict = Dictionary('emotion')
	diadict, emodict = buildEmodict(dirt=dirt, phaselist=phaselist, diadict=diadict, emodict=emodict)
	diadict.delRare(min_count=min_count, padunk=True)
	emodict.delRare(min_count=0, padunk=False)
	saveToPickle(emoset + '_emodict.pt', emodict)
	print('Emotions:\n {}\n {}\n'.format(emodict.word2index, emodict.word2count))

	# add the emodict for training set
	tr_diadict = Dictionary('dialogue_tr')
	tr_emodict = Dictionary('emotion_tr')
	tr_diadict, tr_emodict = buildEmodict(dirt=dirt, phaselist=['train'], diadict=tr_diadict, emodict=tr_emodict)
	tr_diadict.delRare(min_count=min_count, padunk=True)
	tr_emodict.delRare(min_count=0, padunk=False)
	saveToPickle(emoset + '_tr_emodict.pt', tr_emodict)
	print('Training set emotions:\n {}\n {}\n'.format(tr_emodict.word2index, tr_emodict.word2count))

	# load in vocab
	vocab = loadFrPickle('glob_vocab.pt')
	# index and put into fields
	Emofield = dict()
	for phase in phaselist:
		diafield = indexEmo(dirt=dirt, phase=phase, diadict=vocab, emodict=emodict, max_seq_len=max_seq_len)
		Emofield[phase] = diafield

	emo_path = emoset + '_data.pt'
	saveToPickle(emo_path, Emofield)
	print('Data written into {}!!\n'.format(emo_path))

	return 1


def main():
	''' Main function '''
	parser = argparse.ArgumentParser()
	parser.add_argument('-emoset', type=str)
	parser.add_argument('-min_count', type=int, default = 2)
	parser.add_argument('-max_seq_len', type=int, default=60)
	parser.add_argument('-datatype', type=str, default='opsub')

	opt = parser.parse_args()

	print(opt, '\n')

	phaselist = ['train', 'dev', 'test']

	if opt.datatype == 'opsub':
		global_vocab(phaselist=phaselist, min_count=opt.min_count, max_seq_len=opt.max_seq_len)

	if opt.datatype == 'emo':
		dirt = 'Data/' + opt.emoset + '/' + opt.emoset + '_'
		proc_emoset(dirt=dirt, phaselist=phaselist, emoset=opt.emoset, min_count=opt.min_count, max_seq_len=opt.max_seq_len)


if __name__ == '__main__':

	main()
