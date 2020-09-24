"""
Modules including wordenc, sentenc, contenc, biLM and mlpdec.
Date: 2020/09/24
"""
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence
import Const


# Normal attention
def get_attention(q, k, v, attn_mask=None):
	"""
	:param : (batch, seq_len, seq_len)
	:return: (batch, seq_len, seq_len)
	"""
	attn = torch.matmul(q, k.transpose(1, 2))
	if attn_mask is not None:
		attn.data.masked_fill_(attn_mask, -1e10)

	attn = F.softmax(attn, dim=-1)
	output = torch.matmul(attn, v)
	return output, attn


def get_attn_pad_mask(seq_q, seq_k):
	assert seq_q.dim() == 2 and seq_k.dim() == 2

	pad_attn_mask = torch.matmul(seq_q.unsqueeze(2).float(), seq_k.unsqueeze(1).float())
	pad_attn_mask = pad_attn_mask.eq(Const.PAD)  # b_size x 1 x len_k
	#print(pad_attn_mask)

	return pad_attn_mask.cuda(seq_k.device)


def get_biattention(c, q, attn_mask=None):
	"""
	:param : (batch, seq_len, seq_len)
	:return: (batch, seq_len, seq_len)
	"""
	attn = torch.matmul(c, q.transpose(1, 2))
	if attn_mask is not None:
		attn.data.masked_fill_(attn_mask, -1e10)

	attn_c2q = F.softmax(attn, dim=-1)
	# batch x c_len x q_len
	C2Q = output = torch.matmul(attn_c2q, q)
	# batch x c_len x 2d
	attn_q2c = F.softmax(torch.max(attn, dim=-1, keepdim=True)[0], dim=1)
	# batch x c_len x 1
	Q2C_ = torch.matmul(attn_q2c.transpose(1,2), c)
	# batch x 1 x 2d
	Q2C = Q2C_.expand(C2Q.size())

	return C2Q, Q2C, attn_c2q, attn_q2c



class GRUtrans(nn.Module):
	def __init__(self, d_emb, d_out, num_layers):
		super(GRUtrans, self).__init__()
		# default encoder 2 layers
		self.gru = nn.GRU(input_size=d_emb, hidden_size=d_out,
		                  bidirectional=True, num_layers=num_layers, dropout=0.3)

	def forward(self, sent, sent_lens):
		"""
		:param sent: torch tensor, batch_size x seq_len x d_rnn_in
		:param sent_lens: numpy tensor, batch_size x 1
		:return:
		"""
		device = sent.device
		# seq_len x batch_size x d_rnn_in
		sent_embs = sent.transpose(0,1)

		# sort by length
		s_lens, idx_sort = np.sort(sent_lens)[::-1], np.argsort(-sent_lens)
		idx_unsort = np.argsort(idx_sort)

		idx_sort = torch.from_numpy(idx_sort).cuda(device)
		s_embs = sent_embs.index_select(1, Variable(idx_sort))

		# padding
		sent_packed = pack_padded_sequence(s_embs, s_lens)
		sent_output = self.gru(sent_packed)[0]
		sent_output = pad_packed_sequence(sent_output, total_length=sent.size(1))[0]

		# unsort by length
		idx_unsort = torch.from_numpy(idx_unsort).cuda(device)
		sent_output = sent_output.index_select(1, Variable(idx_unsort))

		# batch x seq_len x 2*d_out
		output = sent_output.transpose(0,1)

		return output


class wordEncoder(nn.Module):
	def __init__(self, embedding):
		super(wordEncoder, self).__init__()
		self.embedding = embedding

	def forward(self, sents):
		"""
		:param sents: batch x seq_len
		:return: batch x seq_len x d_word_vec
		"""

		if len(sents.size()) < 2:
			sents = sents.unsqueeze(0)

		w_embed = self.embedding(sents)
		return w_embed


class sentEncoder(nn.Module):
	def __init__(self, d_input, d_output, num_layers=1):
		super(sentEncoder, self).__init__()
		self.sentEnc = GRUtrans(d_emb=d_input, d_out=d_output, num_layers=num_layers)
		self.output1 = nn.Sequential(
			nn.Linear(d_output * 2, d_output),
			nn.Tanh()
		)
		self.dropout_0 = nn.Dropout(0.5)

	def forward(self, w_embed, lens):
		"""
		:param w_embed: batch x seq_len x d_input (not necessarily d_word_vec)
		:param lens:  batch x 1
		:return: batch x d_output, batch x seq_len x d_output
		"""
		w_context_ = self.sentEnc(w_embed, lens)
		w_context = self.output1(w_context_)
		s_embed = torch.max(w_context, dim=1)[0] + torch.mean(w_context, dim=1)
		s_embed = self.dropout_0(s_embed)
		# batch x d_output, batch x seq_len x d_output
		return s_embed, w_context


class sentGRUEncoder(nn.Module):
	def __init__(self, d_input, d_output, num_layers=1):
		super(sentGRUEncoder, self).__init__()
		self.sentEnc = GRUtrans(d_emb=d_input, d_out=d_output, num_layers=num_layers)
		self.output1 = nn.Sequential(
			nn.Linear(d_output * 2, d_output),
			nn.Tanh()
		)
		self.dropout_0 = nn.Dropout(0.5)

	def forward(self, w_embed, lens):
		"""
		:param w_embed: batch x seq_len x d_input (not necessarily d_word_vec)
		:param lens:  batch x 1
		:return: batch x d_output, batch x seq_len x d_output
		"""
		w_context_ = self.sentEnc(w_embed, lens)
		s_embed_ = torch.max(w_context_, dim=1)[0] + torch.mean(w_context_, dim=1)
		s_embed = self.output1(s_embed_)
		s_embed = self.dropout_0(s_embed)
		# batch x d_output, batch x seq_len x d_output
		return s_embed, w_context_


class sentGRU2LEncoder(nn.Module):
	def __init__(self, d_input, d_output, num_layers=1):
		super(sentGRU2LEncoder, self).__init__()
		self.sentEnc = GRUtrans(d_emb=d_input, d_out=d_output, num_layers=num_layers)
		self.sentEnc2 = GRUtrans(d_emb=d_output * 2, d_out=d_output, num_layers=num_layers)
		self.output1 = nn.Sequential(
			nn.Linear(d_output * 2, d_output),
			nn.Tanh()
		)
		self.dropout_0 = nn.Dropout(0.5)
		self.dropout_1 = nn.Dropout(0.5)

	def forward(self, w_embed, lens):
		"""
		:param w_embed: batch x seq_len x d_input (not necessarily d_word_vec)
		:param lens:  batch x 1
		:return: batch x d_output, batch x seq_len x d_output
		"""
		w_context_ = self.sentEnc(w_embed, lens)
		w_context_ = self.dropout_0(w_context_)
		w_context_2 = self.sentEnc2(w_context_, lens)
		w_context = self.output1(w_context_2)
		s_embed = torch.max(w_context, dim=1)[0] + torch.mean(w_context, dim=1)
		s_embed = self.dropout_1(s_embed)
		# batch x d_output, batch x seq_len x d_output
		return s_embed, w_context


class contEncoder(nn.Module):
	def __init__(self, d_input, d_output, num_layers=1):
		super(contEncoder, self).__init__()
		self.contEnc = nn.GRU(input_size=d_input, hidden_size=d_output,
		                       bidirectional=True, num_layers=num_layers, dropout=0.3)
		self.dropout_0 = nn.Dropout(0.5)

	def forward(self, s_embed):
		"""
		:param s_embed: batch x d_input (not necessarily d_hidden_low)
		:return: batch x d_output * 2
		"""
		# sents: batch x d_in
		s_embed = s_embed.unsqueeze(1)
		s_context = self.contEnc(s_embed)[0].squeeze(1)
		s_context = self.dropout_0(s_context)

		return s_context


class biLM(nn.Module):
	def __init__(self, d_input, d_output):
		super(biLM, self).__init__()
		self.cont_embed = nn.Sequential(
			nn.Linear(d_input, d_output),
			nn.Tanh()
		)

	def forward(self, t_embed, t_context, f_embed):
		"""
		:param t_embed: batch x d_hidden_low
		:param t_context: batch x d_hidden_up*2
		:param f_context: N x d_hidden_low
		:return: batch x (1+N)
		"""
		dia_len = t_embed.size()[0]
		lcont, rcont = t_context.chunk(2, 1)
		t_embed_ = t_embed[1:dia_len-1].unsqueeze(1)
		lcont_ = lcont[:dia_len-2].unsqueeze(1)
		rcont_ = rcont[2:].unsqueeze(1)
		cont_embed = self.cont_embed(torch.cat([lcont_, rcont_], dim=-1))
		true_label = torch.matmul(cont_embed, t_embed_.transpose(1,2))
		f_embed = f_embed.unsqueeze(0)
		f_embed_ = f_embed.expand([t_embed_.size()[0], f_embed.size()[1], f_embed.size()[2]])
		neg_label = torch.matmul(cont_embed, f_embed_.transpose(1,2))
		all_label = torch.cat([true_label, neg_label], dim=-1)
		all_label = all_label.squeeze(1)
		# [[1,0,0],...,[1,0,0]]

		return all_label


class mlpDecoder(nn.Module):
	def __init__(self, d_input, d_output, n_class):
		super(mlpDecoder, self).__init__()
		# concat the input with the output of RNN-up
		self.output1 = nn.Sequential(
			nn.Linear(d_input, d_output),
			nn.Tanh(),
			nn.Linear(d_output, n_class)
		)

	def forward(self, s_context):
		"""
		:param s_context: batch x d_input (not necessarily d_hidden_up * 2)
		:return: batch x n_class
		"""
		output = self.output1(s_context)
		return F.log_softmax(output, dim=1)

