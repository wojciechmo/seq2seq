import cv2
import re, os
import unidecode
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

hidden_size = 200
input_embedd_size = 300 
output_embedd_size = 200 
train_iterations = 5000000
train_eval_interval = 50000
train_save_interval = 500000
train_loss_average_interval = 1000

SOS_token = 0	# start of sequence
EOS_token = 1	# end of sequence
MIN_LENGTH = 3	# without EOS token
MAX_LENGTH = 15	# with EOS token

model_path = './model'
sentence_pairs_path = 'eng-spa.txt'
eval_sentences = ['el es viejo', 'cuando se construyo', 'quien es tu padre']

#------------------------------------------------------------------------------------------------------
#------------------------------------------ data reader -----------------------------------------------
#------------------------------------------------------------------------------------------------------

def unicodeToAscii(s):
	
    return ''.join(c for c in unidecode.unidecode(s))

def sent2seq(lang, sentence):
	
	return [lang.word2index[word] for word in sentence.split(' ')]
	
def seq2sent(lang, sequence):
	
	return ' '.join([lang.index2word[indx] for indx in sequence])	

from io import open		
		
class LanguageStats:
	
    def __init__(self):
		
        self.word2index = {"SOS": 0, "EOS": 1}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2

    def add_sentence(self, sentence):
		
        for word in sentence.split(' '):
            self.add_word(word)

    def add_word(self, word):
		
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.index2word[self.n_words] = word
            self.n_words += 1

class Reader(object):
	
	def __init__ (self, filepath):
		
		raw_text = open(filepath, encoding='utf-8').read()
		lines = raw_text.splitlines()

		self.pairs = []

		for line in lines:
			
			lang2_sent, lang1_sent = line.split('\t') 
			
			lang1_sent = unicodeToAscii(lang1_sent)
			lang2_sent = unicodeToAscii(lang2_sent)
			lang2_sent = re.sub('[!?.,]', '', lang2_sent)
			lang1_sent = re.sub('[!?.,]', '', lang1_sent)

			if (len(lang1_sent.split(' '))< MAX_LENGTH and len(lang1_sent.split(' '))>= MIN_LENGTH and len(lang2_sent.split(' '))< MAX_LENGTH and len(lang2_sent.split(' '))>= MIN_LENGTH ):
				self.pairs.append([lang1_sent.lower(), lang2_sent.lower()])

		self.input_lang = LanguageStats()
		self.output_lang = LanguageStats()
		
		for pair in self.pairs:
			self.input_lang.add_sentence(pair[0])
			self.output_lang.add_sentence(pair[1])
			
		self.idx = 0
		self.pairs_num = len(self.pairs)
		self.choice = range(self.pairs_num)
		np.random.shuffle(self.choice)
		
		print ('number of pairs: ', self.pairs_num)
		print ('input lang distinct words: ', self.input_lang.n_words)
		print ('output lang distinct words: ', self.output_lang.n_words)
		
	def get_pair(self):
		
		pair = self.pairs[self.choice[self.idx]]
		input_sentence =  pair[0] + " EOS"
		target_sentence = pair[1] + " EOS" 		#translation + EOS
		teacher_sentence = "SOS " + pair[1]		#SOS + translation
		
		input_sequence = sent2seq(self.input_lang, input_sentence)
		target_sequence = sent2seq(self.output_lang, target_sentence)
		teacher_sequence = sent2seq(self.output_lang, teacher_sentence)

		if self.idx == 0:
			print ('sample from training data:')
			print (input_sentence, ' -> ', input_sequence)
			print (target_sentence, ' -> ', target_sequence)
			print (teacher_sentence, ' -> ', teacher_sequence)
		
		self.idx = self.idx + 1
		
		if self.idx == self.pairs_num:
			self.idx = 0
			np.random.shuffle(self.choice)
			
		return input_sequence, target_sequence, teacher_sequence

#------------------------------------------------------------------------------------------------------
#--------------------------------------------- encoder ------------------------------------------------
#------------------------------------------------------------------------------------------------------	
		
class Encoder(nn.Module):
	
    def __init__(self, vocab_size, embedd_size, hidden_size):
		
        super(Encoder, self).__init__()
        
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embedd_size)
        self.lstm = nn.LSTM(embedd_size, hidden_size)

    def forward(self, input_seq, hidden):
		
		# input - LongTensor of shape(seq_len, 1)
		# hidden - tuple (hidden_state, cell-state), each element is floatTensor of shape(1, 1, hidden_size)

		embedded = self.embedding(input_seq) # floatTensor of shape (seq_len, 1, embedd_size)
		output, hidden = self.lstm(embedded, hidden) # floatTensor of shape (seq_len, 1, hidden_size)

		return output, hidden

    def initHidden(self):
		
        return (Variable(torch.zeros(1, 1, self.hidden_size)).cuda(),
				Variable(torch.zeros(1, 1, self.hidden_size)).cuda())	

#------------------------------------------------------------------------------------------------------
#--------------------------------------------- decoder ------------------------------------------------
#------------------------------------------------------------------------------------------------------

class Decoder(nn.Module):
	
	def __init__(self, vocab_size, embedd_size, hidden_size):
		
		super(Decoder, self).__init__()
        
		self.hidden_size = hidden_size
		self.embedding = nn.Embedding(vocab_size, embedd_size)
		self.lstm = nn.LSTM(embedd_size, hidden_size)
		self.atten = nn.Linear(hidden_size, MAX_LENGTH)
		self.out = nn.Linear(2*hidden_size, vocab_size)
        
	def forward(self, input, hidden, encoder_outputs, use_teacher):
	
		if use_teacher == True: # pass target words to decoder inputs
		
			# input - LongTensor of shape(seq_len, 1)
			# hidden - tuple (hidden_state, cell-state), each element is floatTensor of shape(1, 1, hidden_size)
			# encoder_outputs - floatTensor of shape(MAX_LENGTH, 1, hidden_size)
			
			embedded = self.embedding(input) # floatTensor of shape (seq_len, 1, embedd_size)
			output, hidden = self.lstm(embedded, hidden) # floatTensor of shape (seq_len, 1, hidden_size)
			attentions = F.softmax(self.atten(output), dim=2) # floatTensor of shape (seq_len, 1, MAX_LENGTH)
			context_vector = torch.matmul(torch.squeeze(attentions, 1), torch.squeeze(encoder_outputs, 1))
			context_vector = context_vector.unsqueeze(1) # floatTensor of shape (seq len, 1, hidden_size)
			attention_vector = torch.cat((context_vector, output), 2)
			output = self.out(attention_vector) # floatTensor of shape (seq_len, 1, vocab_size)
			output = F.log_softmax(output, dim=2)
			
		else: # at each time step pass computed word to decoder input 
			
			input_length = input.size()[0]			
			input_token = input[0].unsqueeze(0)
			output = []
			
			for i in range(input_length):
			
				embedded = self.embedding(input_token)
				output_token, hidden = self.lstm(embedded, hidden)
				attentions = F.softmax(self.atten(output_token), dim=2)	
				
				context_vector = torch.matmul(torch.squeeze(attentions, 1),torch.squeeze(encoder_outputs, 1))
				context_vector = context_vector.unsqueeze(1)
				attention_vector = torch.cat((context_vector, output_token), 2)
				output_token = self.out(attention_vector)
				preds_token = F.log_softmax(output_token, dim=2)
				
				topv, topi = preds_token.topk(1)
				pred_token = topi[0]
				output.append(preds_token)
				input_token = pred_token

			output = torch.cat(output,0)

		return output
		
	def evaluate(self, hidden, encoder_outputs):
			
		input_token = Variable(torch.LongTensor([[SOS_token]])).cuda()

		output = []
		attention_matrix = []
			
		for i in range(MAX_LENGTH):
			
			embedded = self.embedding(input_token)
			output_token, hidden = self.lstm(embedded, hidden)
			attentions = F.softmax(self.atten(output_token), dim=2)	
			context_vector = torch.matmul(torch.squeeze(attentions, 1), torch.squeeze(encoder_outputs, 1))
			context_vector = context_vector.unsqueeze(1)
			
			attention_vector = torch.cat((context_vector, output_token), 2)
			output_token = self.out(attention_vector)
			preds_token = F.log_softmax(output_token, dim=2)
			
			topv, topi = preds_token.topk(1)
			pred_token = topi[0]
			pred_token_data = pred_token[0].data.cpu().numpy()[0]
			
			if pred_token_data == EOS_token:
				break
			else:
				output.append(pred_token_data)	
				attention_matrix.append(attentions[0][0].data.cpu().numpy())
				input_token = pred_token

		return output, np.array(attention_matrix)
			
	def initHidden(self, hidden_state):
		
		return (hidden_state, Variable(torch.zeros(1, 1, self.hidden_size)).cuda())	
		
#------------------------------------------------------------------------------------------------------
#------------------------------------------ evaluation ------------------------------------------------
#------------------------------------------------------------------------------------------------------		

def evaluate(input_sentence, encoder, decoder, reader, path, iteration=None):

	make_dir(path)
	
	input_sentence = input_sentence + " EOS"
	input_sequence = sent2seq(reader.input_lang, input_sentence)
	input_var = Variable(torch.LongTensor(input_sequence)).unsqueeze(1).cuda()

	encoder_hidden = encoder.initHidden()
	encoder_outputs, (context_vector, _) = encoder(input_var, encoder_hidden)

	input_length = len(input_var)
	if input_length != MAX_LENGTH:
		encoder_outputs_fill = Variable(torch.zeros(MAX_LENGTH-input_length, 1, hidden_size)).cuda()
		encoder_outputs = torch.cat((encoder_outputs, encoder_outputs_fill), dim=0)

	decoder_hidden = decoder.initHidden(context_vector)
	decoded_sequence, attention_matrix = decoder.evaluate(decoder_hidden, encoder_outputs)
	decoded_sentence = seq2sent(reader.output_lang, decoded_sequence)
	
	if attention_matrix.shape[0] == 0: return
	
	attention_matrix = cv2.resize(attention_matrix, (0, 0), fx=100.0, fy=100.0, interpolation=cv2.INTER_NEAREST)
	attention_matrix = (attention_matrix*255).astype(np.uint8)
	attention_matrix = cv2.applyColorMap( attention_matrix, cv2.COLORMAP_OCEAN) #cv2.COLORMAP_HOT
	
	input_sentence = input_sentence[:-4] # strip EOS 
	filepath = path + '/'
	if iteration is not None:
		filepath = filepath + 'iter-' + str(iteration) + ': '
	filepath = filepath + input_sentence + '->' + decoded_sentence
	
	cv2.imwrite(filepath + '.png', attention_matrix)
	
	print ('evaluation result:')
	print ('input sentence:', input_sentence)
	print ('output sentence:', decoded_sentence)

def make_dir(directory):

	if not os.path.exists(directory):
		os.makedirs(directory)		

#------------------------------------------------------------------------------------------------------
#-------------------------------------------- training ------------------------------------------------
#------------------------------------------------------------------------------------------------------

if __name__ == "__main__":

	reader = Reader(sentence_pairs_path)
	input_vocab_size = reader.input_lang.n_words
	output_vocab_size = reader.output_lang.n_words
	encoder = Encoder(input_vocab_size, input_embedd_size, hidden_size).cuda()
	decoder = Decoder(output_vocab_size, output_embedd_size, hidden_size).cuda()

	criterion = nn.NLLLoss()
	encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=0.0001)
	decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=0.0001)
	
	loss_averaged = 0.0
	make_dir(model_path)
	
	for train_iteration in range(train_iterations):
		
		if (train_iteration + 1) % train_eval_interval==0:
			
			for i, eval_sentence in enumerate(eval_sentences):
				evaluate(eval_sentence, encoder, decoder, reader, './train_results/' + str(i + 1), train_iteration + 1)
		
		input_sequence, target_sequence, teacher_sequence = reader.get_pair()
		
		input_var = Variable(torch.LongTensor(input_sequence)).unsqueeze(1).cuda()
		target_var = Variable(torch.LongTensor(target_sequence)).unsqueeze(1).cuda()
		teacher_var = Variable(torch.LongTensor(teacher_sequence)).unsqueeze(1).cuda()
		
		encoder_hidden = encoder.initHidden()
		encoder_outputs, (context_vector, _) = encoder(input_var, encoder_hidden)

		input_length = len(input_var)
		if input_length != MAX_LENGTH:
			encoder_outputs_fill = Variable(torch.zeros(MAX_LENGTH-input_length, 1, hidden_size)).cuda()
			encoder_outputs = torch.cat((encoder_outputs, encoder_outputs_fill), dim=0)
			
		decoder_hidden = decoder.initHidden(context_vector)
		
		if np.random.uniform() > 0.5:
			preds = decoder(teacher_var, decoder_hidden, encoder_outputs, use_teacher=True)
		
		else:
			preds = decoder(teacher_var, decoder_hidden, encoder_outputs, use_teacher=False)

		preds = torch.squeeze(preds, 1)
		topv, topi = preds.data.topk(1)

		encoder_optimizer.zero_grad()
		decoder_optimizer.zero_grad()

		loss = criterion(preds, torch.squeeze(target_var, 1))
		loss.backward()
		
		loss_averaged += loss.data[0]

		encoder_optimizer.step()
		decoder_optimizer.step()
		
		if (train_iteration + 1) % train_save_interval == 0:	
			
			torch.save(encoder, model_path + '/encoder-iter-' + str(train_iteration + 1) + '.pt')
			torch.save(decoder, model_path + '/decoder-iter-' + str(train_iteration + 1) + '.pt')
			
		if (train_iteration + 1) % train_loss_average_interval == 0:	
			
			print ('iter:', train_iteration + 1, 'loss:', loss_averaged/float(train_loss_average_interval))
			loss_averaged = 0.0

	torch.save(encoder, model_path + '/encoder-final.pt')
	torch.save(decoder, model_path + '/decoder-final.pt')
