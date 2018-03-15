import sys
from train import *

encoder_path = './model/encoder-final.pt'
decoder_path = './model/decoder-final.pt'
sentence_pairs_path = 'eng-spa.txt'
test_sentences_path = 'test_sentences.txt'

decoder = torch.load(decoder_path)
encoder = torch.load(encoder_path)
reader = Reader(sentence_pairs_path)
test_sentences = open(test_sentences_path).read().splitlines()

for test_sentence in test_sentences:
	for word in test_sentence.split(' '):
		if not word in reader.input_lang.word2index:
			print 'ERROR in sentence: ' + test_sentence
			print 'word: \'' + word + '\' does not exist in input bag of words'
			sys.exit()

for i, test_sentence in enumerate(test_sentences):
	evaluate(test_sentence, encoder, decoder, reader, './eval_results')
