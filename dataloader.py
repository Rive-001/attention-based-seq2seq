import numpy as np
import torch
from torch.utils.data import Dataset 
from torch.nn.utils.rnn import pad_sequence


'''
Loading all the numpy files containing the utterance information and text information
'''
def load_data():
    speech_train = np.load('./hw4p2/train.npy', allow_pickle=True, encoding='bytes')
    speech_valid = np.load('./hw4p2/dev.npy', allow_pickle=True, encoding='bytes')
    speech_test = np.load('./hw4p2/test.npy', allow_pickle=True, encoding='bytes')

    transcript_train = np.load('./hw4p2/train_transcripts.npy', allow_pickle=True,encoding='bytes')
    transcript_valid = np.load('./hw4p2/dev_transcripts.npy', allow_pickle=True,encoding='bytes')

    return speech_train, speech_valid, speech_test, transcript_train, transcript_valid


'''
Transforms alphabetical input to numerical input, replace each letter by its corresponding 
index from letter_list
'''
def transform_letter_to_index(transcript, letter_list):
    '''
    :param transcript :(N, ) Transcripts are the text input
    :param letter_list: Letter list defined above
    :return letter_to_index_list: Returns a list for all the transcript sentence to index
    '''
    
    list_of_strings = [' '.join([s.decode('UTF-8') for s in t]) for t in transcript]
    letter_to_index_list = [[letter_list.index(ch) for ch in list(t)] for t in list_of_strings]
    return letter_to_index_list


def create_dictionaries(letter_list):
    letter2index = dict()
    index2letter = dict()
    return letter2index, index2letter


class Speech2TextDataset(Dataset):
    '''
    Dataset class for the speech to text data. 
    '''
    def __init__(self, speech, text=None, isTrain=True):
        self.speech = speech
        self.isTrain = isTrain
        if (text is not None):
            self.text = text

    def __len__(self):
        return self.speech.shape[0]

    def __getitem__(self, index):
        if (self.isTrain == True):
            return torch.tensor(self.speech[index].astype(np.float32)), torch.tensor(self.text[index])
        else:
            return torch.tensor(self.speech[index].astype(np.float32))


def collate_train_val(batch_data):
    ### Return the padded speech and text data, and the length of utterance and transcript ###

    X = [item[0] for item in batch_data]
    Y = [item[1] for item in batch_data]
    X_lens = torch.Tensor([item.size()[0] for item in X])
    Y_lens = torch.Tensor([item.size()[0] for item in Y])
    X = pad_sequence(X)
    Y = pad_sequence(Y)

    return X, Y, X_lens, Y_lens 


def collate_test(batch_data):
    ### Return padded speech and length of utterance ###

    # X = [item[0] for item in batch_data]
    
    X_lens = torch.Tensor([item.size()[0] for item in batch_data])
    
    X = pad_sequence(batch_data)
    

    return X, X_lens
