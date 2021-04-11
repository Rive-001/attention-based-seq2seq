import torch
import torch.nn as nn
import torch.nn.utils as utils
import torch.nn.functional as f

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class Attention(nn.Module):
    '''
    Attention is calculated using key, value and query from Encoder and decoder.
    '''
    def __init__(self):
        super(Attention, self).__init__()

    def forward(self, query, key, value, x_lens):
        '''
        :param query :(batch_size, hidden_size) Query is the output of LSTMCell from Decoder
        :param keys: (batch_size, max_len, encoder_size) Key Projection from Encoder
        :param values: (batch_size, max_len, encoder_size) Value Projection from Encoder
        :return context: (batch_size, encoder_size) Attended Context
        :return attention_mask: (batch_size, max_len) Attention mask that can be plotted 
        '''

        # key -> (batch_size, max_len, hidden_size)
        # query -> (batch_size, hidden_size)
        # energy -> (batch_size, max_len)
        energy = torch.bmm(key, query.unsqueeze(2)).squeeze(2)

        # mask -> (batch_size, max_len)
        mask = torch.arange(energy.size(1)).unsqueeze(0)>=x_lens.unsqueeze(1)
        mask = mask.to(DEVICE)

        energy.masked_fill_(mask, -1e9)

        # attention -> (batch_size, max_len)
        attention = torch.softmax(energy, dim=1)

        # value -> (batch_size, max_len, hidden_size)
        # context -> (batch_size, hidden_size)
        context = torch.bmm(attention.unsqueeze(1), value).squeeze(1)

        return context, attention.unsqueeze(1)

class LockedDropout(nn.Module):
    """
    Args:
        p (float): Probability of an element in the dropout mask to be zeroed.
    """

    def __init__(self, p=0.5):
        self.p = p
        super().__init__()

    def forward(self, x):
        """
        Args:
            x (:class:`torch.FloatTensor` [sequence length, batch size, rnn hidden size]): Input to
                apply dropout too.
        """
        if not self.training or not self.p:
            return x
        x = x.clone()
        mask = x.new_empty(1, x.size(1), x.size(2), requires_grad=False).bernoulli_(1 - self.p)
        mask = mask.div_(1 - self.p)
        mask = mask.expand_as(x)
        return x * mask


class pBLSTM(nn.Module):
    '''
    Pyramidal BiLSTM
    '''
    def __init__(self, input_dim, hidden_dim):
        super(pBLSTM, self).__init__()
        self.blstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=1, bidirectional=True)
        self.dropout = LockedDropout(0.3)

    def forward(self, x, lens):
        '''
        :param x :(N, T) input to the pBLSTM
        :return output: (N, T, H) encoded sequence from pyramidal Bi-LSTM 
        '''

        x, _ = utils.rnn.pad_packed_sequence(x)

        # x -> (T,B,D)
        x = x.transpose(0,1)
        B,T,D = x.size()
        if T%2!=0:
            x = x[:,:-1,:]
        x = torch.reshape(x, (B,T//2,D*2))
        # x -> (T/2,B,D*2)
        x = x.transpose(0,1)

        x = self.dropout(x)

        x = utils.rnn.pack_padded_sequence(x, lengths=lens, batch_first=False, enforce_sorted=False)

        return self.blstm(x)


class Encoder(nn.Module):
    '''
    Encoder takes the utterances as inputs and returns the key and value.
    '''
    def __init__(self, input_dim, hidden_dim, value_size=128,key_size=128):
        super(Encoder, self).__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=1, bidirectional=True)
        
        ### Add code to define the blocks of pBLSTMs! ###
        self.pblstm1 = pBLSTM(hidden_dim*4,hidden_dim)
        self.pblstm2 = pBLSTM(hidden_dim*4,hidden_dim)
        self.pblstm3 = pBLSTM(hidden_dim*4,hidden_dim)

        self.key_network = nn.Linear(hidden_dim*2, value_size)
        self.value_network = nn.Linear(hidden_dim*2, key_size)

    def forward(self, x, lens, isTrain=True):
        rnn_inp = utils.rnn.pack_padded_sequence(x, lengths=lens, batch_first=False, enforce_sorted=False)
        outputs, _ = self.lstm(rnn_inp)

        ### Use the outputs and pass it through the pBLSTM blocks! ###
        lens = lens//2
        outputs, _ = self.pblstm1(outputs, lens)
        lens = lens//2
        outputs, _ = self.pblstm2(outputs, lens)
        lens = lens//2
        outputs, _ = self.pblstm3(outputs, lens)

        linear_input, _ = utils.rnn.pad_packed_sequence(outputs)
        linear_input = linear_input.transpose(0,1)
        keys = self.key_network(linear_input)
        keys = keys.transpose(0,1)
        value = self.value_network(linear_input)
        value = value.transpose(0,1)


        return keys, value


class Decoder(nn.Module):
    '''
    Decoder network
    '''
    def __init__(self, vocab_size, hidden_dim, value_size=128, key_size=128, isAttended=False):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, 256, padding_idx=0)
        self.lstm1 = nn.LSTMCell(input_size=256 + value_size, hidden_size=hidden_dim)
        self.lstm2 = nn.LSTMCell(input_size=hidden_dim, hidden_size=hidden_dim)
        self.lstm3 = nn.LSTMCell(input_size=hidden_dim, hidden_size=key_size)

        self.isAttended = isAttended
        if (isAttended == True):
            self.attention = Attention()
            self.attention = self.attention.to(DEVICE)

        self.character_prob = nn.Linear(key_size + value_size, vocab_size)

        self.character_prob.weight = self.embedding.weight
        

    def forward(self, key, values, x_lens, text=None, isTrain=True, tf_rate=0.05):
        '''
        :param key :(T, N, key_size) Output of the Encoder Key projection layer
        :param values: (T, N, value_size) Output of the Encoder Value projection layer
        :param text: (N, text_len) Batch input of text with text_length
        :param isTrain: Train or eval mode
        :return predictions: Returns the character perdiction probability 
        '''
        # raw_w = getattr(self.embedding, 'weight')
        # w = nn.functional.dropout(raw_w, p=0.1, training=isTrain)
        # setattr(self.embedding, 'weight', nn.Parameter(w))


        att_list = []
        batch_size = key.size(1)

        if (isTrain == True):
            max_len =  text.shape[1]
            # embeddings = self.embedding(torch.cat([(torch.ones(*(batch_size, 1), dtype=torch.long)*33).to(DEVICE),text],dim=1))
            embeddings = self.embedding(text).to(DEVICE)
        else:
            max_len = 600

        predictions = []
        hidden_states = [None, None, None]
        # prediction = torch.zeros(batch_size,1,dtype=torch.long).to(DEVICE)
        prediction = (torch.ones(*(batch_size, 1), dtype=torch.long)*33).to(DEVICE)
        context = torch.zeros((values.size(1),values.size(2))).to(DEVICE)

        for i in range(max_len):

            if (isTrain):

                char_embed = embeddings[:,i,:]
                use_previous = torch.rand(1).item()
                if use_previous<=tf_rate and i>0:

                    char_embed = self.embedding(torch.argmax(out_prediction,dim=-1))
            
            elif i==0:

                char_embed = self.embedding(prediction)
            
            else:

                char_embed = self.embedding(torch.argmax(out_prediction, dim=-1))

            
            
            if self.isAttended==True:

                inp = torch.cat([char_embed.squeeze(1), context], dim=1)
            else:
                inp = torch.cat([char_embed, values[-1,:,:]], dim=1)
            
            hidden_states[0] = self.lstm1(inp, hidden_states[0])

            inp_2 = hidden_states[0][0]
            hidden_states[1] = self.lstm2(inp_2, hidden_states[1])

            inp_3 = hidden_states[1][0]
            hidden_states[2] = self.lstm3(inp_3, hidden_states[2])

            ### Compute attention from the output of the second LSTM Cell ###
            output = hidden_states[2][0]
            
            if self.isAttended==True:
                query = output
                context, att = self.attention(query, key.transpose(0,1), values.transpose(0,1), x_lens)
                att_list.append(att)


            if self.isAttended==True:
                prediction = self.character_prob(torch.cat([output, context], dim=1))
            else:
                prediction = self.character_prob(torch.cat([output, values[-1,:,:]], dim=1))
            
            out_prediction = f.gumbel_softmax(prediction, dim=-1)

            if (isTrain)!=True:
                prediction = out_prediction
            
            predictions.append(prediction.unsqueeze(1))

        if self.isAttended==True:
        
            return torch.cat(predictions, dim=1), torch.cat(att_list, dim=1)
        else:
            return torch.cat(predictions, dim=1), torch.zeros((10,10,10))


class Seq2Seq(nn.Module):
    '''
    We train an end-to-end sequence to sequence model comprising of Encoder and Decoder.
    '''
    def __init__(self, input_dim, vocab_size, hidden_dim, value_size=128, key_size=128, isAttended=False):
        super(Seq2Seq, self).__init__()
        self.encoder = Encoder(input_dim, 256).to(DEVICE)
        self.decoder = Decoder(vocab_size, 256, isAttended=True).to(DEVICE)

    def forward(self, speech_input, speech_len, text_input=None, isTrain=True, epoch=0):
        tf_rate = 0.05+(epoch//2)*0.0125
        # tf_rate = 1
        key, value = self.encoder(speech_input, speech_len, isTrain)
        if (isTrain == True):
            predictions, att = self.decoder(key, value, speech_len, text_input, tf_rate=tf_rate)
        else:
            predictions, att = self.decoder(key, value, speech_len, text=None, isTrain=False)
        
        return predictions, att
