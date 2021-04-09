import time
import torch
import distance
from util import plot_attn_flow
import torch.nn as nn
LETTER_LIST = ['<pad>', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', \
               'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '-', "'", '.', '_', '+', ' ','<sos>','<eos>']
### Add Your Other Necessary Imports Here! ###

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def train(model, train_loader, criterion, optimizer, scheduler, epoch, exp=0):
    print('Epoch:',epoch)
    with open('stats_{}'.format(exp),'a') as file:

        file.write('Epoch: {}\n'.format(epoch))


    epoch_loss = 0
    n_tokens = 0
    distances = 0
    ddistances = 0
    num_strings = 0
    model.train()
    model.to(DEVICE)
    start = time.time()

    for x,y,x_lens,y_lens in train_loader:

        # torch.autograd.set_detect_anomaly(True)
        optimizer.zero_grad()

        x = x.to(DEVICE)
        y = torch.cat([torch.ones((1,y.size(1)),dtype=torch.long)*33,y], dim=0)
        y = y.to(DEVICE)

        b_size = y.size(1)

        predictions, att = model(x,x_lens,torch.transpose(y, 1, 0), epoch=epoch)

        best_predictions = torch.argmax(predictions, dim=-1).detach().cpu()

        # best_sequences = []
        # for seq in best_predictions:
        #     seq = seq.tolist()
        #     if 34 in seq:
        #         seq = seq[:seq.index(34)]
        #     s = ''.join([LETTER_LIST[i] for i in seq if i not in [0,33]])
        #     best_sequences.append(s)

        # true_sequences = []
        # for seq in y.transpose(0,1):
        #     s = ''.join([LETTER_LIST[i] for i in seq if i not in [0,33]])
        #     true_sequences.append(s)

        # for b in range(b_size):

        #     distances+= distance.levenshtein(best_sequences[b],true_sequences[b])
        #     num_strings+= 1

        predictions = predictions.transpose(1,2)
        y = y[1:,:]
        y = y.transpose(0,1)

        y = torch.cat([y,(torch.ones((y.size(0),1), dtype=torch.int)*34).to(DEVICE)], dim=1)

        loss = criterion(predictions,y)

        max_len = loss.size(1)

        mask = torch.arange(max_len).unsqueeze(0)>y_lens.unsqueeze(1)
        mask = mask.to(DEVICE)

        loss.masked_fill_(mask, 0)

        loss = loss.sum()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 2)

        optimizer.step()

        epoch_loss+= loss.item()
        n_tokens+= y_lens.sum().item()

    scheduler.step()

    perplexity = epoch_loss/n_tokens

    print('Train Perplexity:',perplexity)

    # train_distance = distances/num_strings

    # print('Train distance:', train_distance)

    # print('Best',best_sequences)
    # print('Truth',true_sequences)

    with open('stats_{}'.format(exp),'a') as file:

        file.write('Train Perplexity:{}\n'.format(perplexity))



    # 1) Iterate through your loader
        # 2) Use torch.autograd.set_detect_anomaly(True) to get notices about gradient explosion
        
            # 3) Set the inputs to the device.

            # 4) Pass your inputs, and length of speech into the model.

            # 5) Generate a mask based on the lengths of the text to create a masked loss. 
            # 5.1) Ensure the mask is on the device and is the correct shape.

            # 6) If necessary, reshape your predictions and origianl text input 
            # 6.1) Use .contiguous() if you need to. 

            # 7) Use the criterion to get the loss.

            # 8) Use the mask to calculate a masked loss. 

            # 9) Run the backward pass on the masked loss. 

            # 10) Use torch.nn.utils.clip_grad_norm(model.parameters(), 2)
            
            # 11) Take a step with your optimizer

            # 12) Normalize the masked loss

            # 13) Optionally print the training loss after every N batches

    end = time.time()
    # print('Train time:',end-start,'seconds')

def val(model, val_loader, epoch, exp):

    model.eval()
    model.to(DEVICE)
    distances = 0
    num_strings = 0
    epoch_loss = 0
    n_tokens = 0
    criterion = nn.CrossEntropyLoss(reduction='none')
    start = time.time()

    for x,y,x_lens,y_lens in val_loader:

        x = x.to(DEVICE)
        y = y.transpose(0,1)

        b_size = y.size(0)

        predictions, att = model(x,x_lens, text_input=None, isTrain=False)

        att = att.detach().cpu()

        for i in range(5):
            plt = plot_attn_flow(att[i,:,:], './attention/Attention{}_epoch{}_exp{}.png'.format(i,epoch,exp))

        ## Greedy Search
        best_predictions = torch.argmax(predictions, dim=-1).detach().cpu()
 
        
        best_sequences = []
        for seq in best_predictions:
            seq = seq.tolist()
            if 34 in seq:
                seq = seq[:seq.index(34)]
            s = ''.join([LETTER_LIST[i] for i in seq if i not in [0,33]])
            best_sequences.append(s)

        true_sequences = []
        for seq in y.transpose(0,1):
            s = ''.join([LETTER_LIST[i] for i in seq if i not in [0,33]])
            true_sequences.append(s)

        for b in range(b_size):

            distances+= distance.levenshtein(best_sequences[b],true_sequences[b])
            num_strings+= 1

    val_distance = distances/num_strings

    print('Val distance:', val_distance)

    with open('stats_{}'.format(exp),'a') as file:

        file.write('Val distance:{}\n'.format(val_distance))

    end = time.time()
    # print('Validation time:',end-start,'seconds')

    return val_distance

    
def test(model, test_loader, exp):
    ### Write your test code here! ###
    
    # model.load_state_dict(torch.load('BestModel{}.pth'.format(exp)))
    model.eval()
    model.to(DEVICE)
    best_outputs = []

    for x,x_lens in test_loader:

        x = x.to(DEVICE)

        b_size = x.size(1)

        predictions, _ = model(x,x_lens,text_input=None, isTrain=False)

        ## Greedy Search
        best_predictions = torch.argmax(predictions, dim=-1).detach().cpu()
        
        best_sequences = []
        for seq in best_predictions:
            seq = seq.tolist()
            if 34 in seq:
                seq = seq[:seq.index(34)]
            s = ''.join([LETTER_LIST[i] for i in seq if i not in [0,33]])
            best_sequences.append(s)


        best_outputs+= best_sequences

    with open('submission{}.csv'.format(exp),'w') as file:

        file.write('id,label\n')
        for idx,text in enumerate(best_outputs):
            file.write(str(idx)+','+text+'\n')

    return
