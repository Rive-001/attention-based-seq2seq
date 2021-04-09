import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from models import Seq2Seq
from train_test import train, test, val
from dataloader import load_data, collate_train_val, collate_test, transform_letter_to_index, Speech2TextDataset

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

LETTER_LIST = ['<pad>', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', \
               'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '-', "'", '.', '_', '+', ' ','<sos>','<eos>']

def main():
    model = Seq2Seq(input_dim=40, vocab_size=len(LETTER_LIST), hidden_dim=128)
    optimizer = optim.SGD(model.parameters(), lr=1e-4, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.93)
    criterion = nn.CrossEntropyLoss(reduction='none')
    init_epoch = 0
    nepochs = 50
    batch_size = 64 if DEVICE == 'cuda' else 1

    speech_train, speech_valid, speech_test, transcript_train, transcript_valid = load_data()
    character_text_train = transform_letter_to_index(transcript_train, LETTER_LIST)
    character_text_valid = transform_letter_to_index(transcript_valid, LETTER_LIST)

    train_dataset = Speech2TextDataset(speech_train, character_text_train)
    val_dataset = Speech2TextDataset(speech_valid, character_text_valid)
    test_dataset = Speech2TextDataset(speech_test, None, False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_train_val)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_train_val)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_test)

    val_distances = []
    exp = 27
    # model.load_state_dict(torch.load('BestModel9.pth'))
    
    with open('stats_{}'.format(exp),'w') as file:

        file.write('Experiment: {}\n'.format(exp))

    for epoch in range(init_epoch, nepochs):
        train(model, train_loader, criterion, optimizer, scheduler, epoch, exp)
        val_distances.append(val(model, val_loader, epoch, exp))
        if val_distances[-1]==min(val_distances):
            torch.save(model.state_dict(), 'BestModel{}.pth'.format(exp))

        if epoch%3==0 or epoch==nepochs-1:
            test(model, test_loader, exp)



if __name__ == '__main__':
    main()