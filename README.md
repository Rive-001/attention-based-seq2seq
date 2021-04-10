# Speech-to-text transcription using a sequence-to-sequence model

Speech-to-text transcription involves processing recorded speeches to obtain their textual content. In the deep learning era, this task is accomplished using neural networks such as LSTMs and Transformers. This project aims to accomplish this task by using an attention-based sequence-to-sequence neural network. The network used is based on [Listen, Attend and Spell](https://arxiv.org/abs/1508.01211) and involves an encoder for speech-utterances and a decoder to produce text. 

## Technology Stack

* **Language and Frameworks**: Python, Pytorch
* **Visualization**: [Weights & Biases](https://wandb.ai/site)
* **Computing Platform**: AWS EC2

## Model

Input is provided as speech recordings split into 13 frequency bands. The encoder is a pyramidal bi-LSTM, which learns an embedding for the speech input. This embedding is used by the decoder, which is a language model. The decoder augments the embedding with the attention matrix obtained from the entire speech input. It then learns the mapping from the input speech to the output text. The decoder was pre-trained, with the intuition that pre-training would allow it to learn how characters build up words and sentences in the English language. Heuristics for improving model performance included teacher forcing, intra-layer regularization and model compression. 

<div align="center"><img src="https://github.com/Rive-001/attention-based-seq2seq/blob/master/model%20schematic.png" width="800" height="400"></div>
<div align="center"><b>Model based on Listen, Attend and Spell</b></div>

## Results

The model was able to achieve a Levenshtein distance of 26 from the ground truth. 
