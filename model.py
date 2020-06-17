import torch
import torch.nn as nn
import torchvision.models as models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1) #reshape to send to linear for creating a fixed embedded layer for encoder
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
#         super( DecoderRNN, self).__init__()

        ''' Initialize the layers of this model.'''
        

        super().__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        # embedding layer that turns words into a vector of a specified size
        self.word_embeddings = nn.Embedding(vocab_size, embed_size)

        # the LSTM takes embedded word vectors (of a specified size) as inputs 
        # and outputs hidden states of size hidden_dim        
        self.lstm = nn.LSTM(input_size = self.embed_size, hidden_size = self.hidden_size, num_layers  = self.num_layers,batch_first = True)


        # the linear layer that maps the hidden state output dimension 
        # to the number of tags we want as output (in this case this is vocab_size)
        self.fc = nn.Linear( self.hidden_size , self.vocab_size  )

    def init_hidden( self,batch_size ):
        a = torch.zeros( self.num_layers , batch_size , self.hidden_size).to(device)
        b = torch.zeros( self.num_layers , batch_size , self.hidden_size).to(device)
        return (a,b)

#         return ( torch.zeros( self.num_layers , batch_size , self.hidden_size, device='cuda:0'),torch.zeros( self.num_layers , batch_size , self.hidden_size,device='cuda:0'))
#         return ( torch.zeros( self.num_layers , batch_size , self.hidden_size),torch.zeros( self.num_layers , batch_size , self.hidden_size))

#         initialize the hidden state (see code below)
#         self.hidden_dim = hidden_dim
#         self.hidden = self.init_hidden()
        
#     def init_hidden(self):
#     ''' At the start of training, we need to initialize a hidden state;
#        there will be none because the hidden state is formed based on perviously seen data.
#        So, this function defines a hidden state with all zeroes and of a specified size.'''
#         # The axes dimensions are (n_layers, batch_size, hidden_dim)
#         return (torch.zeros(1, 1, self.hidden_dim),
#                 torch.zeros(1, 1, self.hidden_dim))
        
    
    def forward(self, features, captions):
        ''' Define the feedforward behavior of the model.'''
        # create embedded word vectors for each word in a sentence from caption
        captions = captions[:, :-1]
        self.batch_size = features.shape[0]
        self.hidden = self.init_hidden( self.batch_size )
        
        embeds = self.word_embeddings(captions)
        input_embeds = torch.cat((features.unsqueeze(dim = 1), embeds), dim = 1)
        
        # get the output and hidden state by passing the lstm over our word embeddings
        # the lstm takes in our embeddings and hiddent state
        lstm_out, self.hidden = self.lstm(input_embeds , self.hidden)
#         embeds.view(len(sentence), 1, -1), self.hidden
        
        # get the scores for the most likely tag for a word
        outputs = self.fc( lstm_out )

        return outputs
        
    def sample(self, inputs, states=None, max_len=20):

        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        output_sentence = []
        batch = inputs.shape[0]
        hidden = self.init_hidden(batch)
        for i in range(max_len):
            lstm_outputs, hidden = self.lstm(inputs,hidden)
            outputs = self.fc(lstm_outputs)
            outputs = outputs.squeeze(1)
            _ ,max_pick = torch.max(outputs,dim=1)
            output_sentence.append(max_pick.cpu().numpy()[0].item())
            inputs = self.word_embeddings(max_pick).unsqueeze(1)

        return output_sentence




    


