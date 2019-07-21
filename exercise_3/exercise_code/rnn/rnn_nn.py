import torch
import torch.nn as nn
from torch import sigmoid, tanh


class RNN(nn.Module):
    def __init__(self, input_size=1, hidden_size=20, activation="tanh"):
        super(RNN, self).__init__()
        """
        Inputs:
        - input_size: Number of features in input vector
        - hidden_size: Dimension of hidden vector
        - activation: Nonlinearity in cell; 'tanh' or 'relu'
        """
        ############################################################################
        # TODO: Build a simple one layer RNN with an activation with the attributes#
        # defined above and a forward function below. Use the nn.Linear() function #
        # as your linear layers.                                                   #
        # Initialse h as 0 if these values are not given.                          #
        ############################################################################
        self.W = nn.Linear(hidden_size, hidden_size)
        self.V = nn.Linear(input_size, hidden_size)

        self.activation = nn.Tanh()
        self.hidden_size = hidden_size
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

    def forward(self, x, h=None):
        """
        Inputs:
        - x: Input tensor (seq_len, batch_size, input_size)
        - h: Optional hidden vector (nr_layers, batch_size, hidden_size)

        Outputs:
        - h_seq: Hidden vector along sequence (seq_len, batch_size, hidden_size)
        - h: Final hidden vetor of sequence(1, batch_size, hidden_size)
        """
        h_seq = []
        ############################################################################
        #                                YOUR CODE                                 #
        ############################################################################
        seq_len, batch_size, input_size = x.size()
        if h is None:
            h = torch.zeros(batch_size, self.hidden_size)

        for i in range(seq_len):
            z = self.V(x[i]) + self.W(h)
            h = self.activation(z)
            h_seq.append(h)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        return torch.stack(h_seq), h
    
    
class LSTM(nn.Module):
    def __init__(self, input_size=1 , hidden_size=20):
        super(LSTM, self).__init__()
    ############################################################################
    # TODO: Build a one layer LSTM with an activation with the attributes      #
    # defined above and a forward function below. Use the nn.Linear() function #
    # as your linear layers.                                                   #
    # Initialse h and c as 0 if these values are not given.                    #
    ############################################################################
        self.Wf = nn.Linear(input_size, hidden_size)
        self.Uf = nn.Linear(hidden_size, hidden_size)
        self.Wi = nn.Linear(input_size, hidden_size)
        self.Ui = nn.Linear(hidden_size, hidden_size)
        self.Wo = nn.Linear(input_size, hidden_size)
        self.Uo = nn.Linear(hidden_size, hidden_size)
        self.Wc = nn.Linear(input_size, hidden_size)
        self.Uc = nn.Linear(hidden_size, hidden_size)
        self.hidden_size = hidden_size


    def forward(self, x, h=None , c=None):
        """
        Inputs:
        - x: Input tensor (seq_len, batch_size, input_size)
        - h: Hidden vector (nr_layers, batch_size, hidden_size)
        - c: Cell state vector (nr_layers, batch_size, hidden_size)

        Outputs:
        - h_seq: Hidden vector along sequence (seq_len, batch_size, hidden_size)
        - h: Final hidden vetor of sequence(1, batch_size, hidden_size)
        - c: Final cell state vetor of sequence(1, batch_size, hidden_size)
        """

        seq_len, batch_size, input_size = x.size()

        if h is None:
            h = torch.zeros(batch_size, self.hidden_size)

        if c is None:
            c = torch.zeros(batch_size, self.hidden_size)

        h_seq = []

        for t in range(seq_len):
            xt = x[t]
            ft = sigmoid(self.Wf(xt) + self.Uf(h))
            it = sigmoid(self.Wi(xt) + self.Ui(h))
            ot = sigmoid(self.Wo(xt) + self.Uo(h))
            c = ft * c + it * tanh(self.Wc(xt) + self.Uc(h))
            h = ot * tanh(c)
            h_seq.append(h)

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
        return torch.stack(h_seq), (h, c)
    

class RNN_Classifier(torch.nn.Module):
    def __init__(self,classes=10, input_size=28 , hidden_size=128, activation="relu" ):
        super(RNN_Classifier, self).__init__()
    ############################################################################
    #  TODO: Build a RNN classifier                                            #
    ############################################################################
        pass
       
    def forward(self, x):
        pass

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)


class LSTM_Classifier(torch.nn.Module):
    def __init__(self,classes=10, input_size=28, hidden_size=128):
        super(LSTM_Classifier, self).__init__()
    ############################################################################
    #  TODO: Build a LSTM classifier                                           #
    ############################################################################
        self.LSTM = LSTM(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, classes)

    def forward(self, x):
        h_seq, last_state = self.LSTM(x)
        h, c = last_state
        x = self.fc(h)
        return x

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)
