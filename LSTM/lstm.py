import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt

class LSTM(nn.Module):
    """
    A reimplementation of the LSTM model with appromixately the same specifications
    as given in the PyTorch documentation here: https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html

    Only the paper and the PyTorch documentation were used as references.
    """
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, batch_first=False, dropout=0, bidirectional=False, proj_size=0):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.proj_size = proj_size

        # Initalize the weights and biases
        for layer in range(num_layers):
            if layer == 0:
                layer_input_size = input_size
            elif proj_size > 0:
                layer_input_size = proj_size * (2 if bidirectional else 1)
            else:
                layer_input_size = hidden_size * (2 if bidirectional else 1)
            layer_hidden_size = proj_size if proj_size > 0 else hidden_size

            self.__setattr__('W_ih_l{}'.format(layer), nn.Parameter(torch.Tensor(4 * hidden_size, layer_input_size)))
            self.__setattr__('W_hh_l{}'.format(layer), nn.Parameter(torch.Tensor(4 * hidden_size, layer_hidden_size)))
            if bias:
                self.__setattr__('b_ih_l{}'.format(layer), nn.Parameter(torch.Tensor(4 * hidden_size)))
                self.__setattr__('b_hh_l{}'.format(layer), nn.Parameter(torch.Tensor(4 * hidden_size)))
            if proj_size > 0:
                self.__setattr__('W_hr_l{}'.format(layer), nn.Parameter(torch.Tensor(proj_size, hidden_size)))
        
        # TODO: Handle bidirectional case
        # TODO: Handle the rest of proj_size
        self.init_params()
    
    def init_params(self):
        bound = sqrt(1 / self.hidden_size)
        for param in self.parameters():
            nn.init.uniform_(param, -bound, bound)
    
    def forward(self, x, h_0=None, c_0=None):
        is_batched = len(x.shape) == 3
        if not is_batched:
            x = x.unsqueeze(int(not self.batch_first))
        if self.batch_first:
            x = x.transpose(0, 1)
        # Now, x is of shape (seq_len, batch, input_size)

        if h_0 is None:
            h_0 = torch.zeros(self.num_layers, x.shape[1], self.hidden_size, device=x.device)
        if c_0 is None:
            c_0 = torch.zeros(self.num_layers, x.shape[1], self.hidden_size, device=x.device)
        # h_0 and c_0 are of shape (num_layers, batch, hidden_size)
        h = h_0
        c = c_0
        # output is of shape (seq_len, batch, hidden_size)
        output = torch.empty(*x.shape[:2], self.proj_size if self.proj_size > 0 else self.hidden_size, device=x.device)
        for layer in range(self.num_layers):
            W_ih = self.__getattr__('W_ih_l{}'.format(layer))
            W_hh = self.__getattr__('W_hh_l{}'.format(layer))
            b_ih = self.__getattr__('b_ih_l{}'.format(layer))
            b_hh = self.__getattr__('b_hh_l{}'.format(layer))
            if self.proj_size > 0:
                W_hr = self.__getattr__('W_hr_l{}'.format(layer))
            
            for t in range(x.shape[0]):
                x_t = x[t]
                if layer > 0 and self.proj_size > 0:
                    x_t = torch.matmul(h[layer - 1], W_hr)
                elif layer > 0:
                    x_t = h[layer - 1]
                # Apply dropout
                if layer > 0 and self.dropout > 0 and self.training:
                    x_t = F.dropout(x_t, p=self.dropout, training=True)
                
                gates = F.linear(x_t, W_ih, b_ih) + F.linear(h[layer].clone(), W_hh, b_hh)
                i, f, g, o = gates.chunk(4, 1)
                i = torch.sigmoid(i)
                f = torch.sigmoid(f)
                g = torch.tanh(g)
                o = torch.sigmoid(o)
                c[layer] = f * c[layer].clone() + i * g
                h[layer] = o * torch.tanh(c[layer])
                output[t] = torch.matmul(h[layer], W_hr) if self.proj_size > 0 else h[layer]

        if self.batch_first:
            output = output.transpose(0, 1)

        return output, (h, c)

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Test the LSTM model
    input_size = 20
    hidden_size = 10
    batch_size = 5
    seq_len = 3

    kwargs = {
        'input_size': input_size,
        'hidden_size': hidden_size,
        'num_layers': 3,
        'batch_first': True,
        'proj_size': 0,
        'dropout': 0.5
    }

    model = LSTM(**kwargs).to(device)
    reference = nn.LSTM(**kwargs).to(device)

    # Train both models to output 1's
    x = torch.ones(batch_size, seq_len, input_size, device=device)
    y = torch.ones(batch_size, seq_len, hidden_size, device=device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    reference_optimizer = torch.optim.Adam(reference.parameters(), lr=0.01)
    for epoch in range(1000):
        optimizer.zero_grad()
        reference_optimizer.zero_grad()
        output, _ = model(x)
        reference_output, _ = reference(x)
        loss = criterion(output, y)
        reference_loss = criterion(reference_output, y)
        loss.backward()
        reference_loss.backward()
        optimizer.step()
        reference_optimizer.step()
        if epoch % 100 == 0:
            print('Epoch: {}, Loss: {}, Reference Loss: {}'.format(epoch, loss.item(), reference_loss.item()))

