import torch
import torch.nn as nn
import numpy as np


class Embedding(nn.Module):
    """Embedding Module"""
    
    def __init__(self, embedding_dim, vocab_size, init_weight=None):
        super(Embedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
    
    def forward(self, x):
        """
            x: (batch_size, timesteps)
        """
        return self.embedding(x)


class Encoder(nn.Module):
    """Encoder Module"""
    def __init__(self, embedding_dim, hidden_dim):
        super(Encoder, self).__init__()
        self.encoder_network = nn.LSTM(embedding_dim, hidden_dim, 2, batch_first=True, bidirectional=False)
    
    def forward(self, x):
        """
            x: (batch_size, timesteps, embedding_dim)
        """
        return self.encoder_network(x)


class Decoder(nn.Module):
    """Decoder Module"""
    def __init__(self, embedding_dim, hidden_dim):
        super(Decoder, self).__init__()
        self.decoder_network = nn.LSTM(embedding_dim, hidden_dim, 2, batch_first=True, bidirectional=False)

    def forward(self, x, hidden):
        """
            wx: (batch_size, timesteps, embedding_dim)
            hidden: (hidden_state, cell_state)
        """
        return self.decoder_network(x, hidden)


class Projection(nn.Module):
    """Projection Module"""
    def __init__(self, hidden_dim, vocab_size):
        super(Projection, self).__init__()
        self.projection_layer = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        """
            x: (batch_size, timesteps, hidden_dim)
        """
        x = self.projection_layer(x)
        return nn.functional.log_softmax(x, dim=-1)


class Attention(nn.Module):
    """Attention Module"""
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
    
    def forward(self, query, encoder_outputs, x_lens):
        energi = torch.bmm(encoder_outputs, query.unsqueeze(2)).squeeze(2)
        
        max_len = torch.max(x_lens)
        mask = torch.arange(max_len).expand(len(x_lens), max_len) < x_lens.unsqueeze(1)
        energi[~mask] = float('-inf')
        
        attention = nn.functional.softmax(energi, dim=1)
        context = torch.bmm(attention.unsqueeze(1), encoder_outputs).squeeze(1)
        return context


class Seq2Seq(nn.Module):
    """ Simple Seq2Seq model """

    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(Seq2Seq, self).__init__()
        self.embedding = Embedding(embedding_dim, vocab_size)
        self.encoder = Encoder(embedding_dim, hidden_dim)
        self.decoder = Decoder(embedding_dim, hidden_dim)
        self.project = Projection(hidden_dim * 2, vocab_size)
        self.attention = Attention(hidden_dim)

    def forward(self, x, x_lens, y, teacher_force_prob=1.):
        """
            x: (batch_size, timesteps)
        """
        batch_size, num_timesteps = y.size()
        encoder_outputs, hidden = self.encoder_forward(x)

        outputs = []
        output = None
        for i in range(num_timesteps):
            if np.random.random() <= teacher_force_prob or i == 0:
                ip = y[:, i].long()
            else:
                ip = output.argmax(dim=-1).long()
            output, hidden = self.decoder_forward(ip, hidden, x_lens)
            outputs.append(output.unsqueeze(1))
        return torch.cat(outputs, dim=1)

    def encoder_forward(self, x):
        # Get embeds
        embedding = self.embedding(x)
        # run the encoder
        encoder_outputs, hidden = self.encoder(embedding)
        # Save encoder_outputs state
        self.encoder_outputs = encoder_outputs
        return encoder_outputs, hidden
    
    def decoder_forward(self, output_id, hidden, x_lens):
        # Get embeds
        ip = self.embedding(output_id).unsqueeze(1)
        # Run the decoder
        output, hidden = self.decoder(ip, hidden)
        output = output.squeeze(1)    # <-- batch_first
        # Apply attention
        context = self.attention(output, self.encoder_outputs, x_lens)
        output = torch.cat([output, context], dim=1)
        # Project to vocabulary space
        output = self.project(output)
        return output, hidden