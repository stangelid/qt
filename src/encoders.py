import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda

class TransformerDocumentEncoder(nn.Module):
    '''Transformer-based encoder that hierarchically encodes a whole document
       (e.g., a review) producing sentence-level multi-head representations.
    '''

    def __init__(self, d_model=320, sentence_nlayers=3, sentence_internal_nheads=4,
                 sentence_d_ff=512, sentence_output_nheads=8, dropout=0.1):
        '''Parameters:
            d_model (int): Transformer dimensionality for sentence encoder
            sentence_nlayers (int): number of Transformer layers for sentence encoder
            sentence_internal_nheads (int): number of attention heads for sentence encoder
            sentence_d_ff (int): dimensionality of ff layer for sentence encoder
            sentence_output_nheads (int): number of heads for sentence encoder
            dropout (float): dropout probability
        '''
        super(TransformerDocumentEncoder, self).__init__()
        assert d_model % sentence_output_nheads == 0, \
                'Number of sentence output heads must divide model dimensionality'
        self.d_model = d_model
        self.sentence_nlayers = sentence_nlayers
        self.sentence_internal_nheads = sentence_internal_nheads
        self.sentence_d_ff = sentence_d_ff
        self.sentence_output_nheads = sentence_output_nheads
        self.dropout = dropout

        transformer_layer = nn.TransformerEncoderLayer(d_model, sentence_internal_nheads,
                dim_feedforward=sentence_d_ff, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(transformer_layer,
                num_layers=sentence_nlayers, norm=nn.LayerNorm(d_model))

        if sentence_output_nheads > 1:
            self.final_linear = nn.Linear(d_model // sentence_output_nheads, d_model)
            self.final_lnorm = nn.LayerNorm(d_model)

    def forward(self, inputs, padding_mask=None):
        '''Parameters:
            inputs [B x S x T x E]: a batch of documents
            padding_mask [B x S x T]: source key padding mask
        '''
        assert inputs.dim() == 4, \
                'Inputs must have 4 dimensions: [B x S x T x E]'

        batch_size, nsent, ntokens, emb_size = inputs.size()
        inputs = inputs.reshape(batch_size * nsent, ntokens, -1)

        if padding_mask is not None:
            padding_mask = padding_mask.reshape(batch_size * nsent, ntokens)

        inputs = inputs.transpose(0, 1)
        token_vecs = self.transformer_encoder(inputs, src_key_padding_mask=padding_mask)
        token_vecs = token_vecs.transpose(0, 1).reshape(batch_size, nsent, -1, self.d_model)

        sentence_vecs = \
                token_vecs[:,:,0,:].reshape(batch_size, nsent, self.sentence_output_nheads, -1)
        if self.sentence_output_nheads > 1:
            sentence_vecs = self.final_lnorm(self.final_linear(sentence_vecs))

        return sentence_vecs


class PositionalEncoding(nn.Module):
    '''Sinusoidal positional encoding for non-recurrent neural networks.
    Implementation based on "Attention Is All You Need"
    '''

    def __init__(self, dropout, dim, max_len=50):
        '''Parameters:
            dropout (float): dropout parameter
            dim (int): embedding size
        '''
        if dim % 2 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                             "odd dim (got dim={:d})".format(dim))
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.float) *
                             -(math.log(10000.0) / dim)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(1)
        super(PositionalEncoding, self).__init__()
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(p=dropout)
        self.dim = dim

    def forward(self, emb, step=None):
        '''Parameters:
            emb (FloatTensor): Sequence of word vectors
                ``(seq_len, batch_size, self.dim)``
            step (int or NoneType): If stepwise (``seq_len = 1``), use
                the encoding for this position.
        '''

        emb = emb * math.sqrt(self.dim)
        if step is None:
            emb = emb + self.pe[:emb.size(0)]
        else:
            emb = emb + self.pe[step]
        emb = self.dropout(emb)
        return emb

class PositionalDocumentEncoding(PositionalEncoding): 
    def __init__(self, dim, dropout=0.1, max_len=5000):
        super(PositionalDocumentEncoding, self).__init__(dropout, dim, max_len)
    
    def forward(self, emb):
        assert emb.dim() == 4, \
                'Inputs must be 4-dimensional'
        assert emb.size(3) == self.dim, \
                'Embedding size of input doesn\'t match pos. encoding size'
        batch_size, nsent, ntokens, emb_size = emb.size()

        emb = emb.reshape(batch_size * nsent, ntokens, emb_size).transpose(0, 1)
        emb = super(PositionalDocumentEncoding, self).forward(emb)
        return emb.transpose(0, 1).reshape(batch_size, nsent, ntokens, emb_size)
