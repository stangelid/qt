import torch
import torch.nn as nn

from utils.data import *
from encoders import *
from quantizers import *


class QuantizedTransformerModel(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 d_model: int = 320,
                 codebook_size: int = 1024,
                 commitment_cost: float = 1.00,
                 ema_decay: float = 0.99,
                 temp: float = 1.0,
                 num_samples: int = 10,
                 epsilon: float = 1e-5,
                 nlayers: int = 3,
                 internal_nheads: int = 4,
                 output_nheads: int = 8, 
                 d_ff: int = 512,
                 dropout: float = 0.1,
                 use_in_pos: bool = False,
                 use_out_pos: bool = False,
                 padding_idx: int = 0,
                 unk_idx: int = 1,
                 bos_idx: int = 2,
                 eos_idx: int = 3):
        super(QuantizedTransformerModel, self).__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.codebook_size = codebook_size
        self.use_in_pos = use_in_pos
        self.use_out_pos = use_out_pos
        self.padding_idx = padding_idx
        self.unk_idx = unk_idx
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx

        self.in_emb = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        self.out_emb = self.in_emb
        if use_in_pos:
            self.in_pos = PositionalDocumentEncoding(d_model, dropout)
            if use_out_pos:
                self.out_pos = self.in_pos
        elif use_out_pos:
            self.out_pos = PositionalDocumentEncoding(d_model, dropout)

        self.encoder = TransformerDocumentQuantizerSoftEMA(
                            codebook_size=codebook_size,
                            d_model=d_model,
                            temp=temp,
                            num_samples=num_samples,
                            commitment_cost=commitment_cost,
                            ema_decay=ema_decay,
                            epsilon=epsilon,
                            nlayers=nlayers,
                            internal_nheads=internal_nheads,
                            output_nheads=output_nheads,
                            d_ff=d_ff,
                            dropout=dropout)

        decoder_layer = nn.TransformerDecoderLayer(
                                d_model,
                                internal_nheads,
                                dim_feedforward=d_ff,
                                dropout=dropout)
        self.decoder = nn.TransformerDecoder(
                                decoder_layer,
                                nlayers,
                                norm=nn.LayerNorm(d_model))

        self.linear = nn.Linear(self.d_model,
                                self.vocab_size)

    def in_embed(self, src):
        emb = self.in_emb(src)
        if self.use_in_pos:
            emb = self.in_pos(emb)
        return emb

    def out_embed(self, tgt):
        emb = self.out_emb(tgt)
        if self.use_out_pos:
            emb = self.out_pos(emb)
        return emb
    
    def encode(self, src, quantize=True, residual_coeff=0.0):
        assert src.dim() == 3, 'Input (source) must be 3-dimensional [B x S x T]'
        batch_size, nsent, ntokens = src.size()
        device = src.device

        sent_tokens = torch.ones(batch_size, nsent, 1).long() * self.unk_idx
        sent_tokens = sent_tokens.to(device)

        src = torch.cat([sent_tokens, src], dim=2)

        if self.padding_idx is not None:
            padding_mask = self.get_padding_mask(src)

        src_emb = self.in_embed(src)
        quantized_memory, encodings, q_loss, perplexity = \
                self.encoder(src_emb, padding_mask=padding_mask, \
                quantize=quantize, residual_coeff=residual_coeff)

        return quantized_memory, encodings, q_loss, perplexity
    
    def decode(self, tgt, memory, memory_mask=None,
            memory_key_padding_mask=None):
        assert tgt.dim() == 3, 'Input (target) must be 3-dimensional [B x S x T]'
        assert memory.dim() == 4, 'Input (memory) must be 4-dimensional [B x S x MT x E]'

        device = tgt.device
        batch_size, nsent, ntokens = tgt.size()
        mem_batch_size, mem_nsent, mem_ntokens, mem_emb_size = memory.size()

        assert batch_size == mem_batch_size \
                and nsent == mem_nsent \
                and mem_emb_size == self.d_model, \
                'Target, memory and model dimensionalities don\'t match'

        tgt_emb = self.out_embed(tgt).reshape(batch_size * nsent, ntokens, -1).transpose(0, 1)
        tgt = tgt.reshape(batch_size * nsent, ntokens)
        tgt_mask = generate_square_subsequent_mask(ntokens).to(device)
        if self.padding_idx is not None:
            tgt_key_padding_mask = self.get_padding_mask(tgt)

        memory = memory.reshape(mem_batch_size * mem_nsent, mem_ntokens, mem_emb_size).transpose(0, 1)
        if memory_key_padding_mask is not None:
            memory_key_padding_mask = \
                    memory_key_padding_mask.reshape(mem_batch_size * mem_nsent, mem_ntokens)

        output = self.decoder(
                    tgt_emb,
                    memory,
                    tgt_mask=tgt_mask,
                    memory_mask=memory_mask,
                    tgt_key_padding_mask=tgt_key_padding_mask,
                    memory_key_padding_mask=memory_key_padding_mask)
        return output.transpose(0, 1).reshape(batch_size, nsent, ntokens, -1)

    def forward(self, src, tgt, quantize=True, residual_coeff=0.0):
        src_batch_size, src_nsent, src_ntokens = src.size()
        tgt_batch_size, tgt_nsent, tgt_ntokens = tgt.size()
        assert src_batch_size == tgt_batch_size and src_nsent == tgt_nsent, \
                'Size mismath between source and target'

        memory, encodings, q_loss, perplexity = \
                self.encode(src, quantize=quantize, residual_coeff=residual_coeff)
        out = self.decode(tgt, memory)

        return self.linear(out), encodings, q_loss, perplexity

    def generate(self, src, maxlen=40, quantize=True, residual_coeff=0.0):
        assert src.dim() == 3, 'Input (source) must be 3-dimensional'
        batch_size, nsent, ntokens = src.size()
        device = src.device
        
        memory, encodings, q_loss, perplexity = \
                self.encode(src, quantize=quantize, residual_coeff=residual_coeff)
        
        # <BOS> tgt seq for generation
        tgt = torch.LongTensor(batch_size, nsent, maxlen).fill_(self.padding_idx).to(device)
        tgt[:,:,0] = self.bos_idx

        for i in range(1, maxlen):
            out = self.decode(tgt[:,:,:i], memory)
            prob = self.linear(out)
            decode_output = prob.argmax(dim=-1)
            tgt[:,:,i] = decode_output[:,:,-1]
        return tgt, encodings

    def cluster(self, src):
        assert src.dim() == 3, 'Input (source) must be 3-dimensional [B x S x T]'
        batch_size, nsent, ntokens = src.size()
        device = src.device

        sent_tokens = torch.ones(batch_size, nsent, 1).long() * self.unk_idx
        sent_tokens = sent_tokens.to(device)

        src = torch.cat([sent_tokens, src], dim=2)

        if self.padding_idx is not None:
            padding_mask = self.get_padding_mask(src)

        src_emb = self.in_embed(src)
        noq_out, q_out, clusters, distances = \
                self.encoder.cluster(src_emb, padding_mask=padding_mask)

        return noq_out, q_out, clusters, distances

    def get_padding_mask(self, batch):
        return batch == self.padding_idx

    def get_tgt_inputs(self, batch):
        batch_size, nsent, ntokens = batch.size()
        bos = torch.ones(batch_size, nsent, 1).long().to(device) * self.bos_idx

        return torch.cat([bos, batch], dim=2)


def generate_square_subsequent_mask(sz):
    """Generate a square mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).
    """
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask
