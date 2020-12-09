import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.multinomial import Multinomial
import numpy as np

from encoders import *

# code based on: https://github.com/zalandoresearch/pytorch-vq-vae/blob/master/vq-vae.ipynb
class SequenceQuantizer(nn.Module):
    def __init__(self, codebook_size, d_model, commitment_cost, padding_idx=None):
        super(SequenceQuantizer, self).__init__()
        
        self.d_model = d_model
        self.codebook_size = codebook_size
        self.padding_idx = padding_idx
        
        self.codebook = nn.Embedding(self.codebook_size, self.d_model, padding_idx=padding_idx)
        if padding_idx is not None:
            self.codebook.weight.data[padding_idx] = 0

        self.commitment_cost = commitment_cost

    def forward(self, inputs, padding_mask=None, commitment_cost=None):
        device = inputs.device

        if commitment_cost is None:
            commitment_cost = self.commitment_cost

        # inputs can be:
        # 2-dimensional [B x E]         (already flattened)
        # 3-dimensional [B x T x E]     (e.g., batch of sentences)
        # 4-dimensional [B x S x T x E] (e.g., batch of documents)
        input_shape = inputs.size()
        input_dims = inputs.dim()

        # Flatten input
        flat_input = inputs.reshape(-1, self.d_model)
        
        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self.codebook.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self.codebook.weight.t()))
        
        # TODO: generalize this to any padding_idx
        if padding_mask is not None:
            # no normal token gets mapped to discrete value 0
            distances[:, 0][~padding_mask.reshape(-1)] = np.inf
            # all pad tokens get mapped to discrete value 0
            distances[:, 1:][padding_mask.reshape(-1, 1).expand(-1, self.codebook_size - 1)] = np.inf

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.codebook_size).to(device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize and unflatten
        quantized = torch.matmul(encodings, self.codebook.weight).view(input_shape)
        
        # Loss
        if padding_mask is not None:
            num_nonpad_elements = torch.sum(~padding_mask) * self.d_model
            e_latent_loss = torch.sum((quantized.detach() - inputs)**2) / num_nonpad_elements
            q_latent_loss = torch.sum((quantized - inputs.detach())**2) / num_nonpad_elements
        else:
            e_latent_loss = torch.mean((quantized.detach() - inputs)**2)
            q_latent_loss = torch.mean((quantized - inputs.detach())**2)
        loss = q_latent_loss + commitment_cost * e_latent_loss
        
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        return quantized, encoding_indices.reshape(input_shape[:input_dims - 1]), loss, perplexity

    def set_codebook(self, new_codebook):
        self.codebook.weight.copy_(new_codebook)


class SequenceQuantizerEMA(nn.Module):
    def __init__(self, codebook_size, d_model, commitment_cost, decay=0.99, epsilon=1e-5, padding_idx=None):
        super(SequenceQuantizerEMA, self).__init__()
        
        self.d_model = d_model
        self.codebook_size = codebook_size
        self.padding_idx = padding_idx
        
        self.codebook = nn.Embedding(self.codebook_size, self.d_model, padding_idx=padding_idx)
        if padding_idx is not None:
            self.codebook.weight.data[padding_idx] = 0

        self.commitment_cost = commitment_cost
        
        self.register_buffer('_ema_cluster_size', torch.zeros(codebook_size))
        self._ema_w = nn.Parameter(torch.Tensor(codebook_size, self.d_model))
        self._ema_w.data.copy_(self.codebook.weight.data)
        
        self._decay = decay
        self._epsilon = epsilon
        self.discard_ema_cluster_sizes = False

    def forward(self, inputs, padding_mask=None, commitment_cost=None, temp=None):
        device = inputs.device

        if commitment_cost is None:
            commitment_cost = self.commitment_cost

        if temp is None:
            temp = self.temp

        # inputs can be:
        # 2-dimensional [B x E]         (already flattened)
        # 3-dimensional [B x T x E]     (e.g., batch of sentences)
        # 4-dimensional [B x S x T x E] (e.g., batch of documents)
        input_shape = inputs.size()
        input_dims = inputs.dim()

        # Flatten input
        flat_input = inputs.reshape(-1, self.d_model)
        
        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self.codebook.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self.codebook.weight.t()))

        # TODO: generalize this to any padding_idx
        if padding_mask is not None:
            # no normal token gets mapped to discrete value 0
            distances[:, 0][~padding_mask.reshape(-1)] = np.inf
            # all pad tokens get mapped to discrete value 0
            distances[:, 1:][padding_mask.reshape(-1, 1).expand(-1, self.codebook_size - 1)] = np.inf

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.codebook_size).to(device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize and unflatten
        quantized = torch.matmul(encodings, self.codebook.weight).view(input_shape)
        
        # Loss
        if padding_mask is not None:
            num_nonpad_elements = torch.sum(~padding_mask) * self.d_model
            e_latent_loss = torch.sum((quantized.detach() - inputs)**2) / num_nonpad_elements
        else:
            e_latent_loss = torch.mean((quantized.detach() - inputs)**2)
        loss = commitment_cost * e_latent_loss
        
        # Use EMA to update the embedding vectors
        if self.training:
            if self.discard_ema_cluster_sizes:
                self._ema_cluster_size = torch.sum(encodings, 0)
                self.discard_ema_cluster_sizes = False
            else:
                self._ema_cluster_size = self._ema_cluster_size * self._decay + \
                                         (1 - self._decay) * torch.sum(encodings, 0)
                
            # Laplace smoothing of the cluster size
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                (self._ema_cluster_size + self._epsilon)
                / (n + self.codebook_size * self._epsilon) * n)
            
            dw = torch.matmul(encodings.t(), flat_input)
            self._ema_w = nn.Parameter(self._ema_w * self._decay + (1 - self._decay) * dw)
            
            normalized_ema_w = self._ema_w / self._ema_cluster_size.unsqueeze(1)
            if self.padding_idx is not None:
                normalized_ema_w[self.padding_idx] = 0
            self.codebook.weight = nn.Parameter(normalized_ema_w)

        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        return quantized, encoding_indices.reshape(input_shape[:input_dims - 1]), loss, perplexity

    def set_codebook(self, new_codebook, discard_ema_cluster_sizes=False):
        self.codebook.weight.copy_(new_codebook)
        self._ema_w.copy_(new_codebook)
        self.discard_ema_cluster_sizes = discard_ema_cluster_sizes


class SequenceQuantizerSoftEMA(nn.Module):
    def __init__(self, codebook_size, d_model, commitment_cost, num_samples=10, temp=1.0,
            ema_decay=0.99, epsilon=1e-5, padding_idx=None):
        super(SequenceQuantizerSoftEMA, self).__init__()
        
        self.d_model = d_model
        self.codebook_size = codebook_size
        self.padding_idx = padding_idx
        
        self.codebook = nn.Embedding(self.codebook_size, self.d_model, padding_idx=padding_idx)
        if padding_idx is not None:
            self.codebook.weight.data[padding_idx] = 0

        self.commitment_cost = commitment_cost
        self.num_samples = num_samples
        self.temp = temp
        
        self.register_buffer('_ema_cluster_size', torch.zeros(codebook_size))
        self._ema_w = nn.Parameter(torch.Tensor(codebook_size, self.d_model))
        self._ema_w.data.copy_(self.codebook.weight.data)
        
        self._decay = ema_decay
        self._epsilon = epsilon
        self.discard_ema_cluster_sizes = False

    def forward(self, inputs, padding_mask=None, commitment_cost=None, temp=None):
        device = inputs.device

        if commitment_cost is None:
            commitment_cost = self.commitment_cost

        if temp is None:
            temp = self.temp

        # inputs can be:
        # 2-dimensional [B x E]         (already flattened)
        # 3-dimensional [B x T x E]     (e.g., batch of sentences)
        # 4-dimensional [B x S x T x E] (e.g., batch of documents)
        input_shape = inputs.size()
        input_dims = inputs.dim()

        # Flatten input
        flat_input = inputs.reshape(-1, self.d_model)

        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True)
                    + torch.sum(self.codebook.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self.codebook.weight.t()))
            
        # TODO: generalize this to any padding_idx
        if padding_mask is not None:
            # no normal token gets mapped to discrete value 0
            distances[:, 0][~padding_mask.reshape(-1)] = np.inf
            # all pad tokens get mapped to discrete value 0
            distances[:, 1:][padding_mask.reshape(-1, 1).expand(-1, self.codebook_size - 1)] = np.inf

        # Define multinomial distribution and sample from it
        multi = Multinomial(total_count=self.num_samples, logits=-distances / temp)
        samples = multi.sample().to(device)

        # Soft-quantize and unflatten
        quantized = torch.matmul(samples, self.codebook.weight).view(input_shape) / self.num_samples
       
        # Loss
        if padding_mask is not None:
            num_nonpad_elements = torch.sum(~padding_mask) * self.d_model
            e_latent_loss = torch.sum((quantized.detach() - inputs)**2) / num_nonpad_elements
        else:
            e_latent_loss = torch.mean((quantized.detach() - inputs)**2)
        loss = commitment_cost * e_latent_loss

        # Use EMA to update the embedding vectors
        if self.training:
            if self.discard_ema_cluster_sizes:
                self._ema_cluster_size = torch.sum(samples, 0) / self.num_samples
                self.discard_ema_cluster_sizes = False
            else:
                self._ema_cluster_size = self._ema_cluster_size * self._decay + \
                                         (1 - self._decay) * \
                                         (torch.sum(samples, 0) / self.num_samples)

            # Laplace smoothing of the cluster size
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                (self._ema_cluster_size + self._epsilon)
                / (n + self.codebook_size * self._epsilon) * n)

            dw = torch.matmul(samples.t(), flat_input) / self.num_samples
            self._ema_w = nn.Parameter(self._ema_w * self._decay + (1 - self._decay) * dw)

            normalized_ema_w = self._ema_w / self._ema_cluster_size.unsqueeze(1)
            if self.padding_idx is not None:
                normalized_ema_w[self.padding_idx] = 0
            self.codebook.weight = nn.Parameter(normalized_ema_w)

        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(samples, dim=0) / self.num_samples
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        samples = samples.reshape(list(input_shape[:input_dims - 1]) + [self.codebook_size])

        return quantized, samples, loss, perplexity

    def cluster(self, inputs, padding_mask=None):
        device = inputs.device

        # inputs can be:
        # 2-dimensional [B x E]         (already flattened)
        # 3-dimensional [B x T x E]     (e.g., batch of sentences)
        # 4-dimensional [B x S x T x E] (e.g., batch of documents)
        input_shape = inputs.size()
        input_dims = inputs.dim()

        # Flatten input
        flat_input = inputs.reshape(-1, self.d_model)

        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True)
                    + torch.sum(self.codebook.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self.codebook.weight.t()))

        # TODO: generalize this to any padding_idx
        if padding_mask is not None:
            # no normal token gets mapped to discrete value 0
            distances[:, 0][~padding_mask.reshape(-1)] = np.inf
            # all pad tokens get mapped to discrete value 0
            distances[:, 1:][padding_mask.reshape(-1, 1).expand(-1, self.codebook_size - 1)] = np.inf

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.codebook_size).to(device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self.codebook.weight).view(input_shape)

        encoding_indices = encoding_indices.reshape(input_shape[:input_dims - 1])
        distances = distances.reshape(list(input_shape[:-1]) + [self.codebook_size]).detach()

        return quantized, encoding_indices, distances

    def set_codebook(self, new_codebook, discard_ema_cluster_sizes=False):
        self.codebook.weight.copy_(new_codebook)
        self._ema_w.copy_(new_codebook)
        self.discard_ema_cluster_sizes = discard_ema_cluster_sizes


class TransformerDocumentQuantizer(SequenceQuantizer):
    def __init__(self, codebook_size=64, d_model=200, commitment_cost=0.25,
            nlayers=3, internal_nheads=4, output_nheads=4, d_ff=512,
            pooling_final_linear=True, output_resize=True, dropout=0.1):
        if output_resize:
            self.d_output = d_model
            self.pooling_final_linear = True
            super(TransformerDocumentQuantizer, self).__init__(
                    codebook_size, d_model, commitment_cost)
        else:
            assert d_model % output_nheads == 0, 'Number of output heads must divide d_model'
            self.d_output = d_model // output_nheads
            self.pooling_final_linear = pooling_final_linear
            super(TransformerDocumentQuantizer, self).__init__(
                    codebook_size, self.d_output, commitment_cost)
        self.nlayers = nlayers
        self.internal_nheads = internal_nheads
        self.output_nheads = output_nheads
        self.d_ff = d_ff
        self.dropout = dropout

        self.encoder = TransformerDocumentEncoder(
                            d_model=d_model,
                            sentence_nlayers=nlayers,
                            sentence_internal_nheads=internal_nheads,
                            sentence_output_nheads=output_nheads,
                            sentence_d_ff=d_ff,
                            use_single_pooling_norm=True,
                            pooling_final_linear=self.pooling_final_linear,
                            pooling_final_linear_size=self.d_output,
                            pooling_same_linear=True,
                            dropout=dropout)

    def forward(self, inputs, padding_mask=None,
            quantize=True, residual_coeff=0.0, commitment_cost=None):
        assert inputs.dim() == 4, \
                'Inputs must have 4 dimensions: [B x S x T x E]'

        out = self.encoder(inputs, padding_mask)
        if quantize:
            if residual_coeff > 0.0:
                quantized, encodings, loss, perplexity = \
                        super(TransformerSentenceQuantizer, self).forward(out,
                                commitment_cost=commitment_cost)
                quantized = residual_coeff * out + (1 - residual_coeff) * quantized
                return quantized, encodings, loss, perplexity
            else:
                return super(TransformerDocumentQuantizer, self).forward(out,
                        commitment_cost=commitment_cost)
        return out, None, 0.0, None


class TransformerDocumentQuantizerEMA(SequenceQuantizerEMA):
    def __init__(self, codebook_size=64, d_model=200,
            commitment_cost=0.25, decay=0.99, epsilon=1e-5,
            nlayers=3, internal_nheads=4, output_nheads=4, d_ff=512,
            dropout=0.1):
        assert d_model % output_nheads == 0, 'Number of output heads must divide d_model'
        super(TransformerDocumentQuantizerEMA, self).__init__(codebook_size, d_model,
                commitment_cost=commitment_cost, decay=decay, epsilon=epsilon)
        self.nlayers = nlayers
        self.internal_nheads = internal_nheads
        self.output_nheads = output_nheads
        self.d_ff = d_ff
        self.dropout = dropout

        self.encoder = TransformerDocumentEncoder(
                            d_model=d_model,
                            sentence_nlayers=nlayers,
                            sentence_internal_nheads=internal_nheads,
                            sentence_output_nheads=output_nheads,
                            sentence_d_ff=d_ff,
                            dropout=dropout)

    def forward(self, inputs, padding_mask=None,
            quantize=True, residual_coeff=0.0, commitment_cost=None):
        assert inputs.dim() == 4, \
                'Inputs must have 4 dimensions: [B x S x T x E]'

        out = self.encoder(inputs, padding_mask)
        if quantize:
            if residual_coeff > 0.0:
                quantized, encodings, loss, perplexity = \
                        super(TransformerDocumentQuantizerEMA, self).forward(out,
                                commitment_cost=commitment_cost)
                quantized = residual_coeff * out + (1 - residual_coeff) * quantized
                return quantized, encodings, loss, perplexity
            else:
                return super(TransformerDocumentQuantizerEMA, self).forward(out,
                            commitment_cost=commitment_cost)
        return out, None, 0.0, None
    
    
class TransformerDocumentQuantizerSoftEMA(SequenceQuantizerSoftEMA):
    def __init__(self, codebook_size=64, d_model=200, temp=1.0, num_samples=10,
            commitment_cost=0.25, ema_decay=0.99, epsilon=1e-5,
            nlayers=3, internal_nheads=4, output_nheads=4, d_ff=512,
            dropout=0.1):
        assert d_model % output_nheads == 0, 'Number of output heads must divide d_model'
        super(TransformerDocumentQuantizerSoftEMA, self).__init__(codebook_size, d_model,
                commitment_cost=commitment_cost, temp=temp, num_samples=num_samples,
                ema_decay=ema_decay, epsilon=epsilon)
        self.nlayers = nlayers
        self.internal_nheads = internal_nheads
        self.output_nheads = output_nheads
        self.d_ff = d_ff
        self.dropout = dropout

        self.encoder = TransformerDocumentEncoder(
                            d_model=d_model,
                            sentence_nlayers=nlayers,
                            sentence_internal_nheads=internal_nheads,
                            sentence_output_nheads=output_nheads,
                            sentence_d_ff=d_ff,
                            dropout=dropout)

    def forward(self, inputs, padding_mask=None,
            quantize=True, residual_coeff=0.0, commitment_cost=None,
            temp=None):
        assert inputs.dim() == 4, \
                'Inputs must have 4 dimensions: [B x S x T x E]'

        out = self.encoder(inputs, padding_mask)
        if quantize:
            if residual_coeff > 0.0:
                quantized, encodings, loss, perplexity = \
                        super(TransformerDocumentQuantizerSoftEMA, self).forward(out,
                                commitment_cost=commitment_cost, temp=temp)
                quantized = residual_coeff * out + (1 - residual_coeff) * quantized
                return quantized, encodings, loss, perplexity
            else:
                return super(TransformerDocumentQuantizerSoftEMA, self).forward(out,
                        commitment_cost=commitment_cost, temp=temp)
        return out, None, 0.0, None

    def cluster(self, inputs, padding_mask):
        assert inputs.dim() == 4, \
                'Inputs must have 4 dimensions: [B x S x T x E]'

        out = self.encoder(inputs, padding_mask)
        q_out, clusters, distances = \
                super(TransformerDocumentQuantizerSoftEMA, self).cluster(out)

        return out, q_out, clusters, distances
