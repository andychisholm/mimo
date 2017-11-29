import torch
import torch.nn as nn
import numpy as np

from mimo.model.components.modules import BottleLinear as Linear
from mimo.model.components.layers import EncoderLayer, DecoderLayer
from mimo.model.components import PAD


def position_encoding_init(n_position, d_pos_vec):
    """ sinusoid position encoding table """

    # keep dim 0 for padding token position encoding zero vector
    position_enc = np.array([
        [pos / np.power(10000, 2 * (j // 2) / d_pos_vec) for j in range(d_pos_vec)]
        if pos != 0 else np.zeros(d_pos_vec) for pos in range(n_position)])

    position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2]) # dim 2i
    position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2]) # dim 2i+1
    return torch.from_numpy(position_enc).type(torch.FloatTensor)


def get_attn_padding_mask(seq_q, seq_k):
    """ Indicate the padding-related part to mask """
    assert seq_q.dim() == 2 and seq_k.dim() == 2
    mb_size, len_q = seq_q.size()
    mb_size, len_k = seq_k.size()
    pad_attn_mask = seq_k.data.eq(PAD).unsqueeze(1)
    pad_attn_mask = pad_attn_mask.expand(mb_size, len_q, len_k)
    return pad_attn_mask


def get_attn_subsequent_mask(seq):
    assert seq.dim() == 2
    attn_shape = (seq.size(0), seq.size(1), seq.size(1))
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    subsequent_mask = torch.from_numpy(subsequent_mask)
    if seq.is_cuda:
        subsequent_mask = subsequent_mask.cuda()
    return subsequent_mask


class Encoder(nn.Module):
    def __init__(self, n_src_vocab, n_max_seq, n_layers=6, n_head=8, d_k=64, d_v=64,
                 d_word_vec=512, d_model=512, d_inner_hid=1024, dropout=0.1):
        super(Encoder, self).__init__()

        n_position = n_max_seq + 1
        self.n_max_seq = n_max_seq
        self.d_model = d_model

        self.position_enc = nn.Embedding(n_position, d_word_vec, padding_idx=PAD)
        self.position_enc.weight.data = position_encoding_init(n_position, d_word_vec)

        self.src_word_emb = nn.Embedding(n_src_vocab, d_word_vec, padding_idx=PAD)

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner_hid, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

    def forward(self, src_seq, src_pos, return_attns=False):
        enc_input = self.src_word_emb(src_seq)
        enc_input += self.position_enc(src_pos)
        if return_attns:
            enc_slf_attns = []

        enc_output = enc_input
        enc_slf_attn_mask = get_attn_padding_mask(src_seq, src_seq)
        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output, slf_attn_mask=enc_slf_attn_mask)
            if return_attns:
                enc_slf_attns += [enc_slf_attn]

        if return_attns:
            return enc_output, enc_slf_attns
        else:
            return enc_output,


class Decoder(nn.Module):
    def __init__(
            self, n_tgt_vocab, n_max_seq, n_layers=6, n_head=8, d_k=64, d_v=64,
            d_word_vec=512, d_model=512, d_inner_hid=1024, dropout=0.1):
        super(Decoder, self).__init__()
        n_position = n_max_seq + 1
        self.n_max_seq = n_max_seq
        self.d_model = d_model

        self.position_enc = nn.Embedding(n_position, d_word_vec, padding_idx=PAD)
        self.position_enc.weight.data = position_encoding_init(n_position, d_word_vec)

        self.tgt_word_emb = nn.Embedding(n_tgt_vocab, d_word_vec, padding_idx=PAD)
        self.dropout = nn.Dropout(dropout)

        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model, d_inner_hid, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)
        ])

    def forward(self, tgt_seq, tgt_pos, src_seq, enc_output, return_attns=False):
        dec_input = self.tgt_word_emb(tgt_seq)
        dec_input += self.position_enc(tgt_pos)

        dec_slf_attn_pad_mask = get_attn_padding_mask(tgt_seq, tgt_seq)
        dec_slf_attn_sub_mask = get_attn_subsequent_mask(tgt_seq)
        dec_slf_attn_mask = torch.gt(dec_slf_attn_pad_mask + dec_slf_attn_sub_mask, 0)

        dec_enc_attn_pad_mask = get_attn_padding_mask(tgt_seq, src_seq)

        if return_attns:
            dec_slf_attns, dec_enc_attns = [], []

        dec_output = dec_input
        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(
                dec_output, enc_output,
                slf_attn_mask=dec_slf_attn_mask,
                dec_enc_attn_mask=dec_enc_attn_pad_mask)

            if return_attns:
                dec_slf_attns += [dec_slf_attn]
                dec_enc_attns += [dec_enc_attn]

        if return_attns:
            return dec_output, dec_slf_attns, dec_enc_attns
        else:
            return dec_output,


class Transformer(nn.Module):
    """ A sequence to sequence model with attention mechanism. """

    def __init__(
            self, n_src_vocab, n_tgt_vocab, n_max_src_seq, n_max_tgt_seq, n_layers=6, n_head=8,
            d_word_vec=512, d_model=512, d_inner_hid=1024, d_k=64, d_v=64,
            dropout=0.1, proj_share_weight=True, embs_share_weight=True):

        super(Transformer, self).__init__()
        self.encoder = Encoder(
            n_src_vocab, n_max_src_seq, n_layers=n_layers, n_head=n_head,
            d_word_vec=d_word_vec, d_model=d_model,
            d_inner_hid=d_inner_hid, dropout=dropout)
        self.decoder = Decoder(
            n_tgt_vocab, n_max_tgt_seq, n_layers=n_layers, n_head=n_head,
            d_word_vec=d_word_vec, d_model=d_model,
            d_inner_hid=d_inner_hid, dropout=dropout)
        self.tgt_word_proj = Linear(d_model, n_tgt_vocab, bias=False)
        self.dropout = nn.Dropout(dropout)

        assert d_model == d_word_vec, "Module output dimensions must be the same for residual connections"

        if proj_share_weight:
            # Share the weight matrix between tgt word embedding/projection
            assert d_model == d_word_vec
            self.tgt_word_proj.weight = self.decoder.tgt_word_emb.weight

        if embs_share_weight:
            # Share the weight matrix between src/tgt word embeddings
            # assume the src/tgt word vec size are the same
            assert n_src_vocab == n_tgt_vocab, "To share a word embedding table, src/tgt vocabs must be the same size"
            self.encoder.src_word_emb.weight = self.decoder.tgt_word_emb.weight

    def get_trainable_parameters(self):
        """ Avoid updating the position encoding """
        enc_freezed_param_ids = set(map(id, self.encoder.position_enc.parameters()))
        dec_freezed_param_ids = set(map(id, self.decoder.position_enc.parameters()))
        freezed_param_ids = enc_freezed_param_ids | dec_freezed_param_ids
        return (p for p in self.parameters() if id(p) not in freezed_param_ids)

    def forward(self, src, tgt):
        src_seq, src_pos = src
        tgt_seq, tgt_pos = tgt

        tgt_seq = tgt_seq[:, :-1]
        tgt_pos = tgt_pos[:, :-1]

        enc_output, *_ = self.encoder(src_seq, src_pos)
        dec_output, *_ = self.decoder(tgt_seq, tgt_pos, src_seq, enc_output)
        seq_logit = self.tgt_word_proj(dec_output)

        return seq_logit.view(-1, seq_logit.size(2))


class MimoTransformer(nn.Module):
    def __init__(self, n_src_vocab, n_max_src_seq, decoder_params, n_layers=6, n_head=8,
            d_word_vec=512, d_model=512, d_inner_hid=1024, dropout=0.1,
            proj_share_weight=True, embs_share_weight=True):
        super().__init__()

        self.encoder = Encoder(
            n_src_vocab, n_max_src_seq, n_layers=n_layers, n_head=n_head,
            d_word_vec=d_word_vec, d_model=d_model,
            d_inner_hid=d_inner_hid, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

        assert d_model == d_word_vec, "Module output dimensions must be the same for residual connections"

        self.decoders = {}
        for name, params in decoder_params.items():
            assert params['d_model'] == params['d_word_vec'], "Module outputs must be same for residuals"
            print("Building decoder: ", name)
            print(params)

            decoder = Decoder(
                params['n_tgt_vocab'], params['n_max_tgt_seq'],
                n_layers=params['n_layers'], n_head=params['n_head'],
                d_word_vec=params['d_word_vec'], d_model=params['d_model'],
                d_inner_hid=params['d_inner_hid'], dropout=dropout)
            decoder.tgt_word_proj = Linear(params['d_model'], params['n_tgt_vocab'], bias=False)

            if proj_share_weight:
                assert params['d_model'] == params['d_word_vec']
                decoder.tgt_word_proj.weight = decoder.tgt_word_emb.weight
            if embs_share_weight:
                assert n_src_vocab == params['n_tgt_vocab'], "To share a word embedding table, src/tgt vocabs must be the same size"
                decoder.tgt_word_emb.weight = self.encoder.src_word_emb.weight

            self.add_module(name, decoder)
            self.decoders[name] = {
                'name': name,
                'params': params,
                'model': decoder
            }

    def get_trainable_parameters(self):
        """ Avoid updating the position encoding """
        enc_freezed_param_ids = set(map(id, self.encoder.position_enc.parameters()))

        dec_freezed_param_ids = set()
        for d in self.decoders.values():
            dec_freezed_param_ids |= set(map(id, d['model'].position_enc.parameters()))

        freezed_param_ids = enc_freezed_param_ids | dec_freezed_param_ids
        return (p for p in self.parameters() if id(p) not in freezed_param_ids)

    def forward(self, src, tgts):
        src_seq, src_pos = src
        enc_output, *_ = self.encoder(src_seq, src_pos)

        dec_outputs = {}
        for name, tgt in tgts.items():
            tgt_seq, tgt_pos = tgt
            tgt_seq = tgt_seq[:, :-1]
            tgt_pos = tgt_pos[:, :-1]

            decoder = self.decoders[name]['model']
            dec_output, *_ = decoder(tgt_seq, tgt_pos, src_seq, enc_output)
            seq_logit = decoder.tgt_word_proj(dec_output)
            dec_outputs[name] = seq_logit.view(-1, seq_logit.size(2))

        return dec_outputs
