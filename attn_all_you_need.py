import torch
import torch.nn as nn

"""
Implementation of Transformer (Encoder and Decoder) with self attention

"""


class SelfAttention(nn.Module):
    def __init__(self, embed_size=256, num_heads=8):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.head_dim = self.embed_size // self.num_heads

        # Head Dimension can only be integral
        assert (self.head_dim * self.num_heads == self.embed_size), \
            "Embedding size should be an integral multiple of number of heads"

        # Set K, V and Q
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)

        # FC layer
        self.fc = nn.Linear(self.num_heads * self.head_dim, self.embed_size)

    def forward(self, values, keys, queries, mask):
        # My batch size
        N = queries.shape[0]

        # All three lengths corresponds to source and target sentence length and hence will be variable
        value_len, key_len, query_len = values.shape[1], keys.shape[1], queries.shape[1]

        # Split embedding into self.num_heads pieces
        values = values.reshape(N, value_len, self.num_heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.num_heads, self.head_dim)
        queries = queries.reshape(N, query_len, self.num_heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        # Queries shape : N X query_len X num_heads X head_dim
        # Keys shape : N X key_len X num_heads X head_dim
        # energy shape : N X num_heads X query_len X key_len
        # use torch.bmm
        energy = torch.einsum("nqhd, nkhd->nhqk", [queries, keys])

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        # Calculate Attention and normalize on key length (dataset)
        attn = torch.softmax(energy / (self.embed_size ** 0.5), dim=3)
        out = torch.einsum("nhql, nlhd->nqhd", [attn, values])
        # attn shape : N X num_heads X query_len X key_len
        # Values shape: N X value_len X num_heads X head_dim
        # out shape : N X query_le X num_heads X head_dim
        # Remember key_len, and value_len are always same

        # Reshape to flatten num_heads and head_dim
        out = out.reshape(N, query_len, self.num_heads * self.head_dim)

        # Pass it to FC and return
        out = self.fc(out)

        return out


class TransformerBlock(nn.Module):
    def __init__(self, embed_size, num_heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, num_heads)

        # Two layer norms. Layer norm is diff from batch norm as it operates over one example
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        # FFN sandwich b/w both LNs
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        attn = self.attention(value, key, query, mask)

        # skip or residual connection
        x = self.dropout(self.norm1(attn + query))

        feed_fwd = self.feed_forward(x)
        out = self.dropout(self.norm2(x+feed_fwd))

        # Return the output
        return out


# My encoder class
class Encoder(nn.Module):
    def __init__(self,
                 source_vocab_size,
                 embed_size,
                 num_layers,
                 num_heads,
                 device,
                 forward_expansion,
                 dropout,
                 # Maximum length of my dataset text length of single data point
                 max_length
                 ):
        super(Encoder, self).__init__()
        # All my Hyper parameters
        self.device = device
        self.embed_size = embed_size
        self.word_embedding = nn.Embedding(source_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_size,
                    num_heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion
                )
                for _ in range(num_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        N, seq_len = x.shape
        positions = torch.arange(0, seq_len).expand(N, seq_len).to(self.device)

        out = self.dropout(self.word_embedding(x)+self.position_embedding(positions))

        for layer in self.layers:
            # In encoder value, key and query are all same
            out = layer(value=out, key=out, query=out, mask=mask)

        return out


class DecoderBlock(nn.Module):
    def __init__(self, embed_size, num_heads, forward_expansion, dropout, device):
        super(DecoderBlock, self).__init__()
        self.attention = SelfAttention(embed_size, num_heads)
        self.norm = nn.LayerNorm(embed_size)
        self.device = device

        self.transformer_block = TransformerBlock(embed_size,
                                                  num_heads,
                                                  dropout,
                                                  forward_expansion
                                                  )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, value, key, source_mask, target_mask):
        """
        source mask helps us to prevent unnecessary computation
        for those target mask values that are padded

        :param x:
        :param value:
        :param key:
        :param source_mask:
        :param target_mask:
        :return:

        """
        attn = self.attention(x, x, x, target_mask)
        query = self.dropout(self.norm(attn + x))
        out = self.transformer_block(value, key, query, source_mask)

        return out


class Decoder(nn.Module):
    def __init__(self,
                 target_vocab_size,
                 embed_size,
                 num_layers,
                 num_heads,
                 device,
                 forward_expansion,
                 dropout,
                 # Maximum length of my dataset text length of single data point
                 max_length
                 ):
        super(Decoder, self).__init__()
        # All my Hyper parameters
        self.device = device
        self.word_embedding = nn.Embedding(target_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList(
            [DecoderBlock(embed_size, num_heads, forward_expansion, dropout, device)
             for _ in range(num_layers)]
        )
        self.fc = nn.Linear(embed_size, target_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, source_mask, target_mask):
        N, seq_len = x.shape
        positions = torch.arange(0, seq_len).expand(N, seq_len).to(self.device)

        x = self.dropout(self.word_embedding(x)+self.position_embedding(positions))

        for layer in self.layers:
            x = layer(x, enc_out, enc_out, source_mask, target_mask)

        out = self.fc(x)
        return out


class Transformer(nn.Module):
    def __init__(self,
                 source_vocab_size,
                 target_vocab_size,
                 src_pad_idx,
                 trg_pad_idx,
                 embed_size=512,
                 num_layers=6,
                 forward_expansion=4,
                 num_heads=8,
                 dropout=0,
                 device="cpu",
                 max_length=100,
                 ):
        super(Transformer, self).__init__()

        self.encoder = Encoder(
            source_vocab_size,
            embed_size,
            num_layers,
            num_heads,
            device,
            forward_expansion,
            dropout,
            max_length
        )

        self.decoder = Decoder(
            target_vocab_size,
            embed_size,
            num_layers,
            num_heads,
            device,
            forward_expansion,
            dropout,
            max_length
        )

        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        # (N, 1, 1, src_len)
        return src_mask.to(self.device)

    def make_trg_mask(self, trg):
        N, trg_len = trg.shape
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(
            N, 1, trg_len, trg_len
        )

        return trg_mask.to(self.device)

    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        enc_src = self.encoder(src, src_mask)
        out = self.decoder(trg, enc_src, src_mask, trg_mask)
        return out


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # Random Tokens of some vocabulary
    x = torch.tensor([[1, 5, 6, 4, 3, 9, 5, 2, 0], [1, 8, 7, 3, 4, 5, 6, 7, 2]]).to(
        device
    )
    trg = torch.tensor([[1, 7, 4, 3, 5, 9, 2, 0], [1, 5, 6, 2, 4, 7, 6, 2]]).to(device)
    print(x.shape)
    print(trg.shape)

    src_pad_idx = 0
    trg_pad_idx = 0
    src_vocab_size = 10
    trg_vocab_size = 10
    model = Transformer(src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx, device=device).to(
        device
    )
    # We shifted it by one to left in the end of each target row so
    # that we can see how it predicts end of sentence token
    out = model(x, trg[:, :-1])
    print(out.shape)
