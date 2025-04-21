import torch
import torch.nn as nn
import torch.nn.functional as F
from models.pos_encoding import PositionEmbedding
from models.style_transformer import DecoderLayer

class MultiModalTransformer(nn.Module):
    def __init__(self, 
                 num_vq=1024,
                 embed_dim=512,
                 latent_dim=512,
                 block_size=16,
                 num_layers=4,
                 n_head=8,
                 drop_out_rate=0.1,
                 fc_rate=4,
                 style_module="scaled"):  # scaled 或 attn
        super().__init__()
        
        self.token_emb = nn.Embedding(num_vq + 2, embed_dim)
        self.pos_embed = PositionEmbedding(block_size + 1, embed_dim, dropout=0.0)

        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model=embed_dim,
                         d_inner=embed_dim * fc_rate,
                         d_latent=latent_dim,
                         n_head=n_head,
                         d_k=embed_dim // n_head,
                         d_v=embed_dim // n_head,
                         dropout=drop_out_rate,
                         style_module=style_module)
            for _ in range(num_layers)
        ])
        
        self.ln_f = nn.LayerNorm(embed_dim)
        self.output_head = nn.Linear(embed_dim, num_vq + 1)  # 分类到 motion token 空间
        self.block_size = block_size

    def forward(self, idxs, cond_feat):
        """
        idxs: motion token ids (B, T)
        cond_feat: clip_feature (B, latent_dim)
        """
        B, T = idxs.shape
        assert T <= self.block_size, "Input sequence too long."

        token_emb = self.token_emb(idxs)  # (B, T, D)
        token_emb = self.pos_embed(token_emb)

        x = token_emb
        for layer in self.decoder_layers:
            x, _, _ = layer(x, enc_output=x, latent=cond_feat)

        x = self.ln_f(x)
        logits = self.output_head(x)
        return logits


class CausalCrossConditionalSelfAttention(nn.Module):

    def __init__(self, embed_dim=512, block_size=16, n_head=8, drop_out_rate=0.1):
        super().__init__()
        assert embed_dim % 8 == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(embed_dim, embed_dim)
        self.query = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

        self.attn_drop = nn.Dropout(drop_out_rate)
        self.resid_drop = nn.Dropout(drop_out_rate)

        self.proj = nn.Linear(embed_dim, embed_dim)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size))
        self.n_head = n_head

    def forward(self, x):
        B, T, C = x.size() 

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y

class Block(nn.Module):

    def __init__(self, embed_dim=512, block_size=16, n_head=8, drop_out_rate=0.1, fc_rate=4):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.attn = CausalCrossConditionalSelfAttention(embed_dim, block_size, n_head, drop_out_rate)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, fc_rate * embed_dim),
            nn.GELU(),
            nn.Linear(fc_rate * embed_dim, embed_dim),
            nn.Dropout(drop_out_rate),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class CrossCondTransBase(nn.Module):

    def __init__(self, 
                num_vq=1024, 
                embed_dim=512, 
                clip_dim=512, 
                block_size=16, 
                num_layers=2, 
                n_head=8, 
                drop_out_rate=0.1, 
                fc_rate=4):
        super().__init__()
        self.tok_emb = nn.Embedding(num_vq + 2, embed_dim)
        self.cond_emb = nn.Linear(clip_dim, embed_dim)
        self.pos_embedding = nn.Embedding(block_size, embed_dim)
        self.drop = nn.Dropout(drop_out_rate)
        # transformer block
        self.blocks = nn.Sequential(*[Block(embed_dim, block_size, n_head, drop_out_rate, fc_rate) for _ in range(num_layers)])
        self.pos_embed = pos_encoding.PositionEmbedding(block_size, embed_dim, 0.0, False)

        self.block_size = block_size

        self.apply(self._init_weights)

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def forward(self, idx, clip_feature):
        # print(clip_feature.shape)
        if len(idx) == 0:
            token_embeddings = self.cond_emb(clip_feature).unsqueeze(1)
            print("1111")
        else:
            b, t = idx.size()
            # print(f"idx: {idx.shape}")
            # idx: torch.Size([128, 50])
            assert t <= self.block_size, "Cannot forward, model block size is exhausted."
            # forward the Trans model
            token_embeddings = self.tok_emb(idx)
            
            # token_embeddings = torch.cat([self.cond_emb(clip_feature).unsqueeze(1), token_embeddings], dim=1)
            cond = self.cond_emb(clip_feature)
            if cond.dim() == 2:  # (32, 512)
                cond = cond.unsqueeze(1)  # → (32, 1, 512)
            elif cond.dim() == 3 and cond.shape[1] == 1:
                pass  # already (32, 1, 512)
            else:
                raise ValueError(f"Unexpected cond_emb output shape: {cond.shape}")

            token_embeddings = torch.cat([cond, token_embeddings], dim=1)


        # print(f"token_embeddings: {token_embeddings.shape}")
        # print()
            
        x = self.pos_embed(token_embeddings)
        x = self.blocks(x)

        return x


class CrossCondTransHead(nn.Module):

    def __init__(self, 
                num_vq=1024, 
                embed_dim=512, 
                block_size=16, 
                num_layers=2, 
                n_head=8, 
                drop_out_rate=0.1, 
                fc_rate=4):
        super().__init__()

        self.blocks = nn.Sequential(*[Block(embed_dim, block_size, n_head, drop_out_rate, fc_rate) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_vq + 1, bias=False)
        self.block_size = block_size

        self.apply(self._init_weights)

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x):
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits

    


        

