import timm
import torch
import torch.nn as nn

# class ViTPatchEncoder(nn.Module):
#     def __init__(self, out_dim=256, model_name='vit_base_patch16_224'):
#         super().__init__()
#         self.vit = timm.create_model(model_name, pretrained=True)
#         self.vit.head = nn.Identity() # Remove classification head if present
#         self.proj = nn.Linear(self.vit.embed_dim, out_dim)
#         for p in self.vit.parameters():
#             p.requires_grad = True

#     def forward(self, x): # x: (B, 1, H, W) or (B, 3, H, W)
#         if x.shape[1] == 1:
#             x = x.repeat(1, 3, 1, 1)

#         if x.shape[2] != 224 or x.shape[3] != 224:
#             x = nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        
#         # timm ViT returns patch tokens by default (no [CLS] unless you ask for it)
#         patch_tokens = self.vit.forward_features(x) # (B, num_patches, embed_dim)
#         patch_tokens = self.proj(patch_tokens)
#         return patch_tokens # (B, num_patches, out_dim)


class SwinPatchEncoder(nn.Module):
    def __init__(self, out_dim=256, model_name='swin_base_patch4_window7_224'):
        super().__init__()
        self.swin = timm.create_model(model_name, pretrained=True, features_only=True)
        num_chs = self.swin.feature_info[-1]['num_chs']
        self.proj = nn.Linear(num_chs, out_dim)
        for p in self.swin.parameters():
            p.requires_grad = True

    def forward(self, x):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        features = self.swin(x)[-1]  # [B, 7, 7, 1024]
        B, H, W, C = features.shape
        patch_tokens = features.reshape(B, H * W, C)  # [B, 49, 1024]
        patch_tokens = self.proj(patch_tokens)         # [B, 49, out_dim]
        return patch_tokens

# class CNNPatchEncoder(nn.Module):
#     def __init__(self, out_dim=256, model_name='resnet34', pretrained=True):
#         super().__init__()
#         # Create the CNN backbone, remove the classifier head
#         self.cnn = timm.create_model(model_name, pretrained=pretrained, features_only=True)
#         # Get the number of output channels from the last feature map
#         num_chs = self.cnn.feature_info[-1]['num_chs']
#         self.proj = nn.Linear(num_chs, out_dim)
#         for p in self.cnn.parameters():
#             p.requires_grad = False

#     def forward(self, x):
#         # x: (B, 1, H, W) or (B, 3, H, W)
#         if x.shape[1] == 1:
#             x = x.repeat(1, 3, 1, 1)
#         # Get the last feature map (B, C, H', W')
#         features = self.cnn(x)[-1]
#         B, C, H, W = features.shape
#         # Flatten spatial dimensions to patches
#         patch_tokens = features.permute(0, 2, 3, 1).reshape(B, H*W, C)
#         patch_tokens = self.proj(patch_tokens)
#         return patch_tokens  # (B, num_patches, out_dim)

class AttentionPool(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Tanh(),
            nn.Linear(input_dim, 1)
        )

    def forward(self, x, mask=None):
        attn_weights = self.attn(x).squeeze(-1)  # [B, N]
        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask == 0, float('-inf'))
        attn_weights = torch.softmax(attn_weights, dim=-1)
        return torch.sum(attn_weights.unsqueeze(-1) * x, dim=1)  # [B, D]

class LatentCrossAttn(nn.Module):
    def __init__(self, latent_dim=256, num_latents=128, num_layers=6):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(1, num_latents, latent_dim))
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=latent_dim, nhead=8, batch_first=True)
            for _ in range(num_layers)
        ])
        self.cross_attn = nn.MultiheadAttention(latent_dim, 8, batch_first=True)
        self.ln = nn.LayerNorm(latent_dim)

    def forward(self, patch_embeds):
        B = patch_embeds.size(0)
        z = self.latents.expand(B, -1, -1)
        for layer in self.layers:
            z = z + self.cross_attn(z, patch_embeds, patch_embeds)[0]  # Residual add
            z = layer(z)
        z = self.ln(z)
        return z

class CrossModQuery(nn.Module):
    def __init__(self, model_dim=256, num_layers=3):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.ModuleDict({
                "cross_attn": nn.MultiheadAttention(model_dim, 8, batch_first=True),
                "self_attn": nn.TransformerEncoderLayer(d_model=model_dim, nhead=8, batch_first=True)
            }))

    def forward(self, stroke_embeds, z, stroke_mask=None):
        for layer in self.layers:
            stroke_embeds, _ = layer["cross_attn"](stroke_embeds, z, z, key_padding_mask=None)
            stroke_embeds = layer["self_attn"](stroke_embeds, src_key_padding_mask=~stroke_mask if stroke_mask is not None else None)
        return stroke_embeds

class StrokeEmbedder(nn.Module):
    def __init__(self, model_dim):
        super().__init__()
        # self.proj = nn.Linear(3, model_dim)
        pen_dim = model_dim // 8
        self.pen_state_emb = nn.Embedding(2, model_dim // 8)  # 0: up, 1: down
        self.input_dim = 2 + pen_dim  # x, y + pen embedding
        self.proj = nn.Sequential(
            nn.Linear(self.input_dim, model_dim),
            nn.BatchNorm1d(model_dim),
            nn.Dropout(p=0.1)
        )
        self.ln = nn.LayerNorm(model_dim)

    def forward(self, strokes):
        B, T, _ = strokes.shape

        pen_state = strokes[..., 2].long()
        pen_embed = self.pen_state_emb(pen_state)  # (B, T, pen_dim)

        x = torch.cat([strokes[..., :2], pen_embed], dim=-1)  # (B, T, 2 + pen_dim)
        x = self.proj(x.view(-1, self.input_dim)).view(B, T, -1)  # Apply projection
        x = self.ln(x)
        return x

class StrokeEncoder(nn.Module):
    def __init__(self, model_dim=256, num_layers=2):
        super().__init__()
        self.embedder = StrokeEmbedder(model_dim)
        self.rotary_pe = RotaryEmbedding(model_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=model_dim, nhead=8, batch_first=True),
            num_layers=num_layers
        )
        self.post_layer = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, model_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(model_dim * 2, model_dim)
        )

    def forward(self, strokes, mask=None):
        x = self.embedder(strokes)
        x = self.rotary_pe(x)
        src_key_padding_mask = None
        if mask is not None:
            src_key_padding_mask = ~mask  # Convert to False for valid tokens
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        x = self.post_layer(x)
        return x

class RotaryEmbedding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        assert d_model % 2 == 0
        self.d_model = d_model
        inv_freq = 1.0 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x):
        # x: (B, T, D)
        seq_len = x.size(1)
        device = x.device
        positions = torch.arange(seq_len, device=device).type_as(self.inv_freq)
        sinusoid_in = torch.einsum("i,j->ij", positions, self.inv_freq)
        sin = torch.sin(sinusoid_in)[None, :, :]  # (1, T, D/2)
        cos = torch.cos(sinusoid_in)[None, :, :]  # (1, T, D/2)
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        x_rot = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
        # Interleave the even and odd features
        x_out = torch.empty_like(x)
        x_out[..., ::2] = x_rot[..., :self.d_model // 2]
        x_out[..., 1::2] = x_rot[..., self.d_model // 2:]
        return x_out


class HATCharClassifier(nn.Module):
    def __init__(self, patch_dim=256, stroke_dim=3, model_dim=256, vocab_size=58, input_mode="both"):
        super().__init__()
        self.input_mode = input_mode
        self.use_image = input_mode in ["image", "both"]
        self.use_stroke = input_mode in ["stroke", "both"]

        if self.use_image:
            # self.patch_encoder = ViTPatchEncoder(out_dim=patch_dim)
            self.patch_encoder = SwinPatchEncoder(out_dim=patch_dim)
            self.latent_cross_attn = LatentCrossAttn(latent_dim=model_dim, num_latents=64, num_layers=1)

        if self.use_stroke:
            self.stroke_encoder = StrokeEncoder(model_dim=model_dim, num_layers=3)

        if self.use_image and self.use_stroke:
            self.crossmod_query = CrossModQuery(model_dim=model_dim, num_layers=1)

        self.pool = AttentionPool(model_dim)
        self.fc = nn.Linear(model_dim, vocab_size)


    def forward(self, images=None, strokes=None, stroke_mask=None):
        outputs = {}
        if self.input_mode == "image":
            patch_embeds = self.patch_encoder(images)
            z = self.latent_cross_attn(patch_embeds)

            pooled = self.pool(z)
            logits = self.fc(pooled)
            return logits, pooled

        elif self.input_mode == "stroke":
            stroke_mask = (strokes.sum(-1) != 0)  # assumes zero padding
            stroke_embeds = self.stroke_encoder(strokes, mask=stroke_mask)
            pooled = self.pool(stroke_embeds, mask=stroke_mask)
            # stroke_embeds = self.stroke_encoder(strokes)
            # pooled = stroke_embeds.mean(dim=1)
            logits = self.fc(pooled)
            return logits, pooled

        elif self.input_mode == "both":
            patch_embeds = self.patch_encoder(images)
            z = self.latent_cross_attn(patch_embeds)

            stroke_mask = (strokes.sum(-1) != 0)
            stroke_embeds = self.stroke_encoder(strokes, mask=stroke_mask)
            stroke_context = self.crossmod_query(stroke_embeds, z, stroke_mask=stroke_mask)
            pooled = self.pool(stroke_context, mask=stroke_mask)
            logits = self.fc(pooled)
            return logits, pooled



# def generate_square_subsequent_mask(sz, device):
#     # Returns (sz, sz) mask with True in upper triangle (future positions)
#     return torch.triu(torch.ones(sz, sz, device=device), diagonal=1).bool()

# class GPTDecoder(nn.Module):
#     def __init__(self, model_dim=256, vocab_size=58, num_layers=2):
#         super().__init__()
#         self.token_emb = nn.Embedding(vocab_size, model_dim)
#         self.layers = nn.ModuleList([
#             nn.TransformerDecoderLayer(d_model=model_dim, nhead=8, batch_first=True)
#             for _ in range(num_layers)
#         ])
#         self.fc_out = nn.Linear(model_dim, vocab_size)

#     def forward(self, tgt_seq, memory):
#         # tgt_seq: (B, T)
#         x = self.token_emb(tgt_seq)
#         T = x.size(1)
#         mask = generate_square_subsequent_mask(T, x.device)  # (T, T)
#         for layer in self.layers:
#             x = layer(x, memory, tgt_mask=mask)
#         logits = self.fc_out(x)
#         return logits

# class HATModel(nn.Module):
#     def __init__(self, patch_dim=256, stroke_dim=3, model_dim=256, vocab_size=58):
#         super().__init__()
#         # self.patch_encoder = FrozenCNNPatchEncoder(out_dim=patch_dim)
#         self.patch_encoder = FrozenViTPatchEncoder(out_dim=patch_dim)
#         # self.patch_encoder = FrozenViTRelPosPatchEncoder(out_dim=patch_dim)
#         self.latent_cross_attn = LatentCrossAttn(latent_dim=model_dim, num_latents=128, num_layers=6)
#         self.stroke_encoder = StrokeEncoder(model_dim=model_dim, num_layers=2)
#         self.crossmod_query = CrossModQuery(model_dim=model_dim, num_layers=2)
#         self.decoder = GPTDecoder(model_dim=model_dim, vocab_size=vocab_size, num_layers=2)
#         # CTC head: projects stroke_embeds to vocab_size (for CTC loss)
#         self.ctc_head = nn.Linear(model_dim, vocab_size)

#     def forward(self, images, strokes, tgt_seq):
#         assert images.dim() == 4, "images should be (B, C, H, W)"
#         assert strokes.dim() == 3, "strokes should be (B, T, 3)"
#         assert tgt_seq.dim() == 2, "tgt_seq should be (B, T)"
#         patch_embeds = self.patch_encoder(images)
#         z = self.latent_cross_attn(patch_embeds)
#         stroke_embeds = self.stroke_encoder(strokes)
#         # CTC logits BEFORE cross-modal query
#         ctc_logits = self.ctc_head(stroke_embeds)  # (B, T, vocab_size)
#         stroke_context = self.crossmod_query(stroke_embeds, z)
#         logits = self.decoder(tgt_seq, stroke_context)
#         return logits, ctc_logits, z, stroke_context

# if __name__ == "__main__":
#     model = HATCharClassifier()
#     images = torch.randn(1, 1, 224, 224)  # Example batch, grayscale
#     strokes = torch.randn(1, 100, 3)      # Example batch
#     # tgt_seq = torch.randint(0, 58, (2, 50))  # Example batch
#     logits, stroke_context = model(images, strokes)
#     print(logits.shape)  # (B, T, vocab_size)