from typing import Optional

import math
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as f
from torch.nn.modules.transformer import _get_clones

from mmcv.cnn.bricks.drop import build_dropout
from mmcv.runner.base_module import BaseModule
from mmcv.cnn import ConvModule, build_activation_layer
from mmcv.cnn.bricks.transformer import build_positional_encoding, POSITIONAL_ENCODING

from mmdet.models.utils import build_linear_layer
from mmdet.models.utils.builder import TRANSFORMER

from seqtr.core.layers.attention import MultiheadAttention


class LinearModule(nn.Module):
    """A linear block that bundles linear/activation/dropout layers.

    This block simplifies the usage of linear layers, which are commonly
    used with an activation layer (e.g., ReLU) and Dropout layer (e.g., Dropout).
    It is based upon three build methods: `build_linear_layer()`,
    `build_activation_layer()` and `build_dropout`.

    Args:
        linear (dict): Config dict for activation layer. Default: dict(type='Linear', bias=True)
        act (dict): Config dict for activation layer. Default: dict(type='ReLU', inplace=True).
        drop (dict): Config dict for dropout layer. Default: dict(type='Dropout', drop_prob=0.5)
    """

    def __init__(self,
                 linear=dict(type='Linear', bias=True),
                 act=dict(type='ReLU', inplace=True),
                 drop=dict(type='Dropout', drop_prob=0.5)):
        super(LinearModule, self).__init__()
        assert linear is None or isinstance(linear, dict)
        assert act is None or isinstance(act, dict)
        assert drop is None or isinstance(drop, dict)
        assert 'in_features' in linear and 'out_features' in linear

        self.with_activation = act is not None
        self.with_drop = drop is not None

        self.fc = build_linear_layer(linear)

        if self.with_activation:
            self.activate = build_activation_layer(act)

        if self.with_drop:
            self.drop = build_dropout(drop)

    def forward(self, input):
        input = self.fc(input)

        if self.with_activation:
            input = self.activate(input)

        if self.with_drop:
            input = self.drop(input)

        return input


def with_pos_embed(tensor, pos: Optional[Tensor]):
    return tensor if pos is None else tensor + pos


@POSITIONAL_ENCODING.register_module()
class LearnedPositionalEncoding1D(nn.Module):
    """1D Position embedding with learnable embedding weights.

    Args:
        num_feature (int): The feature dimension for each position.
        num_embedding (int, optional): The dictionary size of embeddings.
            Default 5.
    """

    def __init__(self,
                 num_embedding=5,
                 num_feature=256,
                 padding_idx=-1,
                 ):
        super(LearnedPositionalEncoding1D, self).__init__()
        self.num_feature = num_feature

        self.num_embedding = num_embedding
        self.embedding = nn.Embedding(
            num_embedding, num_feature, padding_idx=padding_idx if padding_idx >= 0 else None)

    def forward(self, seq_in_embeds):
        """
        Args:
            seq_in_embeds (tensor): [bs, 5/num_ray*2+1, d_model].

        Returns:
            seq_in_pos_embeds (tensor): [bs, 5/num_ray*2+1, d_model].
        """
        seq_len = seq_in_embeds.size(1)
        position = torch.arange(seq_len, dtype=torch.long,
                                device=seq_in_embeds.device)
        position = position.unsqueeze(0).expand(seq_in_embeds.size()[:2])
        return self.embedding(position)


@POSITIONAL_ENCODING.register_module()
class SinePositionalEncoding2D(BaseModule):
    """Position encoding with sine and cosine functions.
    See `End-to-End Object Detection with Transformers
    <https://arxiv.org/pdf/2005.12872>`_ for details.
    Args:
        num_feats (int): The feature dimension for each position
            along x-axis or y-axis. Note the final returned dimension
            for each position is 2 times of this value.
        temperature (int, optional): The temperature used for scaling
            the position embedding. Defaults to 10000.
        normalize (bool, optional): Whether to normalize the position
            embedding. Defaults to False.
        scale (float, optional): A scale factor that scales the position
            embedding. The scale will be used only when `normalize` is True.
            Defaults to 2*pi.
        eps (float, optional): A value added to the denominator for
            numerical stability. Defaults to 1e-6.
        offset (float): offset add to embed when do the normalization.
            Defaults to 0.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """

    def __init__(self,
                 num_feature,
                 temperature=10000,
                 normalize=False,
                 scale=2 * math.pi,
                 eps=1e-6,
                 offset=0.,
                 init_cfg=None):
        super(SinePositionalEncoding2D, self).__init__(init_cfg)
        if normalize:
            assert isinstance(scale, (float, int)), 'when normalize is set,' \
                'scale should be provided and in float or int type, ' \
                f'found {type(scale)}'
        self.num_feature = num_feature
        self.temperature = temperature
        self.normalize = normalize
        self.scale = scale
        self.eps = eps
        self.offset = offset

    def forward(self, mask):
        """Forward function for `SinePositionalEncoding`.
        Args:
            mask (Tensor): ByteTensor mask. Non-zero values representing
                ignored positions, while zero values means valid positions
                for this image. Shape [bs, h, w].
        Returns:
            pos (Tensor): Returned position embedding with shape
                [bs, num_feats*2, h, w].
        """
        # For convenience of exporting to ONNX, it's required to convert
        # `masks` from bool to int.
        mask = mask.to(torch.int)
        not_mask = 1 - mask  # logical_not
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            y_embed = (y_embed + self.offset) / \
                      (y_embed[:, -1:, :] + self.eps) * self.scale
            x_embed = (x_embed + self.offset) / \
                      (x_embed[:, :, -1:] + self.eps) * self.scale
        dim_t = torch.arange(
            self.num_feature, dtype=torch.float32, device=mask.device)
        dim_t = self.temperature**(2 * (dim_t // 2) / self.num_feature)
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        # use `view` instead of `flatten` for dynamically exporting to ONNX
        B, H, W = mask.size()
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()),
            dim=4).view(B, H, W, -1)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()),
            dim=4).view(B, H, W, -1)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


class TransformerEncoderLayerWithPositionEmbedding(nn.TransformerEncoderLayer):
    def __init__(self, *args, **kwargs):
        super(TransformerEncoderLayerWithPositionEmbedding,
              self).__init__(*args, **kwargs)

    def forward(self,
                src: Tensor,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None) -> Tensor:
        q = k = with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class TransformerEncoderWithPositionEmbedding(nn.TransformerEncoder):
    def __init__(self, *args, **kwargs):
        super(TransformerEncoderWithPositionEmbedding,
              self).__init__(*args, **kwargs)

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src

        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoderLayerWithPositionEmbedding(nn.TransformerDecoderLayer):
    def __init__(self, *args, **kwargs):
        super(TransformerDecoderLayerWithPositionEmbedding,
              self).__init__(*args, **kwargs)
        self.d_model = 256
        self.nhead = 8
        self.multihead_attn = MultiheadAttention(self.d_model, self.nhead, dropout=0.1, batch_first=True)
        self.point = nn.Sequential(
            nn.Linear(self.d_model, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.nhead * 4),
        )
        nn.init.constant_(self.point[-1].weight.data, 0)
        nn.init.constant_(self.point[-1].bias.data, 0)

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h, valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio
    
    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                need_weights: bool = False,
                memory_h=None,
                memory_w=None,
                ):
        bs, seq_len, d_model = tgt.shape
        q = k = with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        
        points = self.point(tgt)
        points_scale = points[..., 2 * self.nhead:].reshape(bs, tgt.shape[1], self.nhead, 2).permute(1, 0, 2, 3)
        points = points[..., :2 * self.nhead].permute(1, 0, 2).sigmoid().reshape(seq_len, bs * self.nhead, 2)
        # points = points[..., :2 * self.nhead].view(bs * tgt.shape[1], 1, self.nhead, 2).tanh()
        
        grid_y, grid_x = torch.meshgrid(torch.arange(0, memory_h), torch.arange(0, memory_w))
        grid = torch.stack((grid_x, grid_y), 2).float().to(tgt.device)
        grid = grid.reshape(-1, 2).unsqueeze(1).repeat(1, bs * 8, 1) # 8 is the heads num.
        
        valid_ratio = self.get_valid_ratio(memory_key_padding_mask[:, :memory_h*memory_w].view(bs, memory_h, memory_w))
        
        # SMCA: start
        gau_point = torch.clone(points)
        gau_point[:, :, 0] *= memory_w
        gau_point[:, :, 1] *= memory_h
        gau_point = gau_point.reshape(seq_len, bs, self.nhead, 2)
        gau_point = gau_point * valid_ratio[:, :2].reshape(1, bs, 1, 2)
        gau_point = gau_point.reshape(seq_len, bs * self.nhead, 2)
        gau_distance = (gau_point.unsqueeze(1) - (grid + 0.5).unsqueeze(0)).pow(2)
        gau_scale = points_scale
        gau_scale = gau_scale * gau_scale
        gau_scale = gau_scale.reshape(seq_len, -1, 2).unsqueeze(1)
        gau_distance = (gau_distance * gau_scale).sum(-1)
        gaussian = -(gau_distance - 0).abs() / 8.0         # 8.0 is the number used in SMCA-DETR
        
        
        if need_weights:
            tgt2, attn_weights = self.multihead_attn(query=with_pos_embed(tgt, query_pos),
                                                     key=with_pos_embed(
                                                         memory, pos),
                                                     value=memory, attn_mask=memory_mask,
                                                     key_padding_mask=memory_key_padding_mask,
                                                     need_weights=True)
        else:
            tgt2 = self.multihead_attn(query=with_pos_embed(tgt, query_pos),
                                       key=with_pos_embed(memory, pos),
                                       value=memory, attn_mask=memory_mask,
                                       key_padding_mask=memory_key_padding_mask,
                                       gaussian=[gaussian],
                                       )[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        if need_weights:
            return tgt, attn_weights
        else:
            return tgt


class TransformerDecoderWithPositionEmbedding(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm):
        super(TransformerDecoderWithPositionEmbedding, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        
    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                memory_h=None,
                memory_w=None,
                ):
        output = tgt

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos,
                           need_weights=False,
                           memory_h=memory_h,
                           memory_w=memory_w,
                           )

        output = self.norm(output)

        return output


@TRANSFORMER.register_module()
class AutoRegressiveTransformer(BaseModule):
    """Implements the Auto-regressive transformer.

    Args:
        encoder (`mmcv.ConfigDict` | Dict): Config of
            TransformerEncoder. Defaults to None.
        decoder ((`mmcv.ConfigDict` | Dict)): Config of
            TransformerDecoder. Defaults to None
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Defaults to None.
    """

    def __init__(self, encoder, decoder):
        super(AutoRegressiveTransformer, self).__init__(init_cfg=None)
        self.d_model = decoder['layer']['d_model']
        self.encoder = TransformerEncoderWithPositionEmbedding(
            TransformerEncoderLayerWithPositionEmbedding(
                **encoder.pop('layer')),
            **encoder)
        self.decoder = TransformerDecoderWithPositionEmbedding(
            TransformerDecoderLayerWithPositionEmbedding(
                **decoder.pop('layer')),
            **decoder,
            norm=nn.LayerNorm(self.d_model))

        self._reset_parameters()

    def _reset_parameters(self):
        for parameter in self.parameters():
            if parameter.dim() > 1:
                nn.init.xavier_uniform_(parameter)

    def _init_layers(self,
                     in_ch,
                     vocab_size,
                     x_positional_encoding,
                     seq_positional_encoding):

        self.x_positional_encoding = build_positional_encoding(
            x_positional_encoding)

        self.seq_positional_encoding = build_positional_encoding(
            seq_positional_encoding)

        self.query_embedding = nn.Embedding(vocab_size, self.d_model)
        self.task_prompt = nn.Embedding(1, self.d_model)
        # self.task_transfer = torch.nn.Linear(self.d_model, self.d_model)
        # self.seg_task_prompt = nn.Embedding(1, self.d_model)

        self.input_proj = ConvModule(
            in_channels=in_ch,
            out_channels=self.d_model,
            kernel_size=1,
            bias=True,
            act_cfg=None,
            norm_cfg=dict(type='GN', num_groups=32)
        )

    def tri_mask(self, length):
        mask = (torch.triu(torch.ones(length, length))
                == 1).float().transpose(0, 1)
        mask.masked_fill_(mask == 0, float('-inf'))
        mask.masked_fill_(mask == 1, float(0.))
        return mask

    def x_mask_pos_enc(self, x, img_metas):
        batch_size = x.size(0)

        input_img_h, input_img_w = img_metas[0]['batch_input_shape']
        x_mask = x.new_ones((batch_size, input_img_h, input_img_w))
        # CAUTION: do not support random flipping
        for img_id in range(batch_size):
            img_h, img_w, _ = img_metas[img_id]['img_shape']
            x_mask[img_id, :img_h, :img_w] = 0

        x_mask = f.interpolate(x_mask.unsqueeze(
            1), size=x.size()[-2:]).to(torch.bool).squeeze(1)

        x_pos_embeds = self.x_positional_encoding(x_mask)

        x_mask = x_mask.view(batch_size, -1)
        x_pos_embeds = x_pos_embeds.view(
            batch_size, self.d_model, -1).transpose(1, 2)

        return x_mask, x_pos_embeds

    def forward_encoder(self, x, x_mask, x_pos_embeds):
        """Args:
            x (Tensor): [batch_size, c, h, w].

            x_mask (tensor): [batch_size, h, w], dtype is torch.bool, True means
                ignored positions.

            x_pos_embeds (tensor): [batch_size, d_model, h, w]. 

        Returns:
            memory (tensor): encoder outputs, [batch_size, h*w, d_model].             
        """
        batch_size = x.size(0)
        # x = self.input_proj(x)
        x = self.input_proj(x.unsqueeze(-1))

        x = x.view(batch_size, self.d_model, -1).transpose(1, 2)  # x_shape [bs, 380, 256]
        memory = self.encoder(x,
                              src_key_padding_mask=x_mask,
                              pos=x_pos_embeds)

        return memory

    def forward_decoder(self,
                        seq_in_embeds,
                        memory,
                        x_pos_embeds,
                        x_mask,
                        memory_h=None,
                        memory_w=None,
                        ):
        seq_in_pos_embeds = self.seq_positional_encoding(seq_in_embeds)

        seq_in_mask = self.tri_mask(
            seq_in_embeds.size(1)).to(seq_in_embeds.device)

        tgt = self.decoder(seq_in_embeds, memory,
                           pos=x_pos_embeds,
                           query_pos=seq_in_pos_embeds,
                           memory_key_padding_mask=x_mask,
                           tgt_mask=seq_in_mask,
                           memory_h=memory_h,
                           memory_w=memory_w)
        return tgt
