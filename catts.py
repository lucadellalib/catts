# ==============================================================================
# Copyright 2026 Luca Della Libera.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""CATTS: Character-Aligned Text-to-Speech with DyCAST Tokens."""

from typing import Any, List, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor, nn
from transformers import AutoModelForAudioXVector

__all__ = ["CATTS"]


try:
    nn.functional.scaled_dot_product_attention(
        *torch.empty(3, 1, 1, 1), enable_gqa=True
    )
    HAS_ENABLE_GQA = True
except Exception:
    HAS_ENABLE_GQA = False


class RMSNorm(nn.Module):
    """Root-mean-square normalization layer.

    This layer normalizes the input tensor along its feature dimension by its
    root-mean-square and scales the result by a learned weight parameter.

    Parameters
    ----------
    dim:
        Dimension of input/output features.
    norm_eps:
        Small constant added to the denominator for numerical stability.

    """

    def __init__(self, dim: "int" = 512, norm_eps: "float" = 1e-6) -> "None":
        super().__init__()
        self.dim = dim
        self.norm_eps = norm_eps

        # Parameters
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, input: "Tensor") -> "Tensor":
        """Forward pass.

        Parameters
        ----------
        input:
            Input tensor of shape (..., dim).

        Returns
        -------
            Output tensor of shape (..., dim).

        """
        input_type = input.type()
        output = input.float()
        output = output * ((output**2).mean(-1, keepdim=True) + self.norm_eps).rsqrt()
        output = output.type(input_type)
        return self.weight * output

    def __repr__(self) -> "str":
        return f"{self.__class__.__name__}(dim={self.dim}, norm_eps={self.norm_eps})"


class FeedForward(nn.Module):
    """Feed-forward neural network module.

    Parameters
    ----------
    dim:
        Dimension of input/output features.
    ffn_dim:
        Dimension of the hidden layer in the feed-forward network.
    dropout:
        Dropout probability applied after the last hidden layer.

    """

    def __init__(
        self,
        dim: "int" = 512,
        ffn_dim: "int" = 2048,
        dropout: "float" = 0.0,
    ) -> "None":
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.dropout_ = dropout

        # Modules
        self.gate_proj = nn.Linear(dim, ffn_dim, bias=False)
        self.activation = nn.SiLU()
        self.in_proj = nn.Linear(dim, ffn_dim, bias=False)
        self.out_proj = nn.Linear(ffn_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input: "Tensor") -> "Tensor":
        """Forward pass.

        Parameters
        ----------
        input:
            Input tensor of shape (..., dim).

        Returns
        -------
            Output tensor of shape (..., dim).

        """
        output = self.gate_proj(input)
        output = self.activation(output)
        output = output * self.in_proj(input)
        output = self.out_proj(output)
        output = self.dropout(output)
        return output


class GroupedQueryAttention(nn.Module):
    """Grouped-query attention layer.

    This layer performs a grouped multi-head attention mechanism, where the number
    of heads for queries can be different from the number of heads for keys and values.
    This approach reduces memory and computational requirements by sharing keys and
    values across multiple query heads.

    Parameters
    ----------
    dim:
        Dimension of input/output features.
    num_heads:
        Number of attention heads for the queries.
    num_kv_heads:
        Number of attention heads for the keys and values, which is
        typically smaller than `num_heads` to save on computation.
    dropout:
        Dropout probability for attention weights.

    """

    def __init__(
        self,
        dim: "int" = 512,
        num_heads: "int" = 16,
        num_kv_heads: "int" = 4,
        dropout: "float" = 0.0,
    ) -> "None":
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.dropout = dropout
        self.head_dim = dim // num_heads
        self.num_kv_head_reps = num_heads // num_kv_heads

        # Modules
        self.q_proj = nn.Linear(dim, num_heads * self.head_dim, bias=False)
        self.kv_proj = nn.Linear(dim, 2 * num_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(num_heads * self.head_dim, dim, bias=False)

    def forward(
        self,
        input: "Tensor",
        freqs_cis: "Tensor",
        mask: "Optional[Tensor]" = None,
    ) -> "Tensor":
        """Forward pass.

        This method applies rotary positional embeddings based on the provided `freqs_cis`,
        applies the grouped multi-head attention mechanism, and handles key-value caching.

        Parameters
        ----------
        input:
            Input tensor of shape (batch_size, seq_length, dim).
        freqs_cis:
            Precomputed rotary positional embeddings for the current input sequence.
        mask:
            Attention mask, shape (batch_size, ..., tgt_seq_length, src_seq_length). Two types of masks are supported:
            - a boolean mask where a value of True indicates that the element should take part in attention;
            - a float mask of the same type as query, key, value that is added to the attention score.

        Returns
        -------
            Output tensor of shape (batch_size, seq_length, dim).

        """
        B = input.shape[0]
        T = input.shape[1]

        qs = self.q_proj(input).reshape(B, T, self.num_heads, -1)
        kvs = self.kv_proj(input).reshape(B, T, -1, self.head_dim)
        ks, vs = kvs.chunk(2, dim=-2)

        qs, ks = self._apply_rotary_emb(qs, ks, freqs_cis)

        # Reshape for scaled_dot_product_attention
        qs = qs.movedim(-3, -2)  # [B, num_heads, T, head_dim]
        ks = ks.movedim(-3, -2)  # [B, num_kv_heads, seq_length, head_dim]
        vs = vs.movedim(-3, -2)  # [B, num_kv_heads, seq_length, head_dim]

        output = self._grouped_query_attention(qs, ks, vs, mask)

        output = (
            output.movedim(1, 2).contiguous().reshape(B, T, -1)
        )  # [B, seq_length, num_heads * head_dim]
        output = self.out_proj(output)  # [B, seq_length, dim]

        return output

    @torch.jit.export
    def _apply_rotary_emb(
        self, xq: "Tensor", xk: "Tensor", freqs_cis: "Tensor"
    ) -> "Tuple[Tensor, Tensor]":
        xq_ = torch.view_as_complex(xq.float().reshape(xq.shape[:-1] + (-1, 2)))
        xk_ = torch.view_as_complex(xk.float().reshape(xk.shape[:-1] + (-1, 2)))

        # Reshape for broadcast
        assert xq_.ndim > 1
        assert freqs_cis.shape == (xq_.shape[1], xq_.shape[-1])
        shape = [1] * len(xq_.shape)
        shape[1] = xq_.shape[1]
        shape[-1] = xq_.shape[-1]
        freqs_cis = freqs_cis.reshape(shape)

        xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(start_dim=3)
        xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(start_dim=3)
        return xq_out.type_as(xq), xk_out.type_as(xk)

    @torch.jit.export
    def _grouped_query_attention(
        self,
        query: "Tensor",
        key: "Tensor",
        value: "Tensor",
        attn_mask: "Optional[Tensor]" = None,
    ) -> "Tensor":
        return nn.functional.scaled_dot_product_attention(
            query,
            key.repeat_interleave(self.num_kv_head_reps, dim=-3),
            value.repeat_interleave(self.num_kv_head_reps, dim=-3),
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )


class GroupedQueryAttentionNative(GroupedQueryAttention):
    """See documentation of `GroupedQueryAttention`."""

    @torch.jit.export
    def _grouped_query_attention(
        self,
        query: "Tensor",
        key: "Tensor",
        value: "Tensor",
        attn_mask: "Optional[Tensor]" = None,
    ) -> "Tensor":
        return nn.functional.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            enable_gqa=True,
        )


class LlamaLayer(nn.Module):
    """Llama layer.

    This layer combines a grouped-query attention mechanism with a
    feed-forward network, both of which are normalized by RMSNorm layers.

    Parameters
    ----------
    dim:
        Dimension of input/output features.
    ffn_dim:
        Dimension of the hidden layer in the feed-forward network.
    num_heads:
        Number of attention heads for the queries in the grouped-query
        attention mechanism.
    num_kv_heads:
        Number of heads for the keys and values in the grouped-query
        attention mechanism, typically fewer than `num_heads`.
    dropout:
        Dropout probability applied in the attention and feed-forward layers.
    norm_eps:
        Small constant added to the RMS normalization denominator for
        numerical stability.

    """

    def __init__(
        self,
        dim: "int" = 512,
        ffn_dim: "int" = 2048,
        num_heads: "int" = 16,
        num_kv_heads: "int" = 4,
        dropout: "float" = 0.0,
        norm_eps: "float" = 1e-6,
    ) -> "None":
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.dropout = dropout
        self.norm_eps = norm_eps

        # Modules
        self.attention = (
            GroupedQueryAttentionNative if HAS_ENABLE_GQA else GroupedQueryAttention
        )(dim, num_heads, num_kv_heads, dropout)
        self.attention_norm = RMSNorm(dim, norm_eps)
        self.feed_forward = FeedForward(dim, ffn_dim, dropout)
        self.feed_forward_norm = RMSNorm(dim, norm_eps)

    def forward(
        self,
        input: "Tensor",
        freqs_cis: "Tensor",
        mask: "Optional[Tensor]" = None,
    ) -> "Tensor":
        """See documentation of `GroupedQueryAttention.forward`."""
        hidden = self.attention(
            self.attention_norm(input),
            freqs_cis,
            mask,
        )
        hidden += input
        output = self.feed_forward_norm(hidden)
        output = self.feed_forward(output)
        output += hidden
        return output


class LlamaEncoder(nn.Module):
    """Llama encoder.

    This class implements a multi-layer encoder with grouped-query attention,
    feed-forward layers, RMS normalization, and rotary positional embeddings.

    Parameters
    ----------
    num_layers:
        Number of Llama layers in the encoder.
    dim:
        Dimension of input/output features.
    ffn_dim:
        Dimension of the hidden layer in the feed-forward network.
    num_heads:
        Number of attention heads for the queries in the grouped-query
        attention mechanism.
    num_kv_heads:
        Number of heads for the keys and values in the grouped-query
        attention mechanism, typically fewer than `num_heads`.
    dropout:
        Dropout probability applied in the attention and feed-forward layers.
    norm_eps:
        Small constant added to the RMS normalization denominator for
        numerical stability.
    rope_theta:
        Scaling factor for the rotary positional embeddings,
        controlling the frequency range of positional encoding.
    max_seq_len:
        Maximum sequence length supported by the positional embeddings.

    """

    def __init__(
        self,
        num_layers: "int" = 8,
        dim: "int" = 512,
        ffn_dim: "int" = 2048,
        num_heads: "int" = 16,
        num_kv_heads: "int" = 4,
        dropout: "float" = 0.0,
        norm_eps: "float" = 1e-6,
        rope_theta: "float" = 10000.0,
        max_seq_len: "int" = 4096,
    ) -> "None":
        super().__init__()
        self.num_layers = num_layers
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.dropout = dropout
        self.norm_eps = norm_eps
        self.rope_theta = rope_theta
        self.max_seq_len = max_seq_len

        # Modules
        self.layers = nn.ModuleList(
            LlamaLayer(
                dim,
                ffn_dim,
                num_heads,
                num_kv_heads,
                dropout,
                norm_eps,
            )
            for _ in range(self.num_layers)
        )
        self.norm = RMSNorm(dim, norm_eps)

        # Non-persistent buffers
        self.register_buffer(
            "freqs_cis",
            self._precompute_freqs_cis(
                dim // num_heads,
                rope_theta,
                max_seq_len * 2,
            ),
            persistent=False,
        )

    def forward(
        self,
        input: "Tensor",
        prompt: "Optional[Tensor]" = None,
        length: "Optional[Tensor]" = None,
    ) -> "Tensor":
        """Forward pass.

        Parameters
        ----------
        input:
            Input tensor of shape (batch_size, seq_length, dim).
        prompt:
            Prompt tensor of shape (batch_size, seq_length, dim).
        length:
            Absolute lengths of each element in the batch.

        Returns
        -------
            Output tensor of shape (batch_size, seq_length, dim).

        """
        T = input.shape[1]
        device = input.device

        if length is None:
            key_padding_mask = None
        else:
            B = input.shape[0]
            key_padding_mask = (
                torch.arange(T, device=input.device).expand(B, T) < length[:, None]
            )
            input = input.clone()
            input[~key_padding_mask] = 0
            float_mask = torch.full(
                (B, T), -float("inf"), dtype=input.dtype, device=input.device
            )
            float_mask[key_padding_mask] = 0.0
            key_padding_mask = float_mask[:, None, None]

        output = input
        self.freqs_cis = self.freqs_cis.to(device)
        freqs_cis = self.freqs_cis[:T]
        for i, layer in enumerate(self.layers):
            if prompt is not None:
                # Prompt injection
                output += prompt
            output = layer(output, freqs_cis, key_padding_mask)
        output = self.norm(output)
        return output

    @torch.jit.export
    def _precompute_freqs_cis(
        self,
        dim: "int" = 64,
        rope_theta: "float" = 10000.0,
        max_seq_len: "int" = 4096,
        device: "torch.device" = "cpu",
    ) -> "Tensor":
        freqs = 1.0 / (
            rope_theta
            ** (torch.arange(0, dim, 2, device=device)[: (dim // 2)].float() / dim)
        )
        t = torch.arange(max_seq_len, device=freqs.device, dtype=torch.float32)
        freqs = t.outer(freqs)
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
        return freqs_cis


class MultiHeadLinear(nn.Linear):
    def __init__(
        self,
        in_features: "int",
        out_features: "int",
        num_codebooks: "int",
        **kwargs: "Any",
    ) -> "None":
        total_out_features = out_features * num_codebooks
        super().__init__(in_features, total_out_features, **kwargs)
        self.num_codebooks = num_codebooks

    def forward(self, input: "Tensor") -> "Tensor":
        input_shape = input.shape
        output = super().forward(input)
        output = output.reshape(*input_shape[:-1], self.num_codebooks, -1)
        return output


class CATTSModel(nn.Module):
    def __init__(
        self,
        vocab_size: "int" = 35,
        dim: "int" = 512,
        num_layers: "int" = 12,
        num_heads: "int" = 4,
        num_kv_heads: "int" = 1,
        dropout: "float" = 0.1,
        rope_theta: "float" = 10000.0,
        max_seq_len: "int" = 8192,
        num_codebooks: "int" = 32,
        codebook_size: "int" = 4,
        padding_idx: "int" = 0,
    ) -> "None":
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.dropout = dropout
        self.rope_theta = rope_theta
        self.max_seq_len = max_seq_len
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        self.padding_idx = padding_idx

        self.config = {
            "vocab_size": vocab_size,
            "dim": dim,
            "num_layers": num_layers,
            "num_heads": num_heads,
            "num_kv_heads": num_kv_heads,
            "dropout": dropout,
            "rope_theta": rope_theta,
            "max_seq_len": max_seq_len,
            "num_codebooks": num_codebooks,
            "codebook_size": codebook_size,
            "padding_idx": padding_idx,
        }

        # Modules
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=dim,
            padding_idx=padding_idx,
        )
        self.encoder = LlamaEncoder(
            num_layers=num_layers,
            dim=dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            dropout=dropout,
            rope_theta=rope_theta,
            max_seq_len=max_seq_len,
        )
        self.head = MultiHeadLinear(
            in_features=dim,
            out_features=codebook_size,
            num_codebooks=num_codebooks,
        )

    def forward(
        self,
        text_tokens: "Tensor",  # [B, N]
        text_lengths: "Tensor",  # [B]
        speaker_embs: "Tensor",  # [B, D]
    ) -> "Tensor":
        text_embs = self.embedding(text_tokens)  # [B, N, D]
        speaker_embs = speaker_embs[:, None, :].expand(
            -1, text_embs.shape[1], -1
        )  # [B, N, D]
        embs = self.encoder(text_embs, speaker_embs, length=text_lengths)  # [B, N, D]
        logits = self.head(embs)  # [B, N, K, C]
        audio_tokens = logits.argmax(dim=-1)  # [B, N, K]
        return audio_tokens

    def save_pretrained(self, checkpoint_path: "str") -> "None":
        torch.save(
            {
                "config": self.config,
                "state_dict": self.state_dict(),
            },
            checkpoint_path,
        )

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: "str",
        map_location: "Union[str, torch.device]" = "cpu",
        weights_only: "bool" = False,
        strict: "bool" = True,
    ) -> "CATTSModel":
        checkpoint = torch.load(
            checkpoint_path, map_location=map_location, weights_only=weights_only
        )
        config = dict(checkpoint["config"])
        model = cls(**config)
        model.load_state_dict(checkpoint["state_dict"], strict=strict)
        return model


class CATTS(nn.Module):
    def __init__(
        self,
        vocab_size: "int" = 35,
        dim: "int" = 512,
        num_layers: "int" = 12,
        num_heads: "int" = 4,
        num_kv_heads: "int" = 1,
        dropout: "float" = 0.1,
        rope_theta: "float" = 10000.0,
        max_seq_len: "int" = 8192,
        num_codebooks: "int" = 32,
        codebook_size: "int" = 4,
        padding_idx: "int" = 0,
    ) -> "None":
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.dropout = dropout
        self.rope_theta = rope_theta
        self.max_seq_len = max_seq_len
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        self.padding_idx = padding_idx

        # Modules
        self.model = CATTSModel(
            vocab_size=vocab_size,
            dim=dim,
            num_layers=num_layers,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            dropout=dropout,
            rope_theta=rope_theta,
            max_seq_len=max_seq_len,
            num_codebooks=num_codebooks,
            codebook_size=codebook_size,
            padding_idx=padding_idx,
        )
        self.codec = torch.hub.load(
            repo_or_dir="lucadellalib/dycast",
            model="dycast",
            config="lucadellalib/dycast",
            overrides={"retriever_name": None},
        )
        self.tokenizer = self.codec.char_aligner.processor.tokenizer

        # Delete unused modules
        self.codec.boundary_predictor = None
        self.codec.downsampler = None
        self.codec.char_aligner = None
        self.codec.retriever = None

        self.speaker_encoder = AutoModelForAudioXVector.from_pretrained(
            "microsoft/wavlm-base-sv"
        )

        self.model.eval().requires_grad_(False)
        self.codec.eval().requires_grad_(False)
        self.speaker_encoder.eval().requires_grad_(False)

    @property
    def sample_rate(self) -> "int":
        return self.codec.sample_rate

    @property
    def device(self) -> "torch.device":
        return next(self.parameters()).device

    def save_pretrained(self, checkpoint_path: "str") -> "None":
        self.model.save_pretrained(checkpoint_path)

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: "str",
        map_location: "Union[str, torch.device]" = "cpu",
        strict: "bool" = True,
    ) -> "CATTS":
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        config = dict(checkpoint["config"])
        model = cls(**config)
        model.model.load_state_dict(checkpoint["state_dict"], strict=strict)
        return model

    def _normalize_text_input(self, text: "Union[str, Sequence[str]]") -> "List[str]":
        if isinstance(text, str):
            return [text]
        return list(text)

    def _normalize_audio_input(
        self, prompt_audio: "Union[Tensor, Sequence[Tensor]]"
    ) -> "Tuple[Tensor, Tensor]":
        if isinstance(prompt_audio, Tensor):
            if prompt_audio.ndim == 1:
                padded_audio = prompt_audio[None]
                lengths = torch.tensor(
                    [prompt_audio.shape[0]],
                    dtype=torch.long,
                    device=prompt_audio.device,
                )
                return padded_audio, lengths

            if prompt_audio.ndim == 2:
                padded_audio = prompt_audio
                lengths = torch.full(
                    (prompt_audio.shape[0],),
                    prompt_audio.shape[1],
                    dtype=torch.long,
                    device=prompt_audio.device,
                )
                return padded_audio, lengths

            if prompt_audio.ndim == 3 and prompt_audio.shape[1] == 1:
                padded_audio = prompt_audio[:, 0]
                lengths = torch.full(
                    (padded_audio.shape[0],),
                    padded_audio.shape[1],
                    dtype=torch.long,
                    device=padded_audio.device,
                )
                return padded_audio, lengths

            raise ValueError("prompt_audio must have shape [T], [B, T], or [B, 1, T]")

        audios = []
        lengths = []
        for wav in prompt_audio:
            if wav.ndim == 2 and wav.shape[0] == 1:
                wav = wav[0]
            if wav.ndim != 1:
                raise ValueError("Each audio item must have shape [T] or [1, T]")
            audios.append(wav)
            lengths.append(wav.shape[0])

        max_len = max(lengths)
        padded_audio = torch.stack(
            [nn.functional.pad(wav, (0, max_len - wav.shape[0])) for wav in audios],
            dim=0,
        )
        audio_lengths = torch.tensor(
            lengths,
            dtype=torch.long,
            device=padded_audio.device,
        )
        return padded_audio, audio_lengths

    @torch.no_grad()
    def _normalize_reference_audio_pool(
        self,
        pool: "Sequence[Tensor]",
    ) -> "List[Tensor]":
        out = []
        for wav in pool:
            if wav.ndim == 2:
                if wav.shape[0] != 1:
                    raise ValueError(
                        "Each reference audio tensor must have shape [T] or [1, T]"
                    )
                wav = wav[0]
            elif wav.ndim != 1:
                raise ValueError(
                    "Each reference audio tensor must have shape [T] or [1, T]"
                )
            out.append(wav)
        return out

    def _broadcast_inputs(
        self,
        texts: "List[str]",
        padded_audio: "Tensor",
        audio_lengths: "Tensor",
    ) -> "Tuple[List[str], Tensor, Tensor]":
        n_text = len(texts)
        n_audio = padded_audio.shape[0]

        if n_text == n_audio:
            return texts, padded_audio, audio_lengths

        if n_text == 1 and n_audio > 1:
            return texts * n_audio, padded_audio, audio_lengths

        if n_audio == 1 and n_text > 1:
            return (
                texts,
                padded_audio.expand(n_text, -1),
                audio_lengths.expand(n_text),
            )

        raise ValueError(
            f"Incompatible batch sizes: {n_text} text items and {n_audio} audio items. "
            "Supported cases are B/B, 1/B, or B/1."
        )

    def _lengths_to_mask(
        self,
        lengths: "Tensor",
        max_len: "Optional[int]" = None,
    ):
        if max_len is None:
            max_len = int(lengths.max())
        return torch.arange(max_len, device=lengths.device)[None] < lengths[:, None]

    def _mask_from_relative_lengths(
        self,
        input: "Tensor",
        relative_lengths: "Optional[Tensor]",
    ) -> "Tensor":
        if relative_lengths is None:
            return input

        B, T = input.shape[:2]
        abs_lengths = (relative_lengths * T).round().clamp(min=0, max=T).long()  # [B]
        mask = (
            torch.arange(T, device=input.device)[None, :] < abs_lengths[:, None]
        )  # [B, T]

        # Expand mask to [B, T, 1, ..., 1]
        mask = mask.reshape(B, T, *([1] * (input.ndim - 2)))

        return input * mask.to(input.dtype)

    @torch.no_grad()
    def tokenize_text(
        self,
        text: "Union[str, Sequence[str]]",
    ) -> "Tuple[Tensor, Tensor]":
        texts = self._normalize_text_input(text)

        tokenized = self.tokenizer(
            [t.lower() for t in texts],
            padding=False,
            return_attention_mask=False,
        )["input_ids"]

        filtered = []
        lengths = []

        for ids in tokenized:
            ids = [
                tok for tok in ids if tok < self.vocab_size and tok != self.padding_idx
            ]
            if len(ids) == 0:
                ids = [self.padding_idx]
            filtered.append(torch.tensor(ids, dtype=torch.long))
            lengths.append(len(ids))

        max_len = max(lengths)
        tokens = torch.stack(
            [
                nn.functional.pad(
                    ids, (0, max_len - ids.numel()), value=self.padding_idx
                )
                for ids in filtered
            ],
            dim=0,
        ).to(self.device)

        lengths = torch.tensor(lengths, dtype=torch.long, device=self.device)
        return tokens, lengths

    @torch.no_grad()
    def encode_speaker(
        self,
        padded_audio: "Tensor",
        audio_lengths: "Tensor",
    ) -> "Tensor":
        padded_audio = padded_audio.to(self.device)
        audio_lengths = audio_lengths.to(self.device)

        attention_mask = self._lengths_to_mask(
            audio_lengths,
            padded_audio.shape[1],
        ).long()

        speaker_embs = self.speaker_encoder(
            input_values=padded_audio,
            attention_mask=attention_mask,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        ).embeddings

        return speaker_embs

    @torch.no_grad()
    def build_matching_pool(
        self,
        reference_audios: "Sequence[Tensor]",
    ) -> "Tensor":
        norm_pool = self._normalize_reference_audio_pool(reference_audios)
        if len(norm_pool) == 0:
            raise ValueError("Reference pool must contain at least one waveform.")

        pool_feats = []
        for sig in norm_pool:
            sig = sig.to(self.device)[None]  # [1, T]
            feats = self.codec.sig_to_feats(sig)  # [1, S, H]
            feats = feats[0]  # [S, H]
            pool_feats.append(feats)

        pool_feats = torch.cat(pool_feats, dim=0)  # [sum_i S_i, H]
        return pool_feats

    @torch.no_grad()
    def predict_tokens(
        self,
        text_tokens: "Tensor",
        text_lengths: "Tensor",
        speaker_embs: "Tensor",
    ) -> "Tensor":
        return self.model(text_tokens, text_lengths, speaker_embs)

    @torch.no_grad()
    def decode_tokens(
        self,
        audio_tokens: "Tensor",
        text_lengths: "Tensor",
        matching_pools: "Optional[Sequence[Tensor]]" = None,
        sample_duration: "bool" = False,
        num_frames: "Optional[Tensor]" = None,
        topk: "int" = 4,
        num_splits: "int" = 1,
    ) -> "Tensor":
        pcodes = self.codec.toks_to_pcodes(audio_tokens)
        relative_text_lengths = text_lengths.float() / audio_tokens.shape[1]
        durs = self.codec.pcodes_to_durs(
            pcodes,
            length=relative_text_lengths,
            sample=sample_duration,
            num_frames=num_frames,
        )
        codes, relative_lengths = self.codec.pcodes_to_codes(pcodes, durs)
        qfeats = self.codec.codes_to_qfeats(codes)  # [B, S, H]
        qfeats = self._mask_from_relative_lengths(qfeats, relative_lengths)

        if matching_pools is None:
            sig = self.codec.feats_to_sig(qfeats)
            return sig

        outs = []
        for i in range(qfeats.shape[0]):
            qfeats_i = qfeats[i : i + 1]  # [1, S, H]
            matching_set_i = matching_pools[i]  # [N_i, H]
            sig_i = self.codec.feats_to_sig(
                qfeats_i,
                matching_set=matching_set_i,
                topk=topk,
                num_splits=num_splits,
            )
            outs.append(sig_i[0])

        max_len = max(x.shape[-1] for x in outs)
        sig = torch.stack(
            [nn.functional.pad(x, (0, max_len - x.shape[-1])) for x in outs],
            dim=0,
        )
        return sig

    @torch.no_grad()
    def forward(
        self,
        text: "Union[str, Sequence[str]]",
        prompt_audio: "Union[Tensor, Sequence[Tensor]]",
        matching_pools: "Optional[Sequence[Tensor]]" = None,
        sample_duration: "bool" = False,
        num_frames: "Optional[Tensor]" = None,
        topk: "int" = 4,
        num_splits: "int" = 1,
    ) -> "Tensor":
        texts = self._normalize_text_input(text)
        padded_audio, audio_lengths = self._normalize_audio_input(prompt_audio)
        texts, padded_audio, audio_lengths = self._broadcast_inputs(
            texts,
            padded_audio,
            audio_lengths,
        )

        text_tokens, text_lengths = self.tokenize_text(texts)
        speaker_embs = self.encode_speaker(padded_audio, audio_lengths)
        audio_tokens = self.predict_tokens(text_tokens, text_lengths, speaker_embs)
        sig = self.decode_tokens(
            audio_tokens,
            text_lengths,
            matching_pools,
            sample_duration,
            num_frames,
            topk,
            num_splits,
        )

        return sig
