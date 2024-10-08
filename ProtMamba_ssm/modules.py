import torch.nn as nn

import json
import os

from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.modules.block import Block
from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
from mamba_ssm.models.mixer_seq_simple import _init_weights, MixerModel
from mamba_ssm.utils.hf import load_config_hf, load_state_dict_hf
from mamba_ssm.utils.generation import *
from transformers import PretrainedConfig
from torch.utils.checkpoint import checkpoint
from mamba_ssm.modules.mamba_simple import Mamba


@dataclass
class MambaConfig(PretrainedConfig):
    d_model: int = 2560
    n_layer: int = 64
    vocab_size: int = 50277
    ssm_cfg: dict = field(default_factory=dict)
    rms_norm: bool = True
    residual_in_fp32: bool = True
    fused_add_norm: bool = True
    pad_vocab_size_multiple: int = 8
    max_position_embeddings: int = 2048

# %% ../nbs/01_modules.ipynb 4
def sample_safe(logits, top_k=1, top_p=0.0, min_p=0.0, temperature=1.0):
    """Sample from top-k logits.
    Arguments:
        logits: Tensor of shape (batch_size, vocab_size)
    """
    if top_k == 1:  # Short-circuit for greedy decoding
        return logits.argmax(dim=-1)
    else:
        if top_p > 0.0:
            assert top_p <= 1.0, "top-p should be in (0, 1]."
        if top_k > 0:
            top_k = min(top_k, logits.size(-1))  # Safety check
            logits_top, indices = torch.topk(logits, top_k, dim=-1)
            if temperature != 1.0:
                logits_top /= temperature
            modify_logits_for_top_p_filtering(logits_top, top_p)
            return indices[
                torch.arange(indices.shape[0], device=indices.device),
                torch.multinomial(torch.softmax(logits_top, dim=-1), num_samples=1).squeeze(dim=-1),
            ]
        else:
            if min_p > 0.0:
                logits_top = logits.clone()
                max_prob = logits_top[..., 0].item()
                min_prob = max_prob * min_p
                modify_logits_for_min_p_filtering(logits_top, min_p)
                if temperature != 1.0:
                    logits_top /= temperature
                return torch.multinomial(torch.softmax(logits_top, dim=-1), num_samples=1).squeeze(dim=-1)
            # Clone so that when we modify for top_p we don't change the original logits
            logits_top = logits / temperature if temperature != 1.0 else logits.clone()
            modify_logits_for_top_p_filtering(logits_top, top_p)
            return torch.multinomial(torch.softmax(logits_top, dim=-1), num_samples=1).squeeze(
                dim=-1
            )

@torch.inference_mode()
def decode_safe(
        input_ids,
        position_ids,
        seq_position_ids,
        is_fim,
        model,
        max_length,
        top_k=1,
        top_p=0.0,
        min_p=0.0,
        temperature=1.0,
        repetition_penalty=1.0,
        eos_token_id=None,
        teacher_outputs=None,
        vocab_size=None,
        cg=False,
        enable_timing=False,
        streamer: Optional[TextStreamer] = None
):
    """Decoding, either greedy or with top-k or top-p sampling.
    If top-k = 0, don't limit the number of candidates (pure sampling).
    Top-k and top-p can be used together. If top_k > 0 and top_p > 0, then top-k is applied first,
    then top-p.
    We assume that all sequences in the same batch have the same length.

    Arguments:
        input_ids: (batch, seq_len)
        max_length: int
        is_fim: dictionary with mask indices and associated position indices
        teacher_outputs (optional): (batch, seq_len). If provided, instead of sampling from the
            logits, the next token is taken from the teacher_outputs. Useful for testing.
    Returns: GreedySearchDecoderOnlyOutput or SampleDecoderOnlyOutput, with the following fields:
        sequences: (batch, max_length)
        scores: tuples of (batch, vocab_size)
    """
    if streamer is not None:
        streamer.put(input_ids.cpu())

    batch_size, seqlen_og = input_ids.shape
    teacher_output_len = teacher_outputs.shape[1] if teacher_outputs is not None else 0
    if cg:
        if not hasattr(model, "_decoding_cache"):
            model._decoding_cache = None
        model._decoding_cache = update_graph_cache(
            model,
            model._decoding_cache,
            batch_size,
            seqlen_og,
            max_length,
        )
        inference_params = model._decoding_cache.inference_params
        inference_params.reset(max_length, batch_size)
    else:
        inference_params = InferenceParams(max_seqlen=max_length, max_batch_size=batch_size)

    def get_logits(input_ids, position_ids, seq_position_ids, inference_params):
        decoding = inference_params.seqlen_offset > 0
        if not cg or not decoding:
            logits = model(
                input_ids,
                position_ids=position_ids,
                seq_position_ids=seq_position_ids,
                inference_params=inference_params,
                num_last_tokens=1,
            ).logits.squeeze(dim=1)
        else:
            logits = model._decoding_cache.run(
                input_ids, position_ids, inference_params.seqlen_offset, seq_position_ids=seq_position_ids
            ).squeeze(dim=1)
        return logits[..., :vocab_size] if vocab_size is not None else logits

    def sample_tokens(logits, inference_params):
        if teacher_outputs is None or teacher_output_len <= inference_params.seqlen_offset:
            token = sample_safe(logits, top_k=top_k, top_p=top_p, min_p=min_p, temperature=temperature)
        else:
            token = teacher_outputs[:, inference_params.seqlen_offset]
        # return rearrange(token, "b -> b 1")
        return token.unsqueeze(1)

    def get_fim_position_id(last_position_ids, sampled_tokens, is_fim, repeat_next=False):
        if type(is_fim) is dict:
            val = int(last_position_ids) + 1
            should_repeat_next = False
            if is_fim and int(sampled_tokens) in is_fim:
                val = is_fim[int(sampled_tokens)]
                should_repeat_next = True
            elif repeat_next:
                val = int(last_position_ids)
            return torch.full_like(last_position_ids, fill_value=val), should_repeat_next
        else:
            t = [get_fim_position_id(last_position_ids_, sampled_tokens_, is_fim_dict, repeat_next) for
                 (last_position_ids_, sampled_tokens_, is_fim_dict) in
                 zip(last_position_ids, sampled_tokens, is_fim)]
            return torch.stack([t_[0] for t_ in t], dim=0), t[0][1]

    def should_stop(current_token, inference_params):
        if inference_params.seqlen_offset == 0:
            return False
        if eos_token_id is not None and (current_token == eos_token_id).any():
            if current_token.shape[1] > 1:
                raise NotImplementedError("Batched eos_token_id not supported")
            return True
        if inference_params.seqlen_offset >= max_length - 1:
            return True
        return False

    start = torch.cuda.Event(enable_timing=enable_timing)
    end = torch.cuda.Event(enable_timing=enable_timing)

    if enable_timing:
        start.record()
    scores, sequences = [], [input_ids]
    new_position_ids, new_seq_position_ids = [position_ids], [seq_position_ids]
    sequences_cat = input_ids
    repeat_next = False
    while not should_stop(sequences[-1], inference_params):
        scores.append(get_logits(sequences[-1], new_position_ids[-1], new_seq_position_ids[-1], inference_params))
        inference_params.seqlen_offset += sequences[-1].shape[1]
        if repetition_penalty == 1.0:
            sampled_tokens = sample_tokens(scores[-1], inference_params)
        else:
            logits = modify_logit_for_repetition_penalty(
                scores[-1].clone(), sequences_cat, repetition_penalty
            )
            sampled_tokens = sample_tokens(logits, inference_params)
            sequences_cat = torch.cat([sequences_cat, sampled_tokens], dim=1)
        sequences.append(sampled_tokens)
        # Update position_ids
        if position_ids is not None:
            last_position_ids, repeat_next = get_fim_position_id(new_position_ids[-1][:, -1:], sampled_tokens, is_fim,
                                                                 repeat_next)
            new_position_ids.append(last_position_ids)
        # Update seq_position_ids
        if seq_position_ids is not None:
            new_seq_position_ids.append(new_seq_position_ids[-1][:, -1:])

        if streamer is not None:
            streamer.put(sampled_tokens.cpu())
    if streamer is not None:
        streamer.end()
    if enable_timing:
        end.record()
        torch.cuda.synchronize()
        print(f"Prompt processing + decoding time: {(start.elapsed_time(end)):.0f}ms")
    output_cls = GreedySearchDecoderOnlyOutput if top_k == 1 else SampleDecoderOnlyOutput
    return output_cls(sequences=torch.cat(sequences, dim=1), scores=tuple(scores))

class GenerationMixinSafe(GenerationMixin):
    
    def generate(
        self,
        input_ids,
        position_ids,
        seq_position_ids,
        is_fim=None,
        max_length=1,
        top_k=1,
        top_p=0.0,
        min_p=0.0,
        temperature=1.0,
        return_dict_in_generate=False,
        output_scores=False,
        **kwargs):
        
        output = decode_safe(
            input_ids, position_ids, seq_position_ids, is_fim, self, max_length, top_k=top_k, top_p=top_p, min_p=min_p, temperature=temperature, **kwargs
        )
        if not output_scores:
            output.scores = None
        return output if return_dict_in_generate else output.sequences

class CheckpointedModule(torch.nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.ckpt_layer = layer

    def forward(self, x, *args, **kwargs):
        return checkpoint(self.ckpt_layer, x, use_reentrant=False)

    # def state_dict(self, **kwargs):
    #     # Get the state dict of the underlying layer
    #     layer_state_dict = self.ckpt_layer.state_dict(**kwargs)
    #     # Create a new state dict with the original keys
    #     state_dict = {k.replace('ckpt_layer.', ''): v for k, v in layer_state_dict.items()}
    #     return state_dict

def create_block(
    d_model,
    ssm_cfg=None,
    norm_epsilon=1e-5,
    rms_norm=False,
    residual_in_fp32=False,
    fused_add_norm=False,
    layer_idx=None,
    device=None,
    dtype=None,
    checkpoint_mixer=False,
):
    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}
    mixer_cls = partial(Mamba, layer_idx=layer_idx, **ssm_cfg, **factory_kwargs)
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )
    block = Block(
        d_model,
        mixer_cls,
        mlp_cls=nn.Identity,
        norm_cls=norm_cls,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
    )
    block.layer_idx = layer_idx
    if checkpoint_mixer:
        block.mixer = CheckpointedModule(block.mixer)
    return block


class MixerModelSafe(MixerModel):
    """
        Overwrite the forward method to allow saving intermediate layers.
    """
    
    def forward(self, input_ids, inference_params=None, save_layer=[]):
        hidden_states = self.embedding(input_ids)
        residual = None
        if len(save_layer) > 0:
            hidden_states_dict = {}
        for i, layer in enumerate(self.layers):
            hidden_states, residual = layer(
                hidden_states, residual, inference_params=inference_params
            )
            if i+1 in save_layer:
                hidden_states_dict[i+1] = hidden_states.detach().cpu().to(torch.float).numpy()
        if len(save_layer) > 0:
            return hidden_states_dict
            
        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            hidden_states = fused_add_norm_fn(
                hidden_states,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )
        return hidden_states

class MambaLMHeadModelSafe(nn.Module, GenerationMixinSafe):

    def __init__(
            self,
            config: MambaConfig,
            initializer_cfg=None,
            device=None,
            dtype=None,
            checkpoint_mixer=False,
    ) -> None:
        self.config = config
        d_model = config.d_model
        n_layer = config.n_layer
        vocab_size = config.vocab_size
        ssm_cfg = config.ssm_cfg
        rms_norm = config.rms_norm
        residual_in_fp32 = config.residual_in_fp32
        fused_add_norm = config.fused_add_norm
        pad_vocab_size_multiple = config.pad_vocab_size_multiple
        factory_kwargs = {"device": device, "dtype": dtype}
        if checkpoint_mixer:
            raise NotImplementedError("Checkpointing is not yet supported for MambaLMHeadModelSafe")

        super().__init__()
        if vocab_size % pad_vocab_size_multiple != 0:
            vocab_size += pad_vocab_size_multiple - (vocab_size % pad_vocab_size_multiple)
        self.backbone = MixerModelSafe(
            d_model=d_model,
            n_layer=n_layer,
            vocab_size=vocab_size,
            ssm_cfg=ssm_cfg,
            rms_norm=rms_norm,
            initializer_cfg=initializer_cfg,
            fused_add_norm=fused_add_norm,
            residual_in_fp32=residual_in_fp32,
            **factory_kwargs,
        )
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False, **factory_kwargs)

        # Initialize weights and apply final processing
        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )
        self.tie_weights()
    
    def tie_weights(self):
        self.lm_head.weight = self.backbone.embedding.weight
        
    def clip_grad_norm_(self, max_norm, norm_type=2.0):
        r"""Clip the norm of the gradients for the model.
        Args:
            max_norm (float or int): The maximum norm of the gradients.
                The gradients are modified in-place.
            norm_type (float or int): The type of the used p-norm. Can be 'inf' for infinity norm.
        Returns:
            Total norm of the parameters (viewed as a single vector).
        """
        return torch.nn.utils.clip_grad_value_(self.parameters(), max_norm)

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.backbone.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)

    def forward(self, input_ids, position_ids=None, inference_params=None, num_last_tokens=0, save_layer=[], *args, **kwargs):
        """
        "position_ids" is just to be compatible with Transformer generation. We don't use it.
        num_last_tokens: if > 0, only return the logits for the last n tokens
        """
        return self.protected_forward(input_ids, position_ids, inference_params, num_last_tokens, save_layer)

    def protected_forward(self, input_ids, position_ids=None, inference_params=None, num_last_tokens=0, save_layer=[]):
        hidden_states = self.backbone(input_ids, inference_params=inference_params, save_layer=save_layer)
        if len(save_layer) > 0:
            return hidden_states
        if num_last_tokens > 0:
            hidden_states = hidden_states[:, -num_last_tokens:]
        lm_logits = self.lm_head(hidden_states)
        CausalLMOutput = namedtuple("CausalLMOutput", ["loss", "logits"])
        return CausalLMOutput(loss=None, logits=lm_logits)

    @classmethod
    def from_pretrained(cls, pretrained_model_name, device=None, dtype=None, **kwargs):
        config_data = load_config_hf(pretrained_model_name)
        config = MambaConfig(**config_data)
        model = cls(config, device=device, dtype=dtype, **kwargs)
        model.load_state_dict(load_state_dict_hf(pretrained_model_name, device=device, dtype=dtype), strict=False)
        return model

    def save_pretrained(self, save_directory):
        """
        Minimal implementation of save_pretrained for MambaLMHeadModel.
        Save the model and its configuration file to a directory.
        """
        # Ensure save_directory exists
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        # Save the model's state_dict
        model_path = os.path.join(save_directory, 'pytorch_model.bin')
        torch.save(self.state_dict(), model_path)

        # Save the configuration of the model
        config_path = os.path.join(save_directory, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(self.config.__dict__, f)

# %% ../nbs/01_modules.ipynb 9
class MixerModelWithPosids(nn.Module):
    r"""Mixer model for Mamba but we add positional encodings to the input embeddings."""

    def __init__(
            self,
            d_model: int,
            n_layer: int,
            vocab_size: int,
            max_position_embeddings: int,
            ssm_cfg=None,
            norm_epsilon: float = 1e-5,
            rms_norm: bool = False,
            initializer_cfg=None,
            fused_add_norm=False,
            residual_in_fp32=False,
            device=None,
            dtype=None,
            checkpoint_mixer=False,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32

        self.embedding = nn.Embedding(vocab_size, d_model // 2, **factory_kwargs)
        self.position_embedding = nn.Embedding(max_position_embeddings, d_model - d_model // 2, **factory_kwargs)

        # We change the order of residual and layer norm:
        # Instead of LN -> Attn / MLP -> Add, we do:
        # Add -> LN -> Attn / MLP / Mixer, returning both the residual branch (output of Add) and
        # the main branch (output of MLP / Mixer). The model definition is unchanged.
        # This is for performance reason: we can fuse add + layer_norm.
        self.fused_add_norm = fused_add_norm
        if self.fused_add_norm:
            if layer_norm_fn is None or rms_norm_fn is None:
                raise ImportError("Failed to import Triton LayerNorm / RMSNorm kernels")

        self.layers = nn.ModuleList(
            [
                create_block(
                    d_model,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    checkpoint_mixer=checkpoint_mixer,
                    **factory_kwargs,
                )
                for i in range(n_layer)
            ]
        )

        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            d_model, eps=norm_epsilon, **factory_kwargs
        )

        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }

    def forward(self, input_ids, position_ids, inference_params=None, save_layer=[]):
        hidden_states = torch.cat([self.embedding(input_ids), self.position_embedding(position_ids), ], -1)
        residual = None
        if len(save_layer) > 0:
            hidden_states_dict = {}
        if 0 in save_layer:
            hidden_states_dict[0] = hidden_states.detach().cpu().to(torch.float).numpy()
        for i, layer in enumerate(self.layers):
            hidden_states, residual = layer(
                hidden_states, residual, inference_params=inference_params
            )
            if i+1 in save_layer:
                hidden_states_dict[i+1] = hidden_states.detach().cpu().to(torch.float).numpy()
        if len(save_layer) > 0:
            return hidden_states_dict
            
        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            hidden_states = fused_add_norm_fn(
                hidden_states,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )
        return hidden_states


class MixerModelWith2DPosids(nn.Module):
    r"""Mixer model for Mamba but we add positional encodings to the input embeddings."""

    def __init__(
            self,
            d_model: int,
            n_layer: int,
            vocab_size: int,
            max_position_embeddings: int,
            max_sequence_position_embeddings: int = 512,
            ssm_cfg=None,
            norm_epsilon: float = 1e-5,
            rms_norm: bool = False,
            initializer_cfg=None,
            fused_add_norm=False,
            residual_in_fp32=False,
            device=None,
            dtype=None,
            checkpoint_mixer=False,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32

        self.embedding = nn.Embedding(vocab_size, d_model - 2 * d_model // 4, **factory_kwargs)
        self.position_embedding = nn.Embedding(max_position_embeddings, d_model // 4, **factory_kwargs)
        self.seq_position_embedding = nn.Embedding(max_sequence_position_embeddings, d_model // 4, **factory_kwargs)
        self.d_embeddings = d_model - 2 * d_model // 4

        # We change the order of residual and layer norm:
        # Instead of LN -> Attn / MLP -> Add, we do:
        # Add -> LN -> Attn / MLP / Mixer, returning both the residual branch (output of Add) and
        # the main branch (output of MLP / Mixer). The model definition is unchanged.
        # This is for performance reason: we can fuse add + layer_norm.
        self.fused_add_norm = fused_add_norm
        if self.fused_add_norm:
            if layer_norm_fn is None or rms_norm_fn is None:
                raise ImportError("Failed to import Triton LayerNorm / RMSNorm kernels")

        self.layers = nn.ModuleList(
            [
                create_block(
                    d_model,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    checkpoint_mixer=checkpoint_mixer,
                    **factory_kwargs,
                )
                for i in range(n_layer)
            ]
        )

        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            d_model, eps=norm_epsilon, **factory_kwargs
        )

        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }

    def forward(self, input_ids, position_ids, seq_position_ids, inference_params=None, save_layer=[]):
        hidden_states = torch.cat([self.embedding(input_ids), self.position_embedding(position_ids), self.seq_position_embedding(seq_position_ids), ], -1)
        residual = None
        if len(save_layer) > 0:
            hidden_states_dict = {}
        for i, layer in enumerate(self.layers):
            hidden_states, residual = layer(
                hidden_states, residual, inference_params=inference_params
            )
            if i+1 in save_layer:
                hidden_states_dict[i+1] = hidden_states.detach().cpu().to(torch.float).numpy()
        if len(save_layer) > 0:
            return hidden_states_dict
        
        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            hidden_states = fused_add_norm_fn(
                hidden_states,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )
        return hidden_states



class MambaLMHeadModelwithPosids(nn.Module, GenerationMixinSafe):

    def __init__(
            self,
            config: MambaConfig,
            initializer_cfg=None,
            device=None,
            dtype=None,
            checkpoint_mixer=False,
    ) -> None:
        self.config = config
        d_model = config.d_model
        n_layer = config.n_layer
        vocab_size = config.vocab_size
        max_position_embeddings = config.max_position_embeddings
        ssm_cfg = config.ssm_cfg
        rms_norm = config.rms_norm
        residual_in_fp32 = config.residual_in_fp32
        fused_add_norm = config.fused_add_norm
        pad_vocab_size_multiple = config.pad_vocab_size_multiple
        factory_kwargs = {"device": device, "dtype": dtype}

        super().__init__()
        if vocab_size % pad_vocab_size_multiple != 0:
            vocab_size += pad_vocab_size_multiple - (vocab_size % pad_vocab_size_multiple)
        self.backbone = MixerModelWithPosids(
            d_model=d_model,
            n_layer=n_layer,
            vocab_size=vocab_size,
            max_position_embeddings=max_position_embeddings,
            ssm_cfg=ssm_cfg,
            rms_norm=rms_norm,
            initializer_cfg=initializer_cfg,
            fused_add_norm=fused_add_norm,
            residual_in_fp32=residual_in_fp32,
            checkpoint_mixer=checkpoint_mixer,
            **factory_kwargs,
        )
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False, **factory_kwargs)

        # Initialize weights and apply final processing
        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )
        self.tie_weights()

    def tie_weights(self):
        self.lm_head.weight = self.backbone.embedding.weight

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.backbone.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)

    def forward(self, input_ids, position_ids=None, inference_params=None, num_last_tokens=0, save_layer=[], *args, **kwargs):
        """
        "position_ids" is just to be compatible with Transformer generation. We don't use it.
        num_last_tokens: if > 0, only return the logits for the last n tokens
        """
        return self.protected_forward(input_ids, position_ids, inference_params, num_last_tokens, save_layer)

    def protected_forward(self, input_ids, position_ids=None, inference_params=None, num_last_tokens=0, save_layer=[], ):
        hidden_states = self.backbone(input_ids, position_ids=position_ids, inference_params=inference_params, save_layer=save_layer)
        if len(save_layer) > 0:
            return hidden_states
        hidden_states = hidden_states[:, :, :self.config.d_model // 2]
        if num_last_tokens > 0:
            hidden_states = hidden_states[:, -num_last_tokens:]
        lm_logits = self.lm_head(hidden_states)
        CausalLMOutput = namedtuple("CausalLMOutput", ["loss", "logits", "hidden_states"])
        if len(save_layer) > 0:
            return CausalLMOutput(loss=None, logits=lm_logits, hidden_states=hidden_states)
        return CausalLMOutput(loss=None, logits=lm_logits, hidden_states=None)
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name, device=None, dtype=None, checkpoint_mixer=False, **kwargs):
        config_data = load_config_hf(pretrained_model_name)
        config = MambaConfig(**config_data)
        model = cls(config, device=device, dtype=dtype, checkpoint_mixer=checkpoint_mixer, **kwargs)
        state_dict = load_state_dict_hf(pretrained_model_name, device=device, dtype=dtype)
        if state_dict.keys() != model.state_dict().keys():
            if checkpoint_mixer:
                for key in model.state_dict().keys():
                    if "ckpt_layer" in key:
                        state_dict[key] = state_dict.pop(key.replace("ckpt_layer.", ""))
                print("Using a model that was pretrained without gradient checkpointing and now want to use it. Changed the keys of the state_dict to match the model's keys.")
            else:
                for key in list(state_dict.keys()):
                    if "ckpt_layer" in key:
                        state_dict[key.replace("ckpt_layer.", "")] = state_dict.pop(key)
                print("Using a model that was pretrained with gradient checkpointing but now do not want to use it. Changed the keys of the state_dict to match the model's keys.")
            assert state_dict.keys() == model.state_dict().keys(), "The keys of the state_dict do not match the model's keys."
        model.load_state_dict(state_dict)
        return model

    def save_pretrained(self, save_directory):
        """
        Minimal implementation of save_pretrained for MambaLMHeadModel.
        Save the model and its configuration file to a directory.
        """
        # Ensure save_directory exists
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        # Save the model's state_dict
        model_path = os.path.join(save_directory, 'pytorch_model.bin')
        torch.save(self.state_dict(), model_path)

        # Save the configuration of the model
        config_path = os.path.join(save_directory, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(self.config.__dict__, f)


class MambaLMHeadModelwith2DPosids(nn.Module, GenerationMixinSafe):

    def __init__(
            self,
            config: MambaConfig,
            initializer_cfg=None,
            device=None,
            dtype=None,
            checkpoint_mixer=False,
    ) -> None:
        self.config = config
        d_model = config.d_model
        n_layer = config.n_layer
        vocab_size = config.vocab_size
        max_position_embeddings = config.max_position_embeddings
        ssm_cfg = config.ssm_cfg
        rms_norm = config.rms_norm
        residual_in_fp32 = config.residual_in_fp32
        fused_add_norm = config.fused_add_norm
        pad_vocab_size_multiple = config.pad_vocab_size_multiple
        factory_kwargs = {"device": device, "dtype": dtype}

        super().__init__()
        if vocab_size % pad_vocab_size_multiple != 0:
            vocab_size += pad_vocab_size_multiple - (vocab_size % pad_vocab_size_multiple)
        self.backbone = MixerModelWith2DPosids(
            d_model=d_model,
            n_layer=n_layer,
            vocab_size=vocab_size,
            max_position_embeddings=max_position_embeddings,
            ssm_cfg=ssm_cfg,
            rms_norm=rms_norm,
            initializer_cfg=initializer_cfg,
            fused_add_norm=fused_add_norm,
            residual_in_fp32=residual_in_fp32,
            checkpoint_mixer=checkpoint_mixer,
            **factory_kwargs,
        )
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False, **factory_kwargs)

        # Initialize weights and apply final processing
        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )
        self.tie_weights()

    def tie_weights(self):
        self.lm_head.weight = self.backbone.embedding.weight

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.backbone.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)

    def forward(self, input_ids, position_ids=None, seq_position_ids=None, inference_params=None, num_last_tokens=0, save_layer=[], *args, **kwargs):
        """
        "position_ids" is just to be compatible with Transformer generation. We don't use it.
        num_last_tokens: if > 0, only return the logits for the last n tokens
        """
        return self.protected_forward(input_ids, position_ids, seq_position_ids, inference_params, num_last_tokens, save_layer)

    def protected_forward(self, input_ids, position_ids=None, seq_position_ids=None, inference_params=None, num_last_tokens=0, save_layer=[]):
        hidden_states = self.backbone(input_ids, position_ids=position_ids, seq_position_ids=seq_position_ids, inference_params=inference_params, save_layer=save_layer)
        if len(save_layer) > 0:
            return hidden_states
        hidden_states = hidden_states[:, :, :self.backbone.d_embeddings]
        if num_last_tokens > 0:
            hidden_states = hidden_states[:, -num_last_tokens:]
        lm_logits = self.lm_head(hidden_states)
        CausalLMOutput = namedtuple("CausalLMOutput", ["loss", "logits"])
        return CausalLMOutput(loss=None, logits=lm_logits)

    @classmethod
    def from_pretrained(cls, pretrained_model_name, device=None, dtype=None, checkpoint_mixer=False, **kwargs):
        config_data = load_config_hf(pretrained_model_name)
        config = MambaConfig(**config_data)
        model = cls(config, device=device, dtype=dtype, checkpoint_mixer=checkpoint_mixer, **kwargs)
        state_dict = load_state_dict_hf(pretrained_model_name, device=device, dtype=dtype)
        if state_dict.keys() != model.state_dict().keys():
            if checkpoint_mixer:
                for key in model.state_dict().keys():
                    if "ckpt_layer" in key:
                        state_dict[key] = state_dict.pop(key.replace("ckpt_layer.", ""))
                print("Using a model that was pretrained without gradient checkpointing and now want to use it. Changed the keys of the state_dict to match the model's keys.")
            else:
                for key in list(state_dict.keys()):
                    if "ckpt_layer" in key:
                        state_dict[key.replace("ckpt_layer.", "")] = state_dict.pop(key)
                print("Using a model that was pretrained with gradient checkpointing but now do not want to use it. Changed the keys of the state_dict to match the model's keys.")
            assert state_dict.keys() == model.state_dict().keys(), "The keys of the state_dict do not match the model's keys."
        model.load_state_dict(state_dict)
        return model

    def save_pretrained(self, save_directory):
        """
        Minimal implementation of save_pretrained for MambaLMHeadModel.
        Save the model and its configuration file to a directory.
        """
        # Ensure save_directory exists
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        # Save the model's state_dict
        model_path = os.path.join(save_directory, 'pytorch_model.bin')
        torch.save(self.state_dict(), model_path)

        # Save the configuration of the model
        config_path = os.path.join(save_directory, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(self.config.__dict__, f)

# %% ../nbs/01_modules.ipynb 10
def load_model(model_path, device, model_class=MambaLMHeadModelSafe, dtype=torch.bfloat16, checkpoint_mixer=False):
    model = model_class.from_pretrained(model_path, device=device, dtype=dtype, checkpoint_mixer=checkpoint_mixer)
    return model
