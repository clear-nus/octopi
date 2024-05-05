import torch
from torch import nn
from transformers.models.clip.modeling_clip import CLIPModel, CLIPVisionTransformer, CLIPTextTransformer, CLIPEncoder, CLIPEncoderLayer
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from typing import Any, Optional, Tuple, Union


class PromptLearningCLIPEncoderLayer(CLIPEncoderLayer):
    def __init__(self, config, configs, text_layer, layer_idx):
        super().__init__(config)
        self.text_layer = text_layer
        if layer_idx != 0:
            self.add_prompt = True
            if self.text_layer:
                self.n_ctx_text = configs["num_context_text"] # hyperparameter
                ctx_vectors = torch.empty(self.n_ctx_text, configs["dim_context_text"])
            else:
                self.n_ctx_visual = configs["num_context_vision"] # hyperparameter
                ctx_vectors = torch.empty(self.n_ctx_visual, configs["dim_context_vision"])
            # Code snippet for per layer visual prompts
            nn.init.normal_(ctx_vectors, std=0.02)
            self.VPT_shallow = nn.Parameter(ctx_vectors)
        else:
            self.add_prompt = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        causal_attention_mask: torch.Tensor,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
                `(config.encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        # add prompts to hidden_states
        if self.add_prompt:
            if not self.text_layer:
                # hidden_states -> (N=40, L=257, DIM=1024)
                # Remove the outputs produced by learnable tokens of previous layer
                prefix = hidden_states[:, :hidden_states.shape[1] - self.n_ctx_visual, :] # (N, L, DIM)
                visual_context = self.VPT_shallow.expand(hidden_states.shape[0], -1, -1) # (N, n_ctx, DIM)
                # visual_context = self.VPT_shallow.expand(hidden_states.shape[1], -1, -1).permute(1, 0, 2) # .half()
                hidden_states = torch.cat([prefix, visual_context], dim=1) # (N, L + n_ctx, DIM)
            else:
                # hidden_states -> (N_CLS, 17, DIM=768) --> 6 belongs to "A tactile sensor photo of"
                # Remove the outputs produced by learnable tokens of previous layer
                prefix = hidden_states[:, :1, :]
                suffix = hidden_states[:, 1 + self.n_ctx_text:, :]
                textual_context = self.VPT_shallow.expand(hidden_states.shape[0], -1, -1)
                # textual_context = self.VPT_shallow.expand(hidden_states.shape[1], -1, -1).permute(1, 0, 2) # .half()
                hidden_states = torch.cat([prefix, textual_context, suffix], dim=1)

        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs
        

class PromptLearningCLIPEncoder(CLIPEncoder):
    def __init__(self, config, configs, text_layer):
        super().__init__(config)
        self.config = config
        self.layers = nn.ModuleList([PromptLearningCLIPEncoderLayer(config, configs, text_layer, layer_idx) for layer_idx in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False


class PromptLearningCLIPVisionTransformer(CLIPVisionTransformer):
    def __init__(self, config, configs, text_layer):
        super().__init__(config)
        self.encoder = PromptLearningCLIPEncoder(config, configs, text_layer)
        if configs["prompt_depth_vision"] == 0:
            self.VPT_shallow = False
        else:
            self.VPT_shallow = True
        if self.VPT_shallow:
            # Add visual prompt tokens here
            n_ctx = configs["num_context_vision"]  # hyperparameter
            ctx_vectors = torch.empty(n_ctx, configs["dim_context_vision"])
            nn.init.normal_(ctx_vectors, std=0.02)
            self.VPT = nn.Parameter(ctx_vectors)
        self.prompt_till_layer_visual = configs["prompt_depth_vision"]
    
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:

        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        hidden_states = self.embeddings(pixel_values) # (N, L, D)

        if self.VPT_shallow:
            visual_ctx = self.VPT.expand(hidden_states.shape[0], -1, -1) # .half() # (N, n_ctx, D)
            hidden_states = torch.cat([hidden_states, visual_ctx], dim=1) # (N, L + n_ctx, D)
        else:
            assert self.prompt_till_layer_visual == 0

        hidden_states = self.pre_layrnorm(hidden_states)

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]
        pooled_output = last_hidden_state[:, 0, :]
        pooled_output = self.post_layernorm(pooled_output)

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class PromptLearningCLIPTextTransformer(CLIPTextTransformer):
    def __init__(self, config, configs, text_layer):
        super().__init__(config)
        self.encoder = PromptLearningCLIPEncoder(config, configs, text_layer)


class PromptLearningCLIPModel(CLIPModel):
    def __init__(self, config, configs):
        super().__init__(config)
        text_config = config.text_config
        vision_config = config.vision_config
        self.text_model = PromptLearningCLIPTextTransformer(text_config, configs, text_layer=True)
        self.vision_model = PromptLearningCLIPVisionTransformer(vision_config, configs, text_layer=False)
        # Initialize weights and apply final processing
        self.post_init()