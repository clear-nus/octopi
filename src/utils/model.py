import torch
import torch.nn.functional as F
from torch import nn
from transformers import CLIPVisionModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from utils.constants import *


class CLIPTactileEncoder(nn.Module):
    def __init__(self, clip_model):
        super(CLIPTactileEncoder, self).__init__()
        self.model = CLIPVisionModel.from_pretrained(clip_model)

    def forward(self, tactile_embeds):
        b, l, c, h, w = tactile_embeds.shape # (b, l, c, h, w)
        tactile_embeds = tactile_embeds.reshape(b * l, c, h, w) # (b * l, c, h, w)
        tactile_forward_outs = self.model(tactile_embeds, output_hidden_states=True)
        tactile_features = tactile_forward_outs.hidden_states[-2][:, 0].to(tactile_embeds.dtype)
        _, patch_embed_size = tactile_features.shape
        tactile_features = tactile_features.reshape(b, l, patch_embed_size) # (b, l, patch_embed_size)
        return tactile_features
    

class CLIPClassifier(nn.Module):
    def __init__(self, output_size, decoupled_heads=False, decoupled_head_dim=128):
        super(CLIPClassifier, self).__init__()
        self.decoupled_heads = decoupled_heads
        if decoupled_heads:
            # Each property gets its own trunk so class-balanced reweighting on one
            # head no longer perturbs the others through a shared fc layer. Width is
            # kept narrow (default 128) so total params stay at/below the shared trunk;
            # shared-feature learning still happens in the (shared) upstream encoder.
            head_in = decoupled_head_dim
            def _trunk():
                return nn.Sequential(nn.Linear(output_size, head_in), nn.Dropout(0.5), nn.ReLU())
            self.hardness_trunk = _trunk()
            self.roughness_trunk = _trunk()
            self.texture_trunk = _trunk()
        else:
            head_in = 512
            self.fc = nn.Linear(output_size, head_in)
            self.act = nn.ReLU()
            self.dropout = nn.Dropout(0.5)
        self.hardness_fc = nn.Linear(head_in, 3)
        self.roughness_fc = nn.Linear(head_in, 3)
        self.texture_fc = nn.Linear(head_in, 3)

    def forward(self, vision_features):
        if self.decoupled_heads:
            hardness_preds = self.hardness_fc(self.hardness_trunk(vision_features))
            roughness_preds = self.roughness_fc(self.roughness_trunk(vision_features))
            texture_preds = self.texture_fc(self.texture_trunk(vision_features))
            return hardness_preds, roughness_preds, texture_preds
        vision_features = self.act(self.dropout(self.fc(vision_features)))
        hardness_preds = self.hardness_fc(vision_features)
        roughness_preds = self.roughness_fc(vision_features)
        texture_preds = self.texture_fc(vision_features)
        return hardness_preds, roughness_preds, texture_preds
        

def sinusoidal_positional_embedding(token_sequence_size, indices, token_embedding_dim, batch_size, n=10000.0):
    # reference: https://pub.aimind.so/creating-sinusoidal-positional-embedding-from-scratch-in-pytorch-98c49e153d6
    if token_embedding_dim % 2 != 0:
        raise ValueError("Sinusoidal positional embedding cannot apply to odd token embedding dim (got dim={:d})".format(token_embedding_dim))
    indices = indices[0]
    positions = []
    for i in range(batch_size):
        positions.append([indice[i].item() for indice in indices])
    positions = torch.FloatTensor(positions).unsqueeze_(2) # [batch_size, sequence_len, 1]
    embeddings = torch.zeros(batch_size, token_sequence_size, token_embedding_dim) # [batch_size, sequence_len, embedding_dim]
    denominators = torch.pow(n, 2 * torch.arange(0, token_embedding_dim // 2) / token_embedding_dim) # 10000^(2i/d_model), i is the index of embedding --> [384]
    embeddings[:, :, 0::2] = torch.sin(positions / denominators) # sin(pos/10000^(2i/d_model)) # [batch_size, sequence_len, 384]
    embeddings[:, :, 1::2] = torch.cos(positions / denominators) # cos(pos/10000^(2i/d_model)) # [batch_size, sequence_len, 384]
    return embeddings

    
class ViFiCLIP(nn.Module):
    def __init__(self, clip_model, freeze_text_encoder, fusion_layers=None):
        super().__init__()
        self.clip_model = clip_model
        self.fusion_layers = fusion_layers if fusion_layers is not None else [-2]
        if freeze_text_encoder:
            for name, param in self.clip_model.named_parameters():
                if "text_model" in name:
                    param.requires_grad_(False)

    def forward(self, tactile_frames, texts, attention_masks, all_indices):
        # video
        b, l, c, h, w = tactile_frames.shape # (b, l, c, h, w)
        tactile_frames = tactile_frames.reshape(b * l, c, h, w) # (b * l, c, h, w)
        vision_outputs = self.clip_model.vision_model(tactile_frames, output_hidden_states=True)
        pooled_output = vision_outputs.hidden_states[-2][:, 0].to(tactile_frames.dtype)
        pooled_output = pooled_output.reshape(b, l, pooled_output.shape[-1]) # (b, l, d)
        feat_mean = pooled_output.mean(dim=1)  # (b, d)
        if texts is None:
            return feat_mean, None, None, None
        # contrastive mode: normalised mean only
        video_features = feat_mean / feat_mean.norm(p=2, dim=-1, keepdim=True)
        if texts is not None:
            # text
            text_outputs = self.clip_model.text_model(texts, attention_mask=attention_masks)
            pooled_output = text_outputs[1]
            text_features = self.clip_model.text_projection(pooled_output)
            text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
            # get logits
            logit_scale = self.clip_model.logit_scale.exp()
            cosine_sim = torch.matmul(text_features, video_features.t())
            logits_per_text = cosine_sim * logit_scale
            logits_per_image = logits_per_text.t()
        else:
            text_features = None
            logits_per_text = None
            logits_per_image = None
        return video_features, text_features, logits_per_image, logits_per_text


class MultimodalLLMForCausalLM(nn.Module):
    def __init__(self, tokenizer, clip_model, encoder_output_size, cutoff_len, llm, device):
        super(MultimodalLLMForCausalLM, self).__init__()
        self.tokenizer = tokenizer
        self.cutoff_len = cutoff_len
        self.device = device
        self.llm_embedding_size = llm.model.embed_tokens.weight.shape[1]
        self.encoder = CLIPTactileEncoder(clip_model=clip_model)
        self.encoder_output_size = encoder_output_size
        self.project = nn.Sequential(
            nn.Linear(encoder_output_size, self.llm_embedding_size),
            nn.GELU(),
            nn.LayerNorm(self.llm_embedding_size),
            nn.Linear(self.llm_embedding_size, self.llm_embedding_size),
        )

    def get_dummy_token(self, answer_embeds, question_embeds_len):
        batch_size = answer_embeds.shape[0]
        answer_embeds_len = answer_embeds.shape[1]
        index_shift = 0
        # labels are shifted by -1 inside the LlamaForCausalLM source code so tokens < n predict n
        pre_label_token = torch.full((batch_size, question_embeds_len + index_shift), fill_value=-100, dtype=torch.int64, device=self.device)
        post_label_token = torch.full((batch_size, self.cutoff_len - (question_embeds_len + answer_embeds_len + index_shift)), fill_value=-100, dtype=torch.int64, device=self.device)
        return pre_label_token, post_label_token

    def forward(self, question, tactile_frames, answer_tokens, all_indices, images=None, conclusion_start=None):
        # 1) question embeds
        question_embeds = []
        img_token_count = 0
        
        # Determine the device where the LLM input embeddings reside
        # self.llm.device might not be reliable if using device_map
        if hasattr(self.llm, "get_input_embeddings"):
            llm_device = self.llm.get_input_embeddings().weight.device
        elif hasattr(self.llm, "model") and hasattr(self.llm.model, "embed_tokens"):
             llm_device = self.llm.model.embed_tokens.weight.device
        else:
            llm_device = self.device

        for i, chunk in enumerate(question):
            chunk = chunk[0]
            if "img_tokens" in chunk:
                if i == 0:
                    # if the first chunk is an image, we need to add a BOS token
                    bos_token = torch.tensor([self.tokenizer.bos_token_id], dtype=torch.int64).to(llm_device)
                    bos_embed = self.llm.get_input_embeddings()(bos_token)
                    bos_embed = torch.unsqueeze(bos_embed, dim=0)
                    question_embeds.append(bos_embed)
                visual_embeds = self.encoder(tactile_frames[img_token_count].to(self.device))
                # idx = [all_indices[img_token_count]]
                # sinusoidal_embeds = sinusoidal_positional_embedding(token_sequence_size=5, indices=idx, token_embedding_dim=self.encoder_output_size, batch_size=visual_embeds.shape[0]).to(visual_embeds.device)
                # chunk_embeds = self.project(visual_embeds + sinusoidal_embeds)
                chunk_embeds = self.project(visual_embeds)
                # Move visual embeddings to LLM device
                chunk_embeds = chunk_embeds.to(llm_device)
                img_token_count += 1
            else:
                if i == 0:
                    chunk_embeds = self.llm.get_input_embeddings()(torch.tensor(self.tokenizer.encode(chunk), dtype=torch.int64).to(llm_device))
                else:
                    chunk_embeds = self.llm.get_input_embeddings()(torch.tensor(self.tokenizer.encode(chunk), dtype=torch.int64)[1:].to(llm_device))
                chunk_embeds = torch.unsqueeze(chunk_embeds, dim=0)
            question_embeds.append(chunk_embeds)
        question_embeds = torch.cat(question_embeds, dim=1)
        # 2) answer embeds
        answer_tokens = answer_tokens.to(llm_device)
        answer_embeds = self.llm.get_input_embeddings()(answer_tokens)
        full_embeds_len = question_embeds.shape[1] + answer_embeds.shape[1]
        question_embeds_len = question_embeds.shape[1]
        batch_size = question_embeds.shape[0]
        if full_embeds_len > self.cutoff_len:
            raise RuntimeError(f"Sample exceeds cutoff_len: full={full_embeds_len} > cutoff={self.cutoff_len}")
        # NOTE: padding token embedding index is 0
        padding_embeds = self.llm.get_input_embeddings()(torch.zeros(batch_size, self.cutoff_len - full_embeds_len, device=llm_device, dtype=torch.int64))
        # 3) combine embeds. Cast to LLM dtype because projected visual embeds are fp32
        # while embed_tokens may be bf16/fp16.
        llm_dtype = self.llm.get_input_embeddings().weight.dtype
        input_embeds = torch.cat((question_embeds, answer_embeds, padding_embeds), dim=1).to(llm_dtype)
        pre_label_dummy_token, post_label_dummy_token = self.get_dummy_token(answer_embeds, question_embeds_len)
        # Outcome supervision: mask description tokens, only backprop through conclusion.
        # conclusion_start=0 means no mask (OPD has no "Conclusion:" so full loss applies).
        if conclusion_start is not None:
            cs = int(conclusion_start[0].item() if torch.is_tensor(conclusion_start) else conclusion_start)
            if cs > 0:
                answer_labels = answer_tokens.clone()
                answer_labels[:, :cs] = -100
            else:
                answer_labels = answer_tokens
        else:
            answer_labels = answer_tokens
        labels = torch.cat((pre_label_dummy_token.to(llm_device), answer_labels, post_label_dummy_token.to(llm_device)), dim=1)
        batch_size = answer_embeds.shape[0]
        attention_mask = torch.cat((torch.ones([batch_size, full_embeds_len]), torch.zeros([batch_size, padding_embeds.shape[1]])), dim=1).to(llm_device)
        seq_len = input_embeds.shape[1]
        position_ids = torch.arange(0, seq_len, dtype=torch.long, device=llm_device).unsqueeze(0).expand(batch_size, -1)
        raw_out = self.llm(inputs_embeds=input_embeds, attention_mask=attention_mask, position_ids=position_ids)
        logits = raw_out.logits.float()
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous().to(shift_logits.device)
        loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=-100)
        out = CausalLMOutputWithPast(loss=loss, logits=logits)
        return out, question_embeds