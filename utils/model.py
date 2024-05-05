import torch
from torch import nn
from transformers import CLIPVisionModel
from utils.constants import *


class CLIPTactileEncoder(nn.Module):
    def __init__(self, clip_model):
        super(CLIPTactileEncoder, self).__init__()
        self.model = CLIPVisionModel.from_pretrained(clip_model)

    def forward(self, tactile_embeds):
        b, l, c, h, w = tactile_embeds.shape # (b, l, c, h, w)
        tactile_embeds = tactile_embeds.reshape(b * l, c, h, w) # (b * l, c, h, w)
        tactile_forward_outs = self.model(tactile_embeds, output_hidden_states=True)
        # pooled output
        tactile_features = tactile_forward_outs.hidden_states[-1][:, 0].to(tactile_embeds.dtype) # (b * l, patch_embed_size)
        _, patch_embed_size = tactile_features.shape
        tactile_features = tactile_features.reshape(b, l, patch_embed_size) # (b, l, patch_embed_size)
        return tactile_features
    

class CLIPClassifier(nn.Module):
    def __init__(self, output_size):
        super(CLIPClassifier, self).__init__()
        self.fc = nn.Linear(output_size, 512)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.hardness_fc = nn.Linear(512, len(list(HARDNESS_MAP.keys())))
        self.roughness_fc = nn.Linear(512, len(list(ROUGHNESS_MAP.keys())))
        self.texture_fc = nn.Linear(512, len(list(TEXTURE_MAP.keys())))

    def forward(self, vision_features):
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
    def __init__(self, clip_model, freeze_text_encoder):
        super().__init__()
        self.clip_model = clip_model
        if freeze_text_encoder:
            for name, param in self.clip_model.named_parameters():
                if "text_model" in name:
                    param.requires_grad_(False)

    def forward(self, tactile_frames, texts, attention_masks, all_indices):
        # video
        b, l, c, h, w = tactile_frames.shape # (b, l, c, h, w)
        tactile_frames = tactile_frames.reshape(b * l, c, h, w) # (b * l, c, h, w)
        vision_outputs = self.clip_model.vision_model(tactile_frames)
        pooled_output = vision_outputs.pooler_output # (b * l, patch_embed_size)
        _, patch_embed_size = pooled_output.shape
        pooled_output = pooled_output.reshape(b, l, patch_embed_size) # (b, l, patch_embed_size)
        # add sinusoidal positional embedding
        vision_features = pooled_output
        sinusoidal_embeds = sinusoidal_positional_embedding(token_sequence_size=5, indices=all_indices, token_embedding_dim=1024, batch_size=vision_features.shape[0]).to(vision_features.device)
        vision_features = vision_features + sinusoidal_embeds
        video_features = vision_features.mean(dim=1, keepdim=False)
        video_features = video_features / video_features.norm(p=2, dim=-1, keepdim=True)
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
    def __init__(self, tokenizer, clip_model, encoder_output_size, cutoff_len, llm, use_vqvae, device):
        super(MultimodalLLMForCausalLM, self).__init__()
        self.tokenizer = tokenizer
        self.cutoff_len = cutoff_len
        self.use_vqvae = use_vqvae
        self.device = device
        self.llm_embedding_size = llm.model.embed_tokens.weight.shape[1]
        self.encoder = CLIPTactileEncoder(clip_model=clip_model)
        self.project = nn.Sequential(
            nn.Linear(encoder_output_size, self.llm_embedding_size),
            nn.GELU(),
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

    def forward(self, question, tactile_frames, answer_tokens, all_indices, images=None):
        # 1) question embeds
        question_embeds = []
        img_token_count = 0
        for chunk in question:
            chunk = chunk[0]
            if "img_tokens" in chunk:
                visual_embeds = self.encoder(tactile_frames[img_token_count].to(self.device))
                idx = [all_indices[img_token_count]]
                sinusoidal_embeds = sinusoidal_positional_embedding(token_sequence_size=5, indices=idx, token_embedding_dim=1024, batch_size=visual_embeds.shape[0]).to(visual_embeds.device)
                chunk_embeds = self.project(visual_embeds + sinusoidal_embeds)
                img_token_count += 1
            else:
                chunk_embeds = self.llm.get_input_embeddings()(torch.tensor(self.tokenizer.encode(chunk), dtype=torch.int64)[1:].to(self.device))
                chunk_embeds = torch.unsqueeze(chunk_embeds, dim=0)
            question_embeds.append(chunk_embeds)
        question_embeds = torch.cat(question_embeds, dim=1)
        # 2) answer embeds
        answer_embeds = self.llm.get_input_embeddings()(answer_tokens)
        full_embeds_len = question_embeds.shape[1] + answer_embeds.shape[1]
        question_embeds_len = question_embeds.shape[1]
        batch_size = question_embeds.shape[0]
        # NOTE: padding token embedding index is 0
        padding_embeds = self.llm.get_input_embeddings()(torch.zeros(batch_size, self.cutoff_len - full_embeds_len, device=self.device, dtype=torch.int64))
        # 3) combine embeds
        input_embeds = torch.cat((question_embeds, answer_embeds, padding_embeds), dim=1)
        pre_label_dummy_token, post_label_dummy_token = self.get_dummy_token(answer_embeds, question_embeds_len)
        labels = torch.cat((pre_label_dummy_token, answer_tokens, post_label_dummy_token), dim=1)
        batch_size = answer_embeds.shape[0]
        attention_mask = torch.cat((torch.ones([batch_size, full_embeds_len]), torch.zeros([batch_size, padding_embeds.shape[1]])), dim=1).to(self.device)
        out = self.llm(inputs_embeds=input_embeds, labels=labels, attention_mask=attention_mask) # pass in embeddings directly: https://huggingface.co/docs/transformers/main/en/model_doc/llama
        return out, question_embeds