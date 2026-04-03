"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import logging

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
from torch.nn import functional as F
import os
from lavis.common.registry import registry
from lavis.models.base_model import all_gather_with_grad, concat_all_gather
from lavis.models.blip2_models.blip2 import (
    Blip2Base,
    compute_sim_matrix,
    disabled_train,
)
from lavis.models.blip_models.blip_outputs import BlipOutput, BlipOutputFeatures
from sklearn.cluster import KMeans
import random


def l2norm(X, dim=-1):
    """L2-normalize columns of X"""
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt()
    X = torch.div(X, norm)
    return X

def sinkhorn_log_domain_torch(p, q, C, Mask=None, reg= 0.03, niter=100, thresh=1e-4):

    def M(u, v):
        "Modified cost for logarithmic updates"
        M = (-C + torch.unsqueeze(u, 1) + torch.unsqueeze(v, 0)) / reg
        if Mask is not None:
            M[Mask == 0] = -1e6


        return M

    def lse(A):
        "log-sum-exp"
        max_A, _ = torch.max(A, dim=1, keepdims=True)
        return torch.log(torch.exp(A - max_A).sum(1, keepdims=True) + 1e-10) + max_A

    # Actual Sinkhorn loop ......................................................................
    u, v, err = 0. * p, 0. * q, 0.
    actual_nits = 0 

    for i in range(niter):
        u1 = u 
        u = reg * (torch.log(p) - lse(M(u, v)).squeeze()) + u
        v = reg * (torch.log(q) - lse(M(u, v).T).squeeze()) + v

        err = torch.sum(torch.abs(u - u1))

        actual_nits += 1
        if err < thresh:
            break
    U, V = u, v
    pi = torch.exp(M(U, V)) 

    return pi

def query_oriented_learning(S, S_no, temp):
    cos_sim = torch.sum(S * S_no, dim=1)/temp
    upper_loss = (cos_sim + (0.002)/temp).clamp(min=0)
    lower_loss = ((-0.007)/temp - cos_sim).clamp(min=0)
    loss = torch.mean(upper_loss + lower_loss)
    return loss

def l1norm(X, dim):
    """L1-normalize columns of X"""
    norm = torch.abs(X).sum(dim=dim, keepdim=True)
    X = torch.div(X, norm)
    return 

def fidelity_measure(x, boundary_estimation=15):
    x = x.diag() - boundary_estimation
    x[x < 0] = 0
    x = x / x.max()

    return -torch.pow(x, 2) * (x - 1) * (x + 1)

def info_nce(query, target):
    bs = query.size(0)
    targets = torch.linspace(0,  bs - 1, bs, dtype=int).to(query.device)
    temp = nn.Parameter(0.07 * torch.ones([]))
    x = torch.matmul(query,target).squeeze().to(query.device)
    sim_i2t,_ = x.max(-1)
    sim_i2t = sim_i2t / temp
    return F.cross_entropy(sim_i2t, targets)

@registry.register_model("ConeSep")
class ConeSep(Blip2Base):
    """
    BLIP2 first-stage model with Q-former and ViT.
    Supported model types:
        - pretrained: pretrained model with vit-g
        - pretrain_vitL: pretrained model with vit-large
        - coco: fintuned model on coco
    Usage:
        >>> from lavis.models import load_model
        >>> model = load_model("blip2", "pretrain")
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain": "configs/models/blip2/blip2_pretrain.yaml",
        "pretrain_vitL": "configs/models/blip2/blip2_pretrain_vitL.yaml",
        "coco": "configs/models/blip2/blip2_coco.yaml",
    }

    def __init__(
        self,
        vit_model="eva_clip_g",
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=True,
        num_query_token=32,
        cross_attention_freq=2,
        embed_dim=256,
        max_txt_len=32,
        
    ):
        super().__init__()

        self.tokenizer = self.init_tokenizer()
        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision
        )
        if freeze_vit:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            logging.info("freeze vision encoder")
        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, self.visual_encoder.num_features, cross_attention_freq
        )
        self.Qformer.resize_token_embeddings(len(self.tokenizer))
        state_dict = self.Qformer.state_dict()
        for name, param in self.Qformer.named_parameters():
            if "_query" in name:
                key_orig = name.replace("_query", "")
                param.data.copy_(state_dict[key_orig])

        self.vision_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)
        self.text_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)

        self.itm_head = nn.Linear(self.Qformer.config.hidden_size, 2)

        self.temp = 0.07

        self.max_txt_len = max_txt_len
        self.negative_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, self.Qformer.config.hidden_size)
        )
        self.negative_tokens.data.normal_(mean=0.0, std=self.Qformer.config.initializer_range)
    
        # prompt settings
        self.no_text = [
            'there is no scene containing',
            'the image does not show',
            'there is no instance of',
            'this picture lacks',
            'the image is devoid of',
            'the picture is absent of',
            'this scene lacks any',
            'no elements of',
            'nothing in this image depicts',
            'the image contains no',
            'there are no appearances of',
            'the scene does not contain',
            'no presence of',
            'the image shows no sign of',
            'cannot find any',
            'the scene is missing',
            'there is no appearance of',
            'the image has no',
            'no occurrence of',
            'completely absent of',
            'the scene is without',
            'does not include any',
            'zero instances of',
            'not showing any',
            'entirely lacking'
        ]

        self.kl_loss = nn.KLDivLoss(reduction="batchmean")


    def prompt_load(self):
        self.negative_tokens.data.copy_(self.query_tokens.data)


    def forward(self, samples, device, if_negative=False, omega = 0.1, target_forgetting_loss_weight = 10, target_oriented_weight = 1, query_oriented_weight = 10):
        image = samples["image"] 
        target = samples["target"]
        text = samples["text_input"]
        text_no = [random.choice(self.no_text) + " " + t for t in text]


        image_embeds = self.ln_vision(self.visual_encoder(image))
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )
        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
            self.device
        )
        text_tokens = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(image.device)
        
        attention_mask = torch.cat([query_atts, text_tokens.attention_mask], dim=1)
        fusion_output = self.Qformer.bert(
            text_tokens.input_ids,
            query_embeds=query_tokens,
            attention_mask=attention_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        # ================ hardest negative fusion
        text_no_tokens = self.tokenizer(
            text_no,
            padding="max_length",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(image.device)
        negative_tokens = self.negative_tokens.expand(image_embeds.shape[0], -1, -1)

        attention_mask_no = torch.cat([query_atts, text_no_tokens.attention_mask], dim=1)
        DNC_output = self.Qformer.bert(
            text_no_tokens.input_ids,
            query_embeds=negative_tokens,
            attention_mask=attention_mask_no,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        DNC_feats = F.normalize(
            self.text_proj(DNC_output.last_hidden_state[:, 32, :]), dim=-1
        )
        # ================== target image feature ===================###
        taregt_embeds = self.ln_vision(self.visual_encoder(target))
        target_atts = torch.ones(taregt_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )
        target_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=taregt_embeds,
            encoder_attention_mask=target_atts,
            use_cache=True,
            return_dict=True,
        )
        #Target fea
        target_feats = F.normalize(
            self.vision_proj(target_output.last_hidden_state), dim=-1
        )
        #fusion fea
        fusion_feats = F.normalize(
            self.text_proj(fusion_output.last_hidden_state[:, 32, :]), dim=-1
        )

        fusion_feats=fusion_feats.unsqueeze(1).unsqueeze(1)
        target_feats=target_feats.permute(0, 2, 1)


        # NBL
        if if_negative: #
            logits_text_per_text = torch.matmul(fusion_output.last_hidden_state[:, 32, :], DNC_output.last_hidden_state[:, 32, :].t())
            logits_no_text_per_no_text = torch.matmul(DNC_output.last_hidden_state[:, 32, :], fusion_output.last_hidden_state[:, 32, :].t())

            logits_text_per_text = (logits_text_per_text / self.temp).softmax(1)
            logits_no_text_per_no_text = (logits_no_text_per_no_text / self.temp).softmax(1)

            logits_fusion_per_target = torch.matmul(fusion_feats,target_feats).squeeze().to(fusion_feats.device).max(-1)[0]
            logits_fusion_per_no_target = torch.matmul(DNC_feats.unsqueeze(1).unsqueeze(1), target_feats).squeeze().to(fusion_feats.device).max(-1)[0]

            logits_fusion_per_target = (logits_fusion_per_target / self.temp).softmax(1)
            logits_fusion_per_no_target = (logits_fusion_per_no_target / self.temp).softmax(1)

            targe_oriented_learning_out = self.targe_oriented_learning(logits_fusion_per_target, logits_fusion_per_no_target)

            query_oriented_learning_ = query_oriented_learning(fusion_feats, DNC_feats, self.temp)
            loss_robust = self.robust_infoNCE(fusion_feats, target_feats)

            return {'loss_robust': loss_robust, 'targe_oriented_learning': targe_oriented_learning_out * target_oriented_weight, 'query_oriented_learning': query_oriented_learning_ * query_oriented_weight}



        device = fusion_feats.device
        # ================  compute boundary_estimationed
        with torch.no_grad():
            guass_image = torch.randn(fusion_feats.size(0), 3, 224, 224).to(fusion_feats.device)
            guass_text_list = [random.choice(text) for _ in range(len(text))]

            guass_text_tokens = self.tokenizer(
                guass_text_list,
                padding="max_length",
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(fusion_feats.device)
            guass_image_embeds = self.ln_vision(self.visual_encoder(guass_image))
            guass_image_atts = torch.ones(guass_image_embeds.size()[:-1], dtype=torch.long).to(fusion_feats.device)
            guass_query_tokens = self.query_tokens.expand(guass_image_embeds.shape[0], -1, -1)
            guass_query_atts = torch.ones(guass_query_tokens.size()[:-1], dtype=torch.long).to(fusion_feats.device)
            guass_attention_mask = torch.cat([guass_query_atts, guass_text_tokens.attention_mask], dim=1)

            guass_fusion_output = self.Qformer.bert(
                guass_text_tokens.input_ids,
                query_embeds=query_tokens,
                attention_mask=guass_attention_mask,
                encoder_hidden_states=guass_image_embeds,
                encoder_attention_mask=guass_image_atts,
                return_dict=True,
            )
            guass_feats = F.normalize(
                self.text_proj(guass_fusion_output.last_hidden_state[:, 32, :]), dim=-1
            )
            guass_feats=guass_feats.unsqueeze(1).unsqueeze(1)
            
            guass_target_image = torch.randn(fusion_feats.size(0), 3, 224, 224).to(fusion_feats.device)
            guass_target_embeds = self.ln_vision(self.visual_encoder(guass_target_image))
            guass_target_atts = torch.ones(guass_target_embeds.size()[:-1], dtype=torch.long).to(fusion_feats.device)
            guass_target_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=guass_target_embeds,
                encoder_attention_mask=guass_target_atts,
                use_cache=True,
                return_dict=True,
            )
            guass_target = F.normalize(
                self.vision_proj(guass_target_output.last_hidden_state), dim=-1
            ) 
            guass_target = guass_target.permute(0, 2, 1)  # [B, 256, 32]
            x_rand = torch.matmul(guass_feats,guass_target).squeeze().to(fusion_feats.device)
            sim, _ = x_rand.max(-1)
            sim = sim
            boundary_estimation = ((sim.mean() + sim.T.mean())/2).item()


        # GFQ
        # ===============  reweight
        x = torch.matmul(fusion_feats,target_feats).squeeze().to(fusion_feats.device)
        x, _ = x.max(-1)
        fidelity = fidelity_measure(x, boundary_estimation=boundary_estimation)


        # ================ OT ===================
        sims_per_img = (x / self.temp).softmax(1)
        x_no = torch.matmul(DNC_feats, target_feats).squeeze().to(fusion_feats.device)
        sims_paired_img_no, _ = x_no.max(-1)


        # preds NC
        with torch.no_grad():
            NC_num = int(omega * fidelity.size(0))
            _, batched_NC_ids = torch.topk(fidelity, NC_num, largest=False)
        

        # BTU
        with torch.no_grad():
            gamma = 0.7
            sims_matrix = torch.cat([sims_per_img, sims_paired_img_no], dim=1)
            cost_v = (1 - sims_matrix) 
            cost_v = cost_v / cost_v.max()
            cost_t = (1 - (sims_matrix).t()) 
            cost_t = cost_t / cost_t.max()

            p = torch.ones(sims_matrix.shape[0], device = device) / sims_matrix.shape[0]
            q = torch.ones(sims_matrix.shape[1], device = device) / sims_matrix.shape[1]

            M_v = torch.ones_like(cost_v, device= device, dtype=torch.int64)
            M_v[:, -1] = 0
            M_v[batched_NC_ids, batched_NC_ids] = 0
            M_v[batched_NC_ids, -1] = 1
            M_t = M_v.t()
            pi_v = sinkhorn_log_domain_torch(p, q, cost_v, Mask = M_v)
            pi_t = sinkhorn_log_domain_torch(q, p, cost_t, Mask = M_t)

            L_v = torch.zeros_like(M_v, device = device)
            L_v.diagonal().fill_(1)
            L_v[batched_NC_ids, batched_NC_ids] = 0
            L_v[batched_NC_ids, -1] = 1
            L_t= L_v.t()

            label_v2t = pi_v / (pi_v.sum(dim=1, keepdim=True) )
            label_v2t = gamma * label_v2t + (1 - gamma) * L_v

            label_t2v = pi_t / (pi_t.sum(dim=1, keepdim=True) )
            label_t2v = gamma * label_t2v + (1 - gamma) * L_t

        logits = torch.cat([sims_per_img, sims_paired_img_no], dim=1)
        log_probs_v2t = F.log_softmax(logits, dim=1)
        label_v2t = F.softmax(label_v2t, dim=1)
        loss_img = self.kl_loss(log_probs_v2t, label_v2t)

        label_t2v = F.softmax(label_t2v, dim=1)
        log_probs_t2v = F.log_softmax(logits.t(), dim=1)
        loss_txt = self.kl_loss(log_probs_t2v, label_t2v)
        loss_ul = (loss_img + loss_txt) / 2 #loss_img #

        query_oriented_learning_ = query_oriented_learning(fusion_feats, DNC_feats, self.temp)
        loss_robust = self.robust_infoNCE(fusion_feats, target_feats)

        return {'loss_ul': loss_ul * target_forgetting_loss_weight, 'query_oriented_learning':query_oriented_learning_ * query_oriented_weight, 'loss_robust':loss_robust}


    def targe_oriented_learning(self, sims_img_per_text, sims_img_per_no_text):
        N = sims_img_per_text.shape[0]
        labels_no = -torch.eye(N, device = 'cuda') + (1 - torch.eye(N, device = 'cuda'))
        loss_sig_n = -F.logsigmoid(labels_no * (sims_img_per_no_text)).sum() / N
        return loss_sig_n



    def robust_infoNCE(self,query, target, label=None):
        eps=1e-7
        bs = query.size(0)
        x = torch.matmul(query,target).squeeze().to(query.device)
        sim_i2t,_ = x.max(-1)
        i2t=(sim_i2t/ self.temp).softmax(1)
        i2t = torch.clamp(i2t, min=eps, max=1-eps)
        targets = torch.arange(query.shape[0]).long().cuda()
        if label is not None:
            clean_mask = label.to(bool)
            noise_mask = ~clean_mask
            i2t = i2t[clean_mask]
            targets = targets[clean_mask]

        mask = torch.ones_like(i2t).to(float).to(i2t.device)
        bs = i2t.shape[0]
        mask[torch.arange(bs), targets] = 0.   
        loss = (- ((1. - i2t).log() * mask).sum() / bs + (- ((1. - i2t.T).log() * mask).sum() / bs)) / 2
        return loss

    @torch.no_grad()
    def extract_retrieval_compose(self, img, mod, return_attns=False):
        with self.maybe_autocast():
            image_embeds_frozen = self.ln_vision(self.visual_encoder(img))
        image_embeds_frozen = image_embeds_frozen.float()

        # return image_embeds
        reference_embeds = image_embeds_frozen

        image_atts = torch.ones(reference_embeds.size()[:-1], dtype=torch.long).to(
            reference_embeds.device
        )
        # query tokens
        query_tokens = self.query_tokens.expand(reference_embeds.shape[0], -1, -1)
        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
            self.device
        )
        # text tokens
        text_tokens = self.tokenizer(
            mod,
            padding="max_length",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(reference_embeds.device)

        attention_mask = torch.cat([query_atts, text_tokens.attention_mask], dim=1)
        fusion_output = self.Qformer.bert(
            text_tokens.input_ids,
            query_embeds=query_tokens,
            attention_mask=attention_mask,
            encoder_hidden_states=reference_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
            output_attentions=return_attns
        )

        fusion_feats = F.normalize(
            self.text_proj(fusion_output.last_hidden_state[:, 32, :]), dim=-1
        )

        return fusion_feats.unsqueeze(1).unsqueeze(1)


    def extract_retrieval_target(self, img):
        with self.maybe_autocast():
            image_embeds_frozen = self.ln_vision(self.visual_encoder(img))
        image_embeds_frozen = image_embeds_frozen.float()
        image_atts = torch.ones(
            image_embeds_frozen.size()[:-1], dtype=torch.long
        ).to(self.device)
        query_tokens = self.query_tokens.expand(
            image_embeds_frozen.shape[0], -1, -1
        )

        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds_frozen,
            encoder_attention_mask=image_atts,
            return_dict=True,
            output_attentions=True
        )
        image_embeds = query_output.last_hidden_state
        image_features = F.normalize(self.vision_proj(image_embeds), dim=-1)
        return image_features.permute(0, 2, 1)


    @torch.no_grad()
    def extract_target_features(self, image, mode='mean'):
        with self.maybe_autocast():
            image_embeds_frozen = self.ln_vision(self.visual_encoder(image))
        image_embeds_frozen = image_embeds_frozen.float()
        image_atts = torch.ones(
            image_embeds_frozen.size()[:-1], dtype=torch.long
        ).to(self.device)
        query_tokens = self.query_tokens.expand(
            image_embeds_frozen.shape[0], -1, -1
        )

        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds_frozen,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        image_embeds = query_output.last_hidden_state

        # return image_embeds
        image_features = F.normalize(self.vision_proj(image_embeds), dim=-1)
        return image_features, image_embeds_frozen



    @classmethod
    def from_config(cls, cfg):
        vit_model = cfg.get("vit_model", "eva_clip_g")
        img_size = cfg.get("image_size")
        num_query_token = cfg.get("num_query_token")
        cross_attention_freq = cfg.get("cross_attention_freq", 2)

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)

        max_txt_len = cfg.get("max_txt_len", 32)

        model = cls(
            vit_model=vit_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            num_query_token=num_query_token,
            cross_attention_freq=cross_attention_freq,
            max_txt_len=max_txt_len,
        )
        model.load_checkpoint_from_config(cfg)

        return model

    def compute_sim_matrix(self, data_loader, task_cfg):
        """
        Compute similarity i2t, t2i matrix for the given data loader.
        """
        k_test = task_cfg.k_test

        return compute_sim_matrix(model=self, data_loader=data_loader, k_test=k_test)
