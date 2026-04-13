# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in HuggingFace Transformers.
# Portions of this code are adapted from:
#   - https://github.com/EleutherAI/gpt-neox (Apache License 2.0)
#   - https://github.com/huggingface/transformers (Apache License 2.0)
#   - https://github.com/SafeAILab/EAGLE (Apache License 2.0)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache

from specforge.core.eagle3_adapters import BackendAdapter, SdpaLikeAdapter, UspAdapter
from specforge.core.loss import LogSoftmaxLoss
from specforge.modeling.draft import Eagle3DraftModel
from specforge.utils import padding


class Eagle3Model(nn.Module):
    pass


class OnlineEagle3Model(Eagle3Model):
    """
    In sgl-spec, we implement offline/online training.
    Online training means we have the target hidden_states available during training.
    Eagle3 using test time training technique (TTT) to train the draft model.
    1. We first extract the hidden states from the target model.
    2. Then concatenate the hidden states from 3 aux layers (layer 1, layer num_layers//2, layer num_layers-4).
    3. We project the concatenated hidden states to the target hidden size. from (batch, seq_len, 3*hidden_size) to (batch, seq_len, hidden_size)
    4. We concat the projected hidden states and embedding output as the input for the draft model.
    5. finally, we run TTT to train the draft model. input size is (batch, seq_len, hidden_size * 2)
    """

    def __init__(
        self,
        draft_model: Eagle3DraftModel,
        length: int = 7,
        attention_backend="sdpa",
        target_model: Optional[Eagle3Model] = None,
    ):
        """
        Args:
            target_model: the target model to extract hidden states.
            draft_model: the draft model to be trained.
            length: TTT length, it means how many turns to unroll during TTT.
        """
        super().__init__()
        self.draft_model = draft_model
        self.length = length
        self.attention_backend = attention_backend
        self.target_model = target_model

    def _make_adapter(self) -> BackendAdapter:
        if self.attention_backend == "usp":
            return UspAdapter(self)
        return SdpaLikeAdapter(self)

    def _acc_and_loss(
        self,
        *,
        logits: torch.Tensor,
        target_p: torch.Tensor,
        position_mask: torch.Tensor,
        loss_mask: torch.Tensor,
        adapter: BackendAdapter,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            local_correct = (
                (logits.argmax(-1) == target_p.argmax(-1)) * position_mask.squeeze(-1)
            ).sum()
            local_denom = loss_mask.sum().clamp_min(1e-6)
            local_correct, local_denom = adapter.reduce_metrics(
                local_correct=local_correct, local_denom=local_denom
            )
            acc = local_correct / local_denom

        loss = LogSoftmaxLoss.apply(logits, target_p, position_mask)
        loss = adapter.reduce_loss(loss)
        return acc, loss

    def _prepare_position_ids(
        self,
        position_ids: Optional[torch.Tensor],
        *,
        seq_length: int,
        past_key_values_length: int,
        device: torch.device,
        is_vlm: bool,
        input_ids: torch.Tensor,
        image_grid_thw: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if self.attention_backend == "usp":
            return position_ids
        if position_ids is None:
            if is_vlm:
                mrope_positions_ids, _ = self.target_model.get_rope_index(
                    input_ids=input_ids, image_grid_thw=image_grid_thw
                )
                return mrope_positions_ids
            return (
                torch.arange(
                    past_key_values_length,
                    seq_length + past_key_values_length,
                    dtype=torch.long,
                    device=device,
                )
                .unsqueeze(0)
                .view(-1, seq_length)
            )

        position_ids = position_ids.long()
        return position_ids.view(-1, seq_length)

    def _sample_bita_window(
        self,
        valid_anchor_mask: torch.Tensor,
        max_groups: int,
        strategy: str,
    ) -> Tuple[int, int]:
        valid_positions = torch.nonzero(valid_anchor_mask, as_tuple=False).view(-1)
        if valid_positions.numel() == 0:
            return 0, 0

        runs = []
        run_start = int(valid_positions[0].item())
        prev = run_start
        for pos in valid_positions[1:]:
            pos = int(pos.item())
            if pos != prev + 1:
                runs.append((run_start, prev))
                run_start = pos
            prev = pos
        runs.append((run_start, prev))

        strategy = strategy.lower()
        if strategy == "tail":
            run_start, run_end = runs[-1]
            start_anchor = max(run_end - max_groups + 1, run_start)
        else:
            run_idx = int(
                torch.randint(len(runs), (1,), device=valid_anchor_mask.device).item()
            )
            run_start, run_end = runs[run_idx]
            start_high = max(run_start, run_end - max_groups + 1)
            if start_high == run_start:
                start_anchor = run_start
            else:
                start_anchor = int(
                    torch.randint(
                        run_start,
                        start_high + 1,
                        (1,),
                        device=valid_anchor_mask.device,
                    ).item()
                )

        groups = min(max_groups, run_end - start_anchor + 1)
        freeze_num = start_anchor + groups
        return freeze_num, groups

    def _run_bita_auxiliary(
        self,
        *,
        input_ids: torch.Tensor,
        loss_mask: torch.Tensor,
        hidden_states: torch.Tensor,
        target_p_padded: torch.Tensor,
        position_mask: torch.Tensor,
        adapter: BackendAdapter,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if not getattr(self.draft_model, "supports_bita_training", False):
            return None, None

        mask_num = int(getattr(self.draft_model, "bita_mask_num", 0) or 0)
        max_groups = int(getattr(self.draft_model, "bita_max_groups", 1) or 1)
        strategy = str(getattr(self.draft_model, "bita_window_strategy", "random"))
        if mask_num <= 0 or max_groups <= 0:
            return None, None

        batch_size, seq_length, hidden_size = hidden_states.shape
        draft_vocab_size = target_p_padded.shape[-1]
        device = hidden_states.device
        dtype = hidden_states.dtype
        target_valid = position_mask.squeeze(-1).bool()

        sample_meta = []
        has_valid_sample = False
        max_total_len = 1
        for batch_idx in range(batch_size):
            valid_anchor_mask = torch.zeros(seq_length, dtype=torch.bool, device=device)
            for anchor in range(max(seq_length - mask_num, 0)):
                valid_anchor_mask[anchor] = target_valid[
                    batch_idx, anchor + 1 : anchor + 1 + mask_num
                ].all()

            freeze_num, groups = self._sample_bita_window(
                valid_anchor_mask=valid_anchor_mask,
                max_groups=max_groups,
                strategy=strategy,
            )
            prefix_len = max(freeze_num, 1)
            total_len = prefix_len + groups * mask_num
            max_total_len = max(max_total_len, total_len)
            has_valid_sample = has_valid_sample or groups > 0
            sample_meta.append(
                {
                    "freeze_num": freeze_num,
                    "groups": groups,
                    "prefix_len": prefix_len,
                    "total_len": total_len,
                }
            )

        if not has_valid_sample:
            return None, None

        aux_input_embeds = torch.zeros(
            batch_size, max_total_len, hidden_size, device=device, dtype=dtype
        )
        aux_hidden_states = torch.zeros_like(aux_input_embeds)
        aux_valid_mask = torch.zeros(
            batch_size, max_total_len, device=device, dtype=torch.bool
        )
        aux_target_p = torch.zeros(
            batch_size,
            max_total_len,
            draft_vocab_size,
            device=device,
            dtype=target_p_padded.dtype,
        )
        aux_position_mask = torch.zeros(
            batch_size, max_total_len, 1, device=device, dtype=position_mask.dtype
        )

        for batch_idx, meta in enumerate(sample_meta):
            prefix_len = meta["prefix_len"]
            total_len = meta["total_len"]
            groups = meta["groups"]
            freeze_num = meta["freeze_num"]
            aux_valid_mask[batch_idx, :total_len] = True

            prefix_embeds = self.draft_model.embed_input_ids(
                input_ids[batch_idx : batch_idx + 1, :prefix_len]
            ).to(dtype=dtype)
            aux_input_embeds[batch_idx, :prefix_len] = prefix_embeds[0]
            aux_hidden_states[batch_idx, :prefix_len] = hidden_states[
                batch_idx, :prefix_len
            ]

            if groups <= 0:
                continue

            mask_embeds = self.draft_model.get_bita_mask_embeddings(
                groups=groups,
                device=device,
                dtype=dtype,
            )
            aux_input_embeds[
                batch_idx, prefix_len : prefix_len + groups * mask_num
            ] = mask_embeds

            for group_idx in range(groups):
                anchor = freeze_num - groups + group_idx
                for mask_idx in range(mask_num):
                    row_idx = prefix_len + group_idx * mask_num + mask_idx
                    target_idx = anchor + 1 + mask_idx
                    aux_target_p[batch_idx, row_idx] = target_p_padded[
                        batch_idx, target_idx
                    ]
                    aux_position_mask[batch_idx, row_idx] = position_mask[
                        batch_idx, target_idx
                    ]

        aux_attention_mask = self.draft_model.prepare_decoder_attention_mask(
            attention_mask=aux_valid_mask,
            hidden_states=aux_hidden_states,
            batch_size=batch_size,
            seq_length=max_total_len,
            past_key_values_length=0,
        )

        if aux_attention_mask is None:
            aux_attention_mask = torch.zeros(
                batch_size,
                1,
                max_total_len,
                max_total_len,
                device=device,
                dtype=dtype,
            )

        for batch_idx, meta in enumerate(sample_meta):
            freeze_num = meta["freeze_num"]
            groups = meta["groups"]
            if groups <= 0:
                continue

            for group_idx in range(groups):
                start_idx = freeze_num + group_idx * mask_num
                hidden_prefix_start = freeze_num - groups + group_idx + 1
                aux_attention_mask[
                    batch_idx,
                    0,
                    start_idx : start_idx + mask_num,
                    hidden_prefix_start:freeze_num,
                ] = torch.finfo(aux_attention_mask.dtype).min
                if group_idx > 0:
                    aux_attention_mask[
                        batch_idx,
                        0,
                        start_idx:,
                        start_idx - mask_num : start_idx,
                    ] = torch.finfo(aux_attention_mask.dtype).min

        aux_position_ids = (aux_attention_mask == 0).sum(dim=-1).squeeze(1).long() - 1

        prompt_key_values = self.draft_model.get_bita_prompt_key_values(
            batch_size=batch_size,
            device=device,
            dtype=dtype,
        )
        if prompt_key_values is not None:
            prompt_len = prompt_key_values[0].shape[-2]
            prompt_attention_mask = torch.full(
                (batch_size, 1, max_total_len, prompt_len),
                torch.finfo(aux_attention_mask.dtype).min,
                device=device,
                dtype=aux_attention_mask.dtype,
            )
            for batch_idx, meta in enumerate(sample_meta):
                if meta["groups"] <= 0:
                    continue
                prompt_attention_mask[
                    batch_idx, 0, meta["freeze_num"] : meta["total_len"]
                ] = 0
            aux_attention_mask = torch.cat(
                [prompt_attention_mask, aux_attention_mask], dim=-1
            )

        aux_hidden_out = self.draft_model.backbone(
            input_embeds=aux_input_embeds,
            hidden_states=aux_hidden_states,
            cache_hidden=None,
            attention_mask=aux_attention_mask,
            position_ids=aux_position_ids,
            past_key_values=None,
            prompt_key_values=prompt_key_values,
            use_cache=False,
        )
        aux_logits = self.draft_model.compute_logits(aux_hidden_out)
        aux_loss_mask = aux_position_mask.squeeze(-1)
        aux_acc, aux_loss = self._acc_and_loss(
            logits=aux_logits,
            target_p=aux_target_p,
            position_mask=aux_position_mask,
            loss_mask=aux_loss_mask,
            adapter=adapter,
        )
        return aux_loss, aux_acc

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        target: torch.Tensor,
        loss_mask: torch.Tensor,
        hidden_states: torch.Tensor,
        past_key_values: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        position_ids: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        is_vlm: bool = False,
        **kwargs,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """
        Online eagle model trainer, modified from: https://github.com/SafeAILab/EAGLE/blob/main/eagle/traineagle3/cnets.py#L711

        Args:
            input_ids: (batch, seq_len)
            attention_mask: (batch, seq_len)
            loss_mask: (batch, seq_len)
            past_key_values: We dont use this past_key_values in eagle3, but keep it for compatibility. We control kvcache by cache_hidden.
            position_ids: (batch, seq_len)
        """
        # Step 1: handle vocab size
        target_p_padded, position_mask = _compute_target_p_padded(
            target=target,
            t2d=self.draft_model.t2d,
            loss_mask=loss_mask,
            length=self.length,
        )
        del target
        torch.cuda.empty_cache()

        # basic info
        batch_size, seq_length, _ = hidden_states.shape
        seq_length_with_past = seq_length
        past_key_values_length = 0

        # Step 2: project the concatenated hidden states to the target hidden size
        hidden_states = self.draft_model.project_hidden_states(hidden_states)

        # Step 3: process kv cache, position ids and position ids
        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length
        position_ids = self._prepare_position_ids(
            position_ids=position_ids,
            seq_length=seq_length,
            past_key_values_length=past_key_values_length,
            device=hidden_states.device,
            is_vlm=is_vlm,
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
        )

        # Step 4: handle attention mask
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past),
                dtype=torch.bool,
                device=hidden_states.device,
            )
        if self.attention_backend == "sdpa":
            attention_mask = self.draft_model.prepare_decoder_attention_mask(
                attention_mask=attention_mask,
                hidden_states=hidden_states,
                batch_size=batch_size,
                seq_length=seq_length,
                past_key_values_length=past_key_values_length,
            )

        # Step 5: run TTT
        plosses = []
        vlosses = []
        acces = []
        adapter = self._make_adapter()
        base_input_ids = input_ids
        base_loss_mask = loss_mask
        base_hidden_states = hidden_states
        base_position_mask = position_mask
        # for sequence paralle, position mask and input ids will split by sequence dim, need to keep origin for ttt shift
        global_input_ids = input_ids
        if self.attention_backend in ["sdpa", "fa", "usp"]:
            cache_hidden = [[], []]
            past_key_values = None
        elif self.attention_backend == "flex_attention":
            cache_hidden = None
            past_key_values = DynamicCache()
        else:
            raise ValueError(f"Unknown attention backend: {self.attention_backend}")

        for idx in range(self.length):
            state = adapter.step_view(
                idx=idx,
                ttt_length=self.length,
                global_input_ids=global_input_ids,
                attention_mask=attention_mask,
                loss_mask=loss_mask,
                position_ids=position_ids,
                hidden_states=hidden_states,
                target_p_padded=target_p_padded,
                position_mask=position_mask,
                seq_length=seq_length,
            )
            is_last = idx == self.length - 1

            # Step 5.1: embed the input ids
            inputs_embeds = self.draft_model.embed_input_ids(state.input_ids)
            inputs_embeds = inputs_embeds.to(hidden_states.dtype)

            # Step 5.2: run the draft model backbone
            hidden_states_out = self.draft_model.backbone(
                input_embeds=inputs_embeds,
                hidden_states=state.hidden_states,
                cache_hidden=cache_hidden,
                attention_mask=state.attention_mask,
                position_ids=state.position_ids,
                past_key_values=past_key_values,
                use_cache=True,
            )

            # update hidden states for next step
            hidden_states = hidden_states_out

            # Step 5.4: get logits
            logits = self.draft_model.compute_logits(hidden_states)

            # Step 5.5 + 5.6: metric and loss
            acc, loss = self._acc_and_loss(
                logits=logits,
                target_p=state.target_p,
                position_mask=state.position_mask,
                loss_mask=state.loss_mask,
                adapter=adapter,
            )
            acces.append(acc)
            plosses.append(loss)

            if not is_last:
                # Step 5.7: we need to update the loss mask
                global_input_ids = padding(global_input_ids, left=False)
                position_mask = padding(position_mask, left=False)
                loss_mask = padding(loss_mask, left=False)
                # Flex attention mask shirnking is handled inside attention module

        bita_loss, bita_acc = self._run_bita_auxiliary(
            input_ids=base_input_ids,
            loss_mask=base_loss_mask,
            hidden_states=base_hidden_states,
            target_p_padded=target_p_padded,
            position_mask=base_position_mask,
            adapter=adapter,
        )
        if bita_loss is not None:
            plosses[0] = (
                plosses[0]
                + getattr(self.draft_model, "bita_loss_weight", 1.0) * bita_loss
            )
            vlosses = [bita_loss.detach(), bita_acc.detach()]
        return plosses, vlosses, acces


class QwenVLOnlineEagle3Model(Eagle3Model):
    """
    In sgl-spec, we implement offline/online training.
    Online training means we have the target hidden_states available during training.
    Eagle3 using test time training technique (TTT) to train the draft model.
    1. We first extract the hidden states from the target model.
    2. Then concatenate the hidden states from 3 aux layers (layer 1, layer num_layers//2, layer num_layers-4).
    3. We project the concatenated hidden states to the target hidden size. from (batch, seq_len, 3*hidden_size) to (batch, seq_len, hidden_size)
    4. We concat the projected hidden states and embedding output as the input for the draft model.
    5. finally, we run TTT to train the draft model. input size is (batch, seq_len, hidden_size * 2)
    """

    def __init__(
        self,
        target_model,
        draft_model: Eagle3DraftModel,
        processor,
        length: int = 7,
        attention_backend: str = "sdpa",
    ):
        """
        Args:
            target_model: the target model to extract hidden states.
            draft_model: the draft model to be trained.
            length: TTT length, it means how many turns to unroll during TTT.
        """
        super().__init__()
        self.target_model = target_model
        self.draft_model = draft_model
        self.processor = processor
        self.length = length
        self.attention_backend = attention_backend

    @torch.no_grad()
    def _prepare_data(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        loss_mask: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        modified from: https://github.com/SafeAILab/EAGLE/blob/main/eagle/traineagle3/cnets.py#L692
        Extract the hidden states from the target model outputs.

        Args:
            input_ids: (batch, seq_len)
            attention_mask: (batch, seq_len)
            loss_mask: (batch, seq_len)
            device: the device to run the target model, if None, use the input_ids device
            pixel_values: image pixel values, used for VLM models
            image_grid_thw: image grid thw, used for VLM models

        Returns:
            hidden_states: (batch, seq_len, 3*hidden_size)
            target: (batch, seq_len, vocab_size)
            loss_mask: (batch, seq_len)
            input_ids: (batch, seq_len)
        """

        if device is None:
            device = input_ids.device

        # run the target model to get the hidden states
        outputs = self.target_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            output_hidden_states=True,
            use_cache=False,
        )

        # extract the aux hidden states
        # output_hidden_states = True will return the embedding output as well
        # so we have an offset of 1
        num_hidden_states = len(outputs.hidden_states)
        offset = 1
        num_layers = num_hidden_states - 1

        # Eagle3 uses 3 aux layers from layer 1, num_layers//2, num_layers-4
        low_aux_layer = 1 + offset
        mid_aux_layer = num_layers // 2 - 1 + offset
        last_aux_layer = num_layers - 4 + offset

        hidden_states0 = outputs.hidden_states[low_aux_layer]
        hidden_states1 = outputs.hidden_states[mid_aux_layer]
        hidden_states2 = outputs.hidden_states[last_aux_layer]

        hidden_states = torch.cat(
            (hidden_states0, hidden_states1, hidden_states2), dim=-1
        )

        # apply pading
        target = outputs.logits
        target = padding(target, left=False)
        input_ids = padding(input_ids, left=False)

        if target is not None:
            target = target.to(device)
            loss_mask = loss_mask[..., None]
            loss_mask = loss_mask.to(device)

        return hidden_states, target, loss_mask, input_ids

    @torch.no_grad()
    def _get_input_embeds(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        # get input embeding with image
        # inputs_embeds = self.target_model.model.get_input_embeddings()(input_ids)
        inputs_embeds = self.draft_model.embed_input_ids(input_ids)
        image_embeds = self.target_model.model.get_image_features(
            pixel_values, image_grid_thw
        )
        image_embeds = torch.cat(image_embeds, dim=0)
        n_image_tokens = (
            input_ids == self.target_model.model.config.image_token_id
        ).sum()
        n_image_features = image_embeds.shape[0]
        if n_image_tokens != n_image_features:
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
            )

        mask = input_ids == self.target_model.model.config.image_token_id
        mask_unsqueezed = mask.unsqueeze(-1)
        mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
        image_mask = mask_expanded.to(inputs_embeds.device)

        image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
        return inputs_embeds

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        loss_mask: torch.Tensor,
        past_key_values: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        position_ids: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """
        Online eagle model trainer, modified from: https://github.com/SafeAILab/EAGLE/blob/main/eagle/traineagle3/cnets.py#L711

        Args:
            input_ids: (batch, seq_len)
            attention_mask: (batch, seq_len)
            loss_mask: (batch, seq_len)
            past_key_values: We dont use this past_key_values in eagle3, but keep it for compatibility. We control kvcache by cache_hidden.
            position_ids: (batch, seq_len)
            pixel_values: batch image pixel values, used for VLM models
            image_grid_thw: (batch, 3), image grid thw, used for VLM models
        """
        # Step 0: prepare data with the target model
        hidden_states, target, loss_mask, input_ids = self._prepare_data(
            input_ids, attention_mask, loss_mask, pixel_values, image_grid_thw
        )

        # Step 1: handle vocab size
        target_p_padded, position_mask = _compute_target_p_padded(
            target=target,
            t2d=self.draft_model.t2d,
            loss_mask=loss_mask,
            length=self.length,
        )
        del target

        # basic info
        batch_size, seq_length, _ = hidden_states.shape
        seq_length_with_past = seq_length
        past_key_values_length = 0

        # Step 2: project the concatenated hidden states to the target hidden size
        hidden_states = self.draft_model.project_hidden_states(hidden_states)

        # Step 3: process kv cache, position ids and position ids
        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length

        if position_ids is None:
            attention_mask_tensor = (
                attention_mask
                if not isinstance(attention_mask, dict)
                else attention_mask["full_attention"]
            )
            if attention_mask_tensor is not None and attention_mask_tensor.ndim == 4:
                attention_mask_tensor = torch.diagonal(
                    attention_mask_tensor[:, 0], dim1=1, dim2=2
                )
                attention_mask_tensor = (
                    attention_mask_tensor / torch.finfo(attention_mask_tensor.dtype).min
                )
                attention_mask_tensor = (1.0 - attention_mask_tensor).int()

            position_ids, rope_deltas = self.target_model.model.get_rope_index(
                input_ids,
                image_grid_thw,
                None,
                second_per_grid_ts=None,
                attention_mask=attention_mask_tensor,
            )
            self.rope_deltas = rope_deltas
        else:
            position_ids = position_ids

        # Step 4: handle attention mask
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past),
                dtype=torch.bool,
                device=hidden_states.device,
            )
        if self.attention_backend == "sdpa":
            attention_mask = self.draft_model.prepare_decoder_attention_mask(
                attention_mask=attention_mask,
                hidden_states=hidden_states,
                batch_size=batch_size,
                seq_length=seq_length,
                past_key_values_length=past_key_values_length,
            )

        # Step 5: run TTT
        plosses = []
        vlosses = []
        acces = []
        if self.attention_backend in ["sdpa", "fa"]:
            cache_hidden = [[], []]
            past_key_values = None
        elif self.attention_backend == "flex_attention":
            cache_hidden = None
            past_key_values = DynamicCache()
        else:
            raise ValueError(f"Unknown attention backend: {self.attention_backend}")

        for idx in range(self.length):
            target_p = target_p_padded[:, idx : idx + seq_length, :].contiguous()
            is_last = idx == self.length - 1

            # Step 5.1: embed the input ids
            # inputs_embeds = self._get_input_embeds(input_ids, pixel_values, image_grid_thw)
            inputs_embeds = self.draft_model.embed_input_ids(input_ids)
            inputs_embeds = inputs_embeds.to(hidden_states.dtype)

            # Step 5.2: run the draft model backbone
            hidden_states_out = self.draft_model.backbone(
                input_embeds=inputs_embeds,
                hidden_states=hidden_states,
                cache_hidden=cache_hidden,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=True,
            )

            # update hidden states for next step
            hidden_states = hidden_states_out

            # Step 5.4: get logits
            logits = self.draft_model.compute_logits(hidden_states)

            # Step 5.5: record metrics first as we in-place modify logits
            with torch.no_grad():
                acces.append(
                    _compute_metric_acc(
                        logits=logits,
                        target_p=target_p,
                        position_mask=position_mask,
                        loss_mask=loss_mask,
                    )
                )

            # Step 5.6: calculate loss, in-place modifies logits!
            loss = LogSoftmaxLoss.apply(logits, target_p, position_mask)
            plosses.append(loss)

            if not is_last:
                # Step 5.7: we need to update the loss mask
                input_ids = padding(input_ids, left=False)
                position_mask = padding(position_mask, left=False)
                loss_mask = padding(loss_mask, left=False)
                # Flex attention mask shirnking is handled inside attention module
        return plosses, vlosses, acces


def _compute_target_p_padded(target, t2d, loss_mask, length):
    with torch.no_grad():
        target_p, position_mask = _compute_target_p(
            target=target,
            t2d=t2d,
            loss_mask=loss_mask,
        )

        assert len(target_p.shape) == 3
        target_p_padded = F.pad(
            target_p,
            pad=(0, 0, 0, length),
            mode="constant",
            # For bitwise equality with previous code
            value=1 / target_p.shape[-1],
        )

        return target_p_padded, position_mask


@torch.compile(dynamic=None)
def _compute_target_p(target, t2d, loss_mask):
    target_head = target
    target_max_token = target_head.argmax(-1)
    target_mask = t2d[target_max_token]
    target_mask = target_mask[..., None].int()
    position_mask = target_mask * loss_mask
    target_head = target_head[..., t2d]
    target_head = target_head.float()
    target_p = nn.Softmax(dim=2)(target_head)
    target_p = target_p.detach()
    return target_p, position_mask


@torch.compile(dynamic=None)
def _compute_metric_acc(logits, target_p, position_mask, loss_mask):
    return (
        (logits.argmax(-1) == target_p.argmax(-1)) * position_mask.squeeze(-1)
    ).sum() / loss_mask.sum().clamp_min(1e-6)
