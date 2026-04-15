from __future__ import annotations

from typing import Optional

import torch

from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.speculative.eagle_info import EagleDraftInput, EagleVerifyInput
from sglang.srt.speculative.eagle_utils import build_tree_kernel_efficient
from sglang.srt.speculative.eagle_worker import EAGLEWorker
from sglang.srt.speculative.spec_utils import detect_nan


class PEagleWorker(EAGLEWorker):
    """Parallel-drafting worker for P-EAGLE checkpoints."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.topk != 1:
            raise ValueError("P-EAGLE runtime currently supports only topk=1.")
        if self.speculative_num_draft_tokens != self.speculative_num_steps + 1:
            raise ValueError(
                "P-EAGLE expects speculative_num_draft_tokens == speculative_num_steps + 1."
            )
        if self.server_args.enable_multi_layer_eagle:
            raise ValueError("P-EAGLE should not be launched with multi-layer EAGLE.")

        model = self.draft_model_runner.model
        if not hasattr(model, "get_parallel_draft_token_id"):
            raise RuntimeError(
                f"Draft model {type(model).__name__} is missing P-EAGLE helpers."
            )
        self.parallel_draft_token_id = model.get_parallel_draft_token_id()

    def _gather_last_per_request(
        self, flat_tensor: torch.Tensor, lengths: list[int] | torch.Tensor
    ) -> torch.Tensor:
        if flat_tensor.numel() == 0:
            return flat_tensor
        if isinstance(lengths, torch.Tensor):
            lengths = lengths.tolist()
        end_indices = torch.tensor(lengths, device=flat_tensor.device, dtype=torch.long)
        end_indices = torch.cumsum(end_indices, dim=0) - 1
        return flat_tensor[end_indices]

    def forward_draft_extend(
        self,
        batch,
        hidden_states: torch.Tensor,
        next_token_ids: torch.Tensor,
        seq_lens_cpu: Optional[torch.Tensor],
        mm_input_embeds: Optional[torch.Tensor] = None,
    ):
        last_hidden_states = self._gather_last_per_request(hidden_states, batch.extend_lens)
        super().forward_draft_extend(
            batch,
            hidden_states,
            next_token_ids,
            seq_lens_cpu,
            mm_input_embeds,
        )
        assert isinstance(batch.spec_info, EagleDraftInput)
        batch.spec_info.hidden_states = last_hidden_states
        batch.spec_info.verified_id = next_token_ids
        batch.spec_info.topk_p = None
        batch.spec_info.topk_index = None

    def forward_draft_extend_after_decode(self, batch):
        assert isinstance(batch.spec_info, EagleDraftInput)

        last_hidden_states = None
        last_token_ids = None
        if batch.spec_info.verified_id.numel() > 0:
            accepted_lengths = [value + 1 for value in batch.spec_info.accept_length_cpu]
            last_hidden_states = self._gather_last_per_request(
                batch.spec_info.hidden_states,
                accepted_lengths,
            )
            last_token_ids = self._gather_last_per_request(
                batch.spec_info.verified_id,
                accepted_lengths,
            )

        super().forward_draft_extend_after_decode(batch)

        if last_hidden_states is not None and last_token_ids is not None:
            assert isinstance(batch.spec_info, EagleDraftInput)
            batch.spec_info.hidden_states = last_hidden_states
            batch.spec_info.verified_id = last_token_ids
            batch.spec_info.topk_p = None
            batch.spec_info.topk_index = None

    def _forward_parallel_draft(
        self,
        batch,
        token_ids: torch.Tensor,
        hidden_states: torch.Tensor,
    ):
        batch_size = batch.batch_size()
        backup = {
            "forward_mode": batch.forward_mode,
            "input_ids": batch.input_ids,
            "spec_info": batch.spec_info,
            "extend_lens": getattr(batch, "extend_lens", None),
            "extend_num_tokens": getattr(batch, "extend_num_tokens", None),
            "prefix_lens": getattr(batch, "prefix_lens", None),
            "extend_logprob_start_lens": getattr(batch, "extend_logprob_start_lens", None),
            "return_logprob": batch.return_logprob,
            "return_hidden_states": batch.return_hidden_states,
        }

        try:
            parallel_spec_info = EagleDraftInput(
                hidden_states=hidden_states,
                num_tokens_per_req=self.speculative_num_steps,
                num_tokens_for_logprob_per_req=self.speculative_num_steps,
            )
            parallel_spec_info.capture_hidden_mode = CaptureHiddenMode.FULL
            parallel_spec_info.extend_seq_lens_cpu = [
                self.speculative_num_steps
            ] * batch_size

            batch.forward_mode = ForwardMode.DRAFT_EXTEND_V2
            batch.input_ids = token_ids
            batch.spec_info = parallel_spec_info
            batch.extend_lens = [self.speculative_num_steps] * batch_size
            batch.extend_num_tokens = token_ids.numel()
            batch.prefix_lens = batch.seq_lens_cpu.tolist()
            batch.extend_logprob_start_lens = [0] * batch_size
            batch.return_logprob = False
            batch.return_hidden_states = False

            model_worker_batch = batch.get_model_worker_batch()
            forward_batch = ForwardBatch.init_new(
                model_worker_batch, self.draft_model_runner
            )
            forward_batch.can_run_dp_cuda_graph = False
            if not forward_batch.forward_mode.is_idle():
                self.draft_model_runner.attn_backend.init_forward_metadata(forward_batch)
            logits_output = self.draft_model_runner.forward(
                forward_batch,
                skip_attn_backend_init=True,
            ).logits_output
            if self.enable_nan_detection:
                detect_nan(logits_output)
            return logits_output
        finally:
            batch.forward_mode = backup["forward_mode"]
            batch.input_ids = backup["input_ids"]
            batch.spec_info = backup["spec_info"]
            batch.extend_lens = backup["extend_lens"]
            batch.extend_num_tokens = backup["extend_num_tokens"]
            batch.prefix_lens = backup["prefix_lens"]
            batch.extend_logprob_start_lens = backup["extend_logprob_start_lens"]
            batch.return_logprob = backup["return_logprob"]
            batch.return_hidden_states = backup["return_hidden_states"]

    def _build_parallel_query_inputs(
        self,
        last_token_ids: torch.Tensor,
        last_hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = last_token_ids.shape[0]
        if self.speculative_num_steps == 1:
            return last_token_ids, last_hidden_states

        mask_tokens = torch.full(
            (batch_size, self.speculative_num_steps - 1),
            self.parallel_draft_token_id,
            dtype=last_token_ids.dtype,
            device=last_token_ids.device,
        )
        mask_hidden = self.draft_model_runner.model.get_parallel_draft_hidden_state()
        mask_hidden = mask_hidden.to(
            device=last_hidden_states.device,
            dtype=last_hidden_states.dtype,
        )
        mask_hidden = mask_hidden.expand(
            batch_size, self.speculative_num_steps - 1, last_hidden_states.shape[-1]
        )

        token_ids = torch.cat([last_token_ids.unsqueeze(1), mask_tokens], dim=1).reshape(
            -1
        )
        hidden_states = torch.cat(
            [last_hidden_states.unsqueeze(1), mask_hidden], dim=1
        ).reshape(-1, last_hidden_states.shape[-1])
        return token_ids, hidden_states

    def _build_linear_verify_input(
        self,
        batch,
        last_token_ids: torch.Tensor,
        predicted_tokens: torch.Tensor,
    ) -> EagleVerifyInput:
        batch_size = predicted_tokens.shape[0]
        query_count = predicted_tokens.shape[1]
        if query_count > 1:
            parent_list = torch.arange(
                query_count - 1,
                device=predicted_tokens.device,
                dtype=torch.long,
            ).unsqueeze(0)
            parent_list = parent_list.expand(batch_size, -1).contiguous()
        else:
            parent_list = torch.empty(
                batch_size, 0, device=predicted_tokens.device, dtype=torch.long
            )

        top_scores_index = torch.arange(
            query_count,
            device=predicted_tokens.device,
            dtype=torch.long,
        ).unsqueeze(0)
        top_scores_index = top_scores_index.expand(batch_size, -1).contiguous()

        (
            tree_mask,
            position,
            retrive_index,
            retrive_next_token,
            retrive_next_sibling,
            draft_tokens,
        ) = build_tree_kernel_efficient(
            last_token_ids,
            parent_list,
            top_scores_index,
            predicted_tokens,
            batch.seq_lens,
            batch.seq_lens_sum,
            self.topk,
            self.speculative_num_steps,
            self.speculative_num_draft_tokens,
        )

        return EagleVerifyInput(
            draft_token=draft_tokens,
            custom_mask=tree_mask,
            positions=position,
            retrive_index=retrive_index,
            retrive_next_token=retrive_next_token,
            retrive_next_sibling=retrive_next_sibling,
            retrive_cum_len=None,
            spec_steps=self.speculative_num_steps,
            topk=self.topk,
            draft_token_num=self.speculative_num_draft_tokens,
            capture_hidden_mode=CaptureHiddenMode.FULL,
            seq_lens_sum=batch.seq_lens_sum,
            seq_lens_cpu=batch.seq_lens_cpu,
        )

    def draft(self, batch):
        if batch.forward_mode.is_idle():
            self._draft_preprocess_idle(batch)
            return EagleVerifyInput.create_idle_input(
                self.topk,
                self.speculative_num_steps,
                self.speculative_num_draft_tokens,
            )

        self._draft_preprocess_decode(batch)

        draft_state = batch.spec_info
        assert isinstance(draft_state, EagleDraftInput)

        query_token_ids, query_hidden_states = self._build_parallel_query_inputs(
            draft_state.verified_id,
            draft_state.hidden_states,
        )
        logits_output = self._forward_parallel_draft(
            batch,
            query_token_ids,
            query_hidden_states,
        )

        batch_size = batch.batch_size()
        logits = logits_output.next_token_logits.view(
            batch_size, self.speculative_num_steps, -1
        )
        predicted_tokens = logits.argmax(dim=-1)
        if self.hot_token_id is not None:
            predicted_tokens = self.hot_token_id[predicted_tokens]

        return self._build_linear_verify_input(
            batch,
            draft_state.verified_id,
            predicted_tokens,
        )
