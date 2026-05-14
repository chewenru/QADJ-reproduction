import copy

import torch as th
from torch.optim import RMSprop

from components.episode_buffer import EpisodeBatch
from modules.mixers.qtran import QTranBase


class QADJQTranPaperLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger

        self.last_target_update_episode = 0
        self.train_step = 0
        self.log_stats_t = -self.args.learner_log_interval - 1

        self.beta_upper = getattr(
            args,
            "qadj_beta_upper_init",
            getattr(args, "qadj_beta_upper", getattr(args, "qadj_beta_init", 0.2)),
        )
        self.beta_lower = getattr(
            args,
            "qadj_beta_lower_init",
            getattr(args, "qadj_beta_lower", getattr(args, "qadj_beta_init", 0.2)),
        )
        self.beta_upper_step = getattr(args, "qadj_beta_upper_step", getattr(args, "qadj_beta_step", 0.02))
        self.beta_lower_step = getattr(args, "qadj_beta_lower_step", getattr(args, "qadj_beta_step", 0.02))
        self.beta_dominance_ratio = float(getattr(args, "qadj_beta_dominance_ratio", 1.2))
        self.beta_min_trigger_frac = float(getattr(args, "qadj_beta_min_trigger_frac", 0.1))
        self.beta_min = getattr(args, "qadj_beta_min", 0.01)
        self.beta_upper_min = getattr(args, "qadj_beta_upper_min", self.beta_min)
        self.beta_lower_min = getattr(args, "qadj_beta_lower_min", self.beta_min)
        self.beta_upper_max = getattr(args, "qadj_beta_upper_max", getattr(args, "qadj_beta_max", 0.95))
        self.beta_lower_max = getattr(args, "qadj_beta_lower_max", getattr(args, "qadj_beta_max", 0.95))
        self.adjust_interval = int(getattr(args, "qadj_t1", 20))
        self.sync_interval = int(getattr(args, "qadj_t2", 400))
        self.base_adjust_interval = max(1, self.adjust_interval)
        self.base_sync_interval = max(1, self.sync_interval)
        self.adaptive_schedule = bool(getattr(args, "qadj_adaptive_schedule", False))
        self.adaptive_adjust_min = max(1, int(getattr(args, "qadj_adaptive_t1_min", max(1, self.base_adjust_interval // 2))))
        self.adaptive_adjust_max = max(
            self.adaptive_adjust_min,
            int(getattr(args, "qadj_adaptive_t1_max", self.base_adjust_interval * 4)),
        )
        self.adaptive_sync_min = max(1, int(getattr(args, "qadj_adaptive_t2_min", max(1, self.base_sync_interval // 2))))
        self.adaptive_sync_max = max(
            self.adaptive_sync_min,
            int(getattr(args, "qadj_adaptive_t2_max", self.base_sync_interval * 4)),
        )
        self.adaptive_beta_gain = float(getattr(args, "qadj_adaptive_beta_gain", 1.0))
        self.adaptive_pressure_target = float(getattr(args, "qadj_adaptive_pressure_target", 1.0))
        self.adaptive_conf_target = float(getattr(args, "qadj_adaptive_conf_target", 0.6))
        self.rho = getattr(args, "qadj_rho", 0.3)
        self.eta = getattr(args, "qadj_eta", 1.0)
        self.upper_imbalance_min_gate = float(getattr(args, "qadj_upper_imbalance_min_gate", 0.3))
        self.lower_imbalance_min_gate = float(getattr(args, "qadj_lower_imbalance_min_gate", 0.3))
        self.upper_correction_scale = float(getattr(args, "qadj_upper_correction_scale", 1.0))
        self.lower_correction_scale = float(getattr(args, "qadj_lower_correction_scale", 1.0))
        self.lower_correction_scale_final = float(
            getattr(args, "qadj_lower_correction_scale_final", self.lower_correction_scale)
        )
        self.lower_correction_scale_schedule_start_frac = float(
            getattr(args, "qadj_lower_correction_scale_schedule_start_frac", 1.0)
        )
        self.lower_correction_scale_schedule_end_frac = float(
            getattr(args, "qadj_lower_correction_scale_schedule_end_frac", 1.0)
        )
        self.aux_loss_weight = getattr(args, "qadj_aux_loss_weight", 0.1)
        self.upper_aux_loss_scale = getattr(args, "qadj_upper_aux_loss_scale", 1.0)
        self.lower_aux_loss_scale = getattr(args, "qadj_lower_aux_loss_scale", 1.0)
        self.lower_aux_loss_scale_final = float(
            getattr(args, "qadj_lower_aux_loss_scale_final", self.lower_aux_loss_scale)
        )
        self.lower_aux_loss_scale_schedule_start_frac = float(
            getattr(args, "qadj_lower_aux_loss_scale_schedule_start_frac", 1.0)
        )
        self.lower_aux_loss_scale_schedule_end_frac = float(
            getattr(args, "qadj_lower_aux_loss_scale_schedule_end_frac", 1.0)
        )
        self.bound_use_bias = getattr(args, "qadj_bound_use_bias", True)
        self.bias_abs_max = float(getattr(args, "qadj_bias_abs_max", 6.0))
        self.within_margin = float(getattr(args, "qadj_within_margin", 1.0))
        self.bound_gap_gate_scale = max(1e-6, float(getattr(args, "qadj_bound_gap_gate_scale", 10.0)))
        self.gap_conf_threshold = float(getattr(args, "qadj_gap_conf_threshold", 0.2))
        self.skip_aux_when_no_conf = getattr(args, "qadj_skip_aux_when_no_conf", True)
        self.beta_adjust_min_valid_frac = float(getattr(args, "qadj_beta_adjust_min_valid_frac", 0.25))
        self.qtran_pressure_tau = getattr(args, "qadj_qtran_pressure_ema_tau", 0.95)
        self.qtran_pressure_scale = getattr(args, "qadj_qtran_pressure_scale", 0.08)
        self.qtran_pressure_scale_final = float(
            getattr(args, "qadj_qtran_pressure_scale_final", self.qtran_pressure_scale)
        )
        self.qtran_pressure_scale_schedule_start_frac = float(
            getattr(args, "qadj_qtran_pressure_scale_schedule_start_frac", 1.0)
        )
        self.qtran_pressure_scale_schedule_end_frac = float(
            getattr(args, "qadj_qtran_pressure_scale_schedule_end_frac", 1.0)
        )
        self.qtran_pressure_ema = getattr(args, "qadj_qtran_pressure_init", 0.0)
        self.decay_start_frac = float(getattr(args, "qadj_decay_start_frac", 0.3))
        self.decay_end_frac = float(getattr(args, "qadj_decay_end_frac", 0.85))
        self.min_apply_ratio = float(getattr(args, "qadj_min_apply_ratio", 0.1))
        self.freeze_beta_decay_frac = float(getattr(args, "qadj_freeze_beta_decay_frac", 0.6))
        # In QTRAN (already with opt+nopt constraints), keep QADJ correction
        # moderate so TD correction does not dominate other objectives.
        self.correction_cap_ratio = getattr(args, "qadj_correction_cap_ratio", 0.5)
        self.target_cap = getattr(args, "qadj_target_cap", 100.0)
        self.enable_upper_bound = getattr(args, "qadj_enable_upper_bound", True)
        self.enable_lower_bound = getattr(args, "qadj_enable_lower_bound", True)
        self.control_bounds = getattr(args, "qadj_control_bounds", True)
        self.aux_init_random = getattr(args, "qadj_aux_init_random", False)
        self.aux_sync_with_target = getattr(args, "qadj_aux_sync_with_target", True)
        self.aux_sync_tau = float(getattr(args, "qadj_aux_sync_tau", 1.0))
        self.sync_age = 0
        self.current_train_frac = 0.0
        self.bound_stats = {"over": 0, "under": 0, "within": 0, "valid": 0.0, "total": 0.0}

        self.main_params = list(mac.parameters())

        if args.mixer != "qtran_base":
            raise ValueError(f"QADJ-QTRAN paper learner only supports mixer=qtran_base, got {args.mixer}")

        self.mixer = QTranBase(args)
        self.target_mixer = copy.deepcopy(self.mixer)
        for param in self.target_mixer.parameters():
            param.requires_grad_(False)

        self.target_mac = copy.deepcopy(mac)
        self.upper_mac = copy.deepcopy(mac)
        self.lower_mac = copy.deepcopy(mac)
        self.upper_mixer = copy.deepcopy(self.mixer)
        self.lower_mixer = copy.deepcopy(self.mixer)

        if self.aux_init_random:
            self._reset_module_parameters(self.upper_mac.agent)
            self._reset_module_parameters(self.lower_mac.agent)
            self._reset_module_parameters(self.upper_mixer)
            self._reset_module_parameters(self.lower_mixer)
        else:
            self._sync_auxiliaries(from_target=True)

        self.main_params += list(self.mixer.parameters())
        self.aux_params = (
            list(self.upper_mac.parameters())
            + list(self.lower_mac.parameters())
            + list(self.upper_mixer.parameters())
            + list(self.lower_mixer.parameters())
        )

        self.main_optimiser = RMSprop(
            params=self.main_params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps
        )
        self.aux_optimiser = RMSprop(
            params=self.aux_params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps
        )

    def _clamp_interval(self, value, low, high):
        return int(max(low, min(high, round(value))))

    def _update_adaptive_intervals(self, over_frac, under_frac, within_frac, valid_frac):
        if not self.adaptive_schedule:
            return

        violation_frac = over_frac + under_frac
        low_conf = max(0.0, self.adaptive_conf_target - valid_frac)
        high_pressure = max(0.0, self.qtran_pressure_ema - self.adaptive_pressure_target)

        # More violations or unstable QTRAN constraints need faster beta feedback
        # and fresher auxiliary bounds. Stable within-bound regimes can afford
        # slower updates, which reduces auxiliary over-intervention late in training.
        urgency = min(1.0, violation_frac + low_conf + 0.1 * high_pressure)
        stable = max(0.0, within_frac - violation_frac)
        adjust_scale = 1.0 - 0.5 * urgency + 0.5 * stable
        sync_scale = 1.0 - 0.6 * urgency + 0.6 * stable

        self.adjust_interval = self._clamp_interval(
            self.base_adjust_interval * adjust_scale,
            self.adaptive_adjust_min,
            self.adaptive_adjust_max,
        )
        self.sync_interval = self._clamp_interval(
            self.base_sync_interval * sync_scale,
            self.adaptive_sync_min,
            self.adaptive_sync_max,
        )

    def _reset_module_parameters(self, module):
        def _maybe_reset(submodule):
            if hasattr(submodule, "reset_parameters"):
                submodule.reset_parameters()

        module.apply(_maybe_reset)

    def _forward_mac(self, mac, batch):
        mac_out = []
        mac_hidden_states = []
        mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs = mac.forward(batch, t=t)
            mac_out.append(agent_outs)
            mac_hidden_states.append(mac.hidden_states)
        mac_out = th.stack(mac_out, dim=1)
        mac_hidden_states = th.stack(mac_hidden_states, dim=1)
        mac_hidden_states = mac_hidden_states.reshape(
            batch.batch_size, self.args.n_agents, batch.max_seq_length, -1
        ).transpose(1, 2)
        return mac_out, mac_hidden_states

    def _make_onehot(self, action_index, template):
        return th.zeros_like(template).scatter(3, action_index, 1)

    def _apply_bias(self, joint_qs, beta, is_upper):
        if is_upper:
            scale = th.where(joint_qs >= 0, 1.0 + beta, 1.0 - beta)
        else:
            scale = th.where(joint_qs >= 0, 1.0 - beta, 1.0 + beta)
        biased_qs = joint_qs * scale
        if self.bias_abs_max > 0:
            bias_delta = (biased_qs - joint_qs).clamp(min=-self.bias_abs_max, max=self.bias_abs_max)
            biased_qs = joint_qs + bias_delta
        return biased_qs.clamp(-self.target_cap, self.target_cap)

    def _adjust_betas(self):
        if not self.control_bounds:
            return

        over = self.bound_stats["over"]
        under = self.bound_stats["under"]
        within = self.bound_stats["within"]
        valid = self.bound_stats["valid"]
        total = self.bound_stats["total"]
        valid_frac = valid / max(1.0, total)
        class_total = max(1.0, over + under + within)
        over_frac = over / class_total
        under_frac = under / class_total
        within_frac = within / class_total

        if valid_frac < self.beta_adjust_min_valid_frac:
            self.bound_stats = {"over": 0, "under": 0, "within": 0, "valid": 0.0, "total": 0.0}
            return

        dominance_threshold = self.beta_dominance_ratio
        self._update_adaptive_intervals(over_frac, under_frac, within_frac, valid_frac)

        upper_step = self.beta_upper_step
        lower_step = self.beta_lower_step
        if self.adaptive_schedule:
            pressure_boost = 1.0 + 0.1 * max(0.0, self.qtran_pressure_ema - self.adaptive_pressure_target)
            upper_imbalance = max(0.0, over_frac - max(under_frac, within_frac))
            lower_imbalance = max(0.0, under_frac - max(over_frac, within_frac))
            upper_step = self.beta_upper_step * self.adaptive_beta_gain * (1.0 + upper_imbalance) * pressure_boost
            lower_step = self.beta_lower_step * self.adaptive_beta_gain * (1.0 + lower_imbalance) * pressure_boost

        # Paper-style control: only adjust when one regime is clearly dominant.
        if over_frac >= self.beta_min_trigger_frac and over > dominance_threshold * max(under, within):
            self.beta_upper = min(self.beta_upper_max, self.beta_upper + upper_step)
        elif (
            self.current_train_frac < self.freeze_beta_decay_frac
            and
            over_frac < self.beta_min_trigger_frac
            and within_frac >= self.beta_min_trigger_frac
            and within > dominance_threshold * max(over, under)
        ):
            self.beta_upper = max(self.beta_upper_min, self.beta_upper - upper_step)

        if under_frac >= self.beta_min_trigger_frac and under > dominance_threshold * max(over, within):
            self.beta_lower = min(self.beta_lower_max, self.beta_lower + lower_step)
        elif (
            self.current_train_frac < self.freeze_beta_decay_frac
            and
            under_frac < self.beta_min_trigger_frac
            and within_frac >= self.beta_min_trigger_frac
            and within > dominance_threshold * max(over, under)
        ):
            self.beta_lower = max(self.beta_lower_min, self.beta_lower - lower_step)

        self.bound_stats = {"over": 0, "under": 0, "within": 0, "valid": 0.0, "total": 0.0}

    def _compute_stage_gate(self, t_env: int):
        if self.args.t_max <= 0:
            return 1.0
        train_frac = max(0.0, min(1.0, float(t_env) / float(self.args.t_max)))
        self.current_train_frac = train_frac
        if train_frac <= self.decay_start_frac:
            return 1.0
        if train_frac >= self.decay_end_frac:
            return self.min_apply_ratio
        span = max(1e-6, self.decay_end_frac - self.decay_start_frac)
        progress = (train_frac - self.decay_start_frac) / span
        return 1.0 - progress * (1.0 - self.min_apply_ratio)

    def _soft_update_module(self, target_module, source_module, tau):
        target_state = target_module.state_dict()
        source_state = source_module.state_dict()
        blended_state = {}
        for key, target_tensor in target_state.items():
            source_tensor = source_state[key]
            if th.is_floating_point(target_tensor):
                blended_state[key] = target_tensor * (1.0 - tau) + source_tensor * tau
            else:
                blended_state[key] = source_tensor
        target_module.load_state_dict(blended_state)

    def _sync_auxiliaries(self, from_target=True):
        if from_target:
            src_mac = self.target_mac
            src_mixer = self.target_mixer
        else:
            src_mac = self.mac
            src_mixer = self.mixer

        tau = min(1.0, max(0.0, self.aux_sync_tau))
        if tau >= 1.0:
            self.upper_mac.load_state(src_mac)
            self.lower_mac.load_state(src_mac)
            self.upper_mixer.load_state_dict(src_mixer.state_dict())
            self.lower_mixer.load_state_dict(src_mixer.state_dict())
        else:
            self._soft_update_module(self.upper_mac.agent, src_mac.agent, tau)
            self._soft_update_module(self.lower_mac.agent, src_mac.agent, tau)
            self._soft_update_module(self.upper_mixer, src_mixer, tau)
            self._soft_update_module(self.lower_mixer, src_mixer, tau)
        self.sync_age = 0

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        self.target_mixer.load_state_dict(self.mixer.state_dict())
        if self.aux_sync_with_target:
            # Keep auxiliary bound estimators aligned with the freshly-updated
            # target networks to avoid transient bound-gap spikes.
            self._sync_auxiliaries(from_target=True)
        self.logger.console_logger.info("Updated target network")

    def _scheduled_value(self, start_value, end_value, start_frac, end_frac):
        start_frac = float(start_frac)
        end_frac = float(end_frac)
        if end_frac <= start_frac:
            return float(end_value)
        frac = self.current_train_frac
        if frac <= start_frac:
            return float(start_value)
        if frac >= end_frac:
            return float(end_value)
        progress = (frac - start_frac) / max(1e-6, end_frac - start_frac)
        return float(start_value) + (float(end_value) - float(start_value)) * progress

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        actions_onehot = batch["actions_onehot"]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        self.train_step += 1
        self.sync_age += 1
        flat_mask = mask.reshape(-1, 1)
        stage_gate = self._compute_stage_gate(t_env)
        current_qtran_pressure_scale = self._scheduled_value(
            self.qtran_pressure_scale,
            self.qtran_pressure_scale_final,
            self.qtran_pressure_scale_schedule_start_frac,
            self.qtran_pressure_scale_schedule_end_frac,
        )
        current_lower_correction_scale = self._scheduled_value(
            self.lower_correction_scale,
            self.lower_correction_scale_final,
            self.lower_correction_scale_schedule_start_frac,
            self.lower_correction_scale_schedule_end_frac,
        )
        current_lower_aux_loss_scale = self._scheduled_value(
            self.lower_aux_loss_scale,
            self.lower_aux_loss_scale_final,
            self.lower_aux_loss_scale_schedule_start_frac,
            self.lower_aux_loss_scale_schedule_end_frac,
        )

        mac_out, mac_hidden_states = self._forward_mac(self.mac, batch)
        upper_out, upper_hidden_states = self._forward_mac(self.upper_mac, batch)
        lower_out, lower_hidden_states = self._forward_mac(self.lower_mac, batch)

        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)
        mac_out_maxs = mac_out.masked_fill(avail_actions == 0, -9999999)
        max_action_qvals, max_actions_current = mac_out_maxs.max(dim=3, keepdim=True)
        max_actions_current_onehot = self._make_onehot(max_actions_current, actions_onehot)
        upper_out_detach = upper_out.detach().masked_fill(avail_actions == 0, -9999999)
        lower_out_detach = lower_out.detach().masked_fill(avail_actions == 0, -9999999)
        upper_max_actions_current = upper_out_detach.max(dim=3, keepdim=True)[1]
        lower_max_actions_current = lower_out_detach.max(dim=3, keepdim=True)[1]
        upper_max_actions_current_onehot = self._make_onehot(upper_max_actions_current, actions_onehot)
        lower_max_actions_current_onehot = self._make_onehot(lower_max_actions_current, actions_onehot)

        with th.no_grad():
            target_mac_out, target_mac_hidden_states = self._forward_mac(self.target_mac, batch)
            target_mac_out = target_mac_out.masked_fill(avail_actions == 0, -9999999)
            target_max_actions = target_mac_out.max(dim=3, keepdim=True)[1]
            if self.args.double_q:
                max_actions_onehot = max_actions_current_onehot
            else:
                max_actions_onehot = self._make_onehot(target_max_actions, actions_onehot)

        joint_qs, vs = self.mixer(batch[:, :-1], mac_hidden_states[:, :-1])
        current_upper_qs, _ = self.upper_mixer(
            batch[:, :-1], hidden_states=upper_hidden_states[:, :-1], actions=actions_onehot[:, :-1]
        )
        current_lower_qs, _ = self.lower_mixer(
            batch[:, :-1], hidden_states=lower_hidden_states[:, :-1], actions=actions_onehot[:, :-1]
        )

        with th.no_grad():
            target_joint_qs, _ = self.target_mixer(
                batch[:, 1:],
                hidden_states=target_mac_hidden_states[:, 1:],
                actions=max_actions_onehot[:, 1:],
            )
            upper_self_target_qs, _ = self.upper_mixer(
                batch[:, 1:],
                hidden_states=upper_hidden_states[:, 1:],
                actions=upper_max_actions_current_onehot[:, 1:],
            )
            lower_self_target_qs, _ = self.lower_mixer(
                batch[:, 1:],
                hidden_states=lower_hidden_states[:, 1:],
                actions=lower_max_actions_current_onehot[:, 1:],
            )

            original_bootstrap = target_joint_qs.clamp(-self.target_cap, self.target_cap)
            # Paper definition: U and L are maxima from upper/lower auxiliary
            # estimators on their own greedy joint actions.
            raw_upper_bound = upper_self_target_qs.clamp(-self.target_cap, self.target_cap)
            raw_lower_bound = lower_self_target_qs.clamp(-self.target_cap, self.target_cap)
            if self.bound_use_bias:
                upper_bound = self._apply_bias(raw_upper_bound, self.beta_upper, is_upper=True)
                lower_bound = self._apply_bias(raw_lower_bound, self.beta_lower, is_upper=False)
            else:
                upper_bound = raw_upper_bound
                lower_bound = raw_lower_bound
            sorted_lower = th.minimum(lower_bound, upper_bound)
            sorted_upper = th.maximum(lower_bound, upper_bound)
            effective_lower = sorted_lower + self.within_margin
            effective_upper = sorted_upper - self.within_margin
            valid_interval = effective_lower <= effective_upper

            raw_over_mask = (
                original_bootstrap > effective_upper
                if self.enable_upper_bound
                else th.zeros_like(original_bootstrap, dtype=th.bool)
            )
            raw_under_mask = (
                original_bootstrap < effective_lower
                if self.enable_lower_bound
                else th.zeros_like(original_bootstrap, dtype=th.bool)
            )
            raw_over_mask = raw_over_mask & valid_interval
            raw_under_mask = raw_under_mask & valid_interval

            corrected_bootstrap = original_bootstrap.clone()
            bound_gap = (sorted_upper - sorted_lower).clamp(min=0.0)
            gap_gate = self.bound_gap_gate_scale / (self.bound_gap_gate_scale + bound_gap)
            low_conf_mask = gap_gate < self.gap_conf_threshold
            over_mask = raw_over_mask & (~low_conf_mask)
            under_mask = raw_under_mask & (~low_conf_mask)
            within_mask = (~(over_mask | under_mask)) & (~low_conf_mask)
            low_conf_frac = (low_conf_mask.float() * flat_mask).sum().item() / flat_mask.sum().item()
            aux_conf_frac = 1.0 - low_conf_frac
            valid_count = ((~low_conf_mask).float() * flat_mask).sum().item()
            total_count = flat_mask.sum().item()
            over_count = float((over_mask.float() * flat_mask).sum().item())
            under_count = float((under_mask.float() * flat_mask).sum().item())
            within_count = float((within_mask.float() * flat_mask).sum().item())
            upper_balance_gate = max(
                self.upper_imbalance_min_gate,
                within_count / max(1e-6, within_count + over_count),
            )
            lower_balance_gate = max(
                self.lower_imbalance_min_gate,
                within_count / max(1e-6, within_count + under_count),
            )

            self.bound_stats["over"] += int((over_mask.float() * flat_mask).sum().item())
            self.bound_stats["under"] += int((under_mask.float() * flat_mask).sum().item())
            self.bound_stats["within"] += int((within_mask.float() * flat_mask).sum().item())
            self.bound_stats["valid"] += valid_count
            self.bound_stats["total"] += total_count
            qtran_pressure_gate = 1.0 / (
                1.0 + current_qtran_pressure_scale * max(0.0, float(self.qtran_pressure_ema))
            )

            if self.enable_upper_bound:
                sync_ratio = self.sync_age / max(1, self.sync_interval)
                lam_upper = min(
                    1.0,
                    max(
                        0.0,
                        (0.5 + sync_ratio) * (self.rho + self.eta * self.beta_upper),
                    ),
                )
                lam_upper = (
                    lam_upper
                    * qtran_pressure_gate
                    * gap_gate
                    * upper_balance_gate
                    * stage_gate
                    * self.upper_correction_scale
                )
                upper_correction = lam_upper * (original_bootstrap - sorted_upper)
                upper_cap = self.correction_cap_ratio * (original_bootstrap.abs() + 1.0)
                upper_correction = th.minimum(upper_correction, upper_cap)
                corrected_bootstrap = th.where(
                    over_mask,
                    original_bootstrap - upper_correction,
                    corrected_bootstrap,
                )

            if self.enable_lower_bound:
                sync_ratio = self.sync_age / max(1, self.sync_interval)
                lam_lower = min(
                    1.0,
                    max(
                        0.0,
                        (0.5 + sync_ratio) * (self.rho + self.eta * self.beta_lower),
                    ),
                )
                lam_lower = (
                    lam_lower
                    * qtran_pressure_gate
                    * gap_gate
                    * lower_balance_gate
                    * stage_gate
                    * current_lower_correction_scale
                )
                lower_correction = lam_lower * (sorted_lower - original_bootstrap)
                lower_cap = self.correction_cap_ratio * (original_bootstrap.abs() + 1.0)
                lower_correction = th.minimum(lower_correction, lower_cap)
                corrected_bootstrap = th.where(
                    under_mask,
                    original_bootstrap + lower_correction,
                    corrected_bootstrap,
                )

            bootstrap = corrected_bootstrap.clamp(-self.target_cap, self.target_cap)
            td_targets = rewards.reshape(-1, 1) + self.args.gamma * (
                1 - terminated.reshape(-1, 1)
            ) * bootstrap

            upper_td_targets = rewards.reshape(-1, 1) + self.args.gamma * (
                1 - terminated.reshape(-1, 1)
            ) * self._apply_bias(
                upper_self_target_qs.clamp(-self.target_cap, self.target_cap),
                self.beta_upper,
                is_upper=True,
            )
            lower_td_targets = rewards.reshape(-1, 1) + self.args.gamma * (
                1 - terminated.reshape(-1, 1)
            ) * self._apply_bias(
                lower_self_target_qs.clamp(-self.target_cap, self.target_cap),
                self.beta_lower,
                is_upper=False,
            )

            correction_abs = (bootstrap - original_bootstrap).abs()
        td_error = joint_qs - td_targets.detach()
        masked_td_error = td_error * flat_mask
        td_loss = (masked_td_error ** 2).sum() / flat_mask.sum()

        max_joint_qs, _ = self.mixer(
            batch[:, :-1],
            mac_hidden_states[:, :-1],
            actions=max_actions_current_onehot[:, :-1],
        )
        opt_error = (
            max_action_qvals[:, :-1].sum(dim=2).reshape(-1, 1) - max_joint_qs.detach() + vs
        )
        masked_opt_error = opt_error * flat_mask
        opt_loss = (masked_opt_error ** 2).sum() / flat_mask.sum()

        nopt_values = chosen_action_qvals.sum(dim=2).reshape(-1, 1) - joint_qs.detach() + vs
        nopt_error = nopt_values.clamp(max=0)
        masked_nopt_error = nopt_error * flat_mask
        nopt_loss = (masked_nopt_error ** 2).sum() / flat_mask.sum()
        self.qtran_pressure_ema = (
            self.qtran_pressure_tau * self.qtran_pressure_ema
            + (1.0 - self.qtran_pressure_tau) * float((opt_loss.detach() + nopt_loss.detach()).item())
        )

        upper_td_error = current_upper_qs - upper_td_targets.detach()
        lower_td_error = current_lower_qs - lower_td_targets.detach()
        aux_mask = flat_mask * (~low_conf_mask).float()
        aux_mask_sum = aux_mask.sum()
        if aux_mask_sum.item() <= 0:
            if self.skip_aux_when_no_conf:
                # Keep a valid autograd path even when skipping auxiliary updates.
                upper_td_loss = current_upper_qs.sum() * 0.0
                lower_td_loss = current_lower_qs.sum() * 0.0
            else:
                aux_mask = flat_mask
                aux_mask_sum = flat_mask.sum()
                upper_td_loss = ((upper_td_error ** 2) * aux_mask).sum() / aux_mask_sum
                lower_td_loss = ((lower_td_error ** 2) * aux_mask).sum() / aux_mask_sum
        else:
            upper_td_loss = ((upper_td_error ** 2) * aux_mask).sum() / aux_mask_sum
            lower_td_loss = ((lower_td_error ** 2) * aux_mask).sum() / aux_mask_sum

        # Paper-aligned QADJ: auxiliary estimators are separated only by biased TD targets.
        upper_opt_loss = th.zeros((), device=td_loss.device)
        lower_opt_loss = th.zeros((), device=td_loss.device)
        upper_nopt_loss = th.zeros((), device=td_loss.device)
        lower_nopt_loss = th.zeros((), device=td_loss.device)

        aux_loss = (
            self.upper_aux_loss_scale
            * upper_td_loss
            + current_lower_aux_loss_scale
            * lower_td_loss
        )
        aux_objective = aux_loss

        main_loss = (
            td_loss
            + self.args.opt_loss * opt_loss
            + self.args.nopt_min_loss * nopt_loss
        )
        # Auxiliary estimators are optimized directly by their own objective.
        # aux_loss_weight is kept as a reporting weight for the combined loss.
        aux_loss_scaled = self.aux_loss_weight * aux_objective

        self.main_optimiser.zero_grad()
        main_loss.backward()
        main_grad_norm = th.nn.utils.clip_grad_norm_(self.main_params, self.args.grad_norm_clip)
        self.main_optimiser.step()

        self.aux_optimiser.zero_grad()
        if aux_objective.requires_grad:
            aux_objective.backward()
            aux_grad_norm = th.nn.utils.clip_grad_norm_(self.aux_params, self.args.grad_norm_clip)
            self.aux_optimiser.step()
        else:
            aux_grad_norm = 0.0

        loss = main_loss + aux_loss_scaled

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if self.control_bounds and self.train_step % self.adjust_interval == 0:
            self._adjust_betas()

        if self.control_bounds and self.train_step % self.sync_interval == 0:
            self._sync_auxiliaries(from_target=self.aux_sync_with_target)

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            mask_elems = flat_mask.sum().item()
            self.logger.log_stat("loss", loss.item(), t_env)
            self.logger.log_stat("td_loss", td_loss.item(), t_env)
            self.logger.log_stat("opt_loss", opt_loss.item(), t_env)
            self.logger.log_stat("nopt_loss", nopt_loss.item(), t_env)
            self.logger.log_stat("qadj_aux_loss", aux_loss.item(), t_env)
            self.logger.log_stat("qadj_upper_td_loss", upper_td_loss.item(), t_env)
            self.logger.log_stat("qadj_lower_td_loss", lower_td_loss.item(), t_env)
            self.logger.log_stat("qadj_upper_opt_loss", upper_opt_loss.item(), t_env)
            self.logger.log_stat("qadj_lower_opt_loss", lower_opt_loss.item(), t_env)
            self.logger.log_stat("qadj_upper_nopt_loss", upper_nopt_loss.item(), t_env)
            self.logger.log_stat("qadj_lower_nopt_loss", lower_nopt_loss.item(), t_env)
            self.logger.log_stat("qadj_bound_gap", (bound_gap * flat_mask).sum().item() / mask_elems, t_env)
            self.logger.log_stat(
                "qadj_effective_bound_gap",
                (((effective_upper - effective_lower).clamp(min=0.0)) * flat_mask).sum().item() / mask_elems,
                t_env,
            )
            self.logger.log_stat("qadj_gap_gate", (gap_gate * flat_mask).sum().item() / mask_elems, t_env)
            self.logger.log_stat("qadj_upper_balance_gate", upper_balance_gate, t_env)
            self.logger.log_stat("qadj_lower_balance_gate", lower_balance_gate, t_env)
            self.logger.log_stat("qadj_low_conf_frac", low_conf_frac, t_env)
            self.logger.log_stat("qadj_aux_conf_frac", aux_conf_frac, t_env)
            self.logger.log_stat(
                "qadj_correction_abs", (correction_abs * flat_mask).sum().item() / mask_elems, t_env
            )
            self.logger.log_stat("grad_norm", main_grad_norm, t_env)
            self.logger.log_stat("qadj_aux_grad_norm", aux_grad_norm, t_env)
            self.logger.log_stat("td_error_abs", masked_td_error.abs().sum().item() / mask_elems, t_env)
            self.logger.log_stat("td_targets", (td_targets * flat_mask).sum().item() / mask_elems, t_env)
            self.logger.log_stat("td_chosen_qs", (joint_qs * flat_mask).sum().item() / mask_elems, t_env)
            self.logger.log_stat("v_mean", (vs * flat_mask).sum().item() / mask_elems, t_env)
            self.logger.log_stat(
                "agent_indiv_qs",
                ((chosen_action_qvals * mask).sum().item() / (mask.sum().item() * self.args.n_agents)),
                t_env,
            )
            self.logger.log_stat("qadj_beta_upper", self.beta_upper, t_env)
            self.logger.log_stat("qadj_beta_lower", self.beta_lower, t_env)
            self.logger.log_stat("qadj_adjust_interval", self.adjust_interval, t_env)
            self.logger.log_stat("qadj_sync_interval", self.sync_interval, t_env)
            self.logger.log_stat("qadj_adaptive_schedule", float(self.adaptive_schedule), t_env)
            self.logger.log_stat("qadj_lower_correction_scale", current_lower_correction_scale, t_env)
            self.logger.log_stat("qadj_lower_aux_loss_scale", current_lower_aux_loss_scale, t_env)
            self.logger.log_stat("qadj_qtran_pressure_scale", current_qtran_pressure_scale, t_env)
            self.logger.log_stat("qadj_train_frac", self.current_train_frac, t_env)
            self.logger.log_stat("qadj_stage_gate", stage_gate, t_env)
            self.logger.log_stat("qadj_qtran_pressure", self.qtran_pressure_ema, t_env)
            self.logger.log_stat(
                "qadj_over_frac", (over_mask.float() * flat_mask).sum().item() / mask_elems, t_env
            )
            self.logger.log_stat(
                "qadj_under_frac", (under_mask.float() * flat_mask).sum().item() / mask_elems, t_env
            )
            self.logger.log_stat(
                "qadj_within_frac", (within_mask.float() * flat_mask).sum().item() / mask_elems, t_env
            )
            self.log_stats_t = t_env

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        self.upper_mac.cuda()
        self.lower_mac.cuda()
        self.mixer.cuda()
        self.target_mixer.cuda()
        self.upper_mixer.cuda()
        self.lower_mixer.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        th.save(self.mixer.state_dict(), f"{path}/mixer.th")
        th.save(self.upper_mac.agent.state_dict(), f"{path}/upper_agent.th")
        th.save(self.lower_mac.agent.state_dict(), f"{path}/lower_agent.th")
        th.save(self.upper_mixer.state_dict(), f"{path}/upper_mixer.th")
        th.save(self.lower_mixer.state_dict(), f"{path}/lower_mixer.th")
        th.save(
            {
                "beta_upper": self.beta_upper,
                "beta_lower": self.beta_lower,
                "train_step": self.train_step,
                "sync_age": self.sync_age,
                "adjust_interval": self.adjust_interval,
                "sync_interval": self.sync_interval,
                "qtran_pressure_ema": self.qtran_pressure_ema,
            },
            f"{path}/qadj_state.th",
        )
        th.save(
            {
                "main": self.main_optimiser.state_dict(),
                "aux": self.aux_optimiser.state_dict(),
            },
            f"{path}/opt.th",
        )

    def load_models(self, path):
        self.mac.load_models(path)
        self.target_mac.load_models(path)
        self.upper_mac.agent.load_state_dict(
            th.load(f"{path}/upper_agent.th", map_location=lambda storage, loc: storage)
        )
        self.lower_mac.agent.load_state_dict(
            th.load(f"{path}/lower_agent.th", map_location=lambda storage, loc: storage)
        )
        self.mixer.load_state_dict(th.load(f"{path}/mixer.th", map_location=lambda storage, loc: storage))
        self.target_mixer.load_state_dict(
            th.load(f"{path}/mixer.th", map_location=lambda storage, loc: storage)
        )
        self.upper_mixer.load_state_dict(
            th.load(f"{path}/upper_mixer.th", map_location=lambda storage, loc: storage)
        )
        self.lower_mixer.load_state_dict(
            th.load(f"{path}/lower_mixer.th", map_location=lambda storage, loc: storage)
        )
        state = th.load(f"{path}/qadj_state.th", map_location=lambda storage, loc: storage)
        self.beta_upper = state["beta_upper"]
        self.beta_lower = state["beta_lower"]
        self.train_step = state["train_step"]
        self.sync_age = state["sync_age"]
        self.adjust_interval = int(state.get("adjust_interval", self.adjust_interval))
        self.sync_interval = int(state.get("sync_interval", self.sync_interval))
        self.qtran_pressure_ema = state.get("qtran_pressure_ema", self.qtran_pressure_ema)
        opt_state = th.load(f"{path}/opt.th", map_location=lambda storage, loc: storage)
        if isinstance(opt_state, dict) and "main" in opt_state and "aux" in opt_state:
            self.main_optimiser.load_state_dict(opt_state["main"])
            self.aux_optimiser.load_state_dict(opt_state["aux"])
        else:
            # Backward compatibility with old checkpoints that stored one optimizer.
            self.main_optimiser.load_state_dict(opt_state)
