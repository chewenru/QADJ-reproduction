import copy
from pathlib import Path

import torch as th
from torch.optim import RMSprop

from components.episode_buffer import EpisodeBatch
from modules.mixers.dmaq_general import DMAQer
from modules.mixers.dmaq_qatten import DMAQ_QattenMixer


class QADJDMAQLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger
        self.last_target_update_episode = 0
        self.train_step = 0
        self.log_stats_t = -self.args.learner_log_interval - 1
        self.n_actions = self.args.n_actions

        self.beta_upper = getattr(args, "qadj_beta_upper", getattr(args, "qadj_beta_init", 0.02))
        self.beta_lower = getattr(args, "qadj_beta_lower", getattr(args, "qadj_beta_init", 0.02))
        self.beta_step = getattr(args, "qadj_beta_step", 0.002)
        self.beta_upper_step = getattr(args, "qadj_beta_upper_step", self.beta_step)
        self.beta_lower_step = getattr(args, "qadj_beta_lower_step", self.beta_step)
        self.beta_min = getattr(args, "qadj_beta_min", 0.01)
        self.beta_max = getattr(args, "qadj_beta_max", 0.1)
        self.beta_upper_min = getattr(args, "qadj_beta_upper_min", self.beta_min)
        self.beta_lower_min = getattr(args, "qadj_beta_lower_min", self.beta_min)
        self.beta_upper_max = getattr(args, "qadj_beta_upper_max", self.beta_max)
        self.beta_lower_max = getattr(args, "qadj_beta_lower_max", self.beta_max)
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
        self.rho = getattr(args, "qadj_rho", 0.3)
        self.eta = getattr(args, "qadj_eta", 1.0)
        self.aux_loss_weight = getattr(args, "qadj_aux_loss_weight", 0.02)
        self.constraint_weight = getattr(args, "qadj_constraint_weight", 0.005)
        self.margin_cap = getattr(args, "qadj_margin_cap", 10.0)
        self.target_cap = getattr(args, "qadj_target_cap", 100.0)
        self.warmup_steps = getattr(args, "qadj_warmup_steps", 0)
        self.main_start_steps = getattr(args, "qadj_main_start_steps", self.warmup_steps)
        self.main_ramp_steps = getattr(args, "qadj_main_ramp_steps", 0)
        self.main_max_weight = getattr(args, "qadj_main_max_weight", 1.0)
        self.correction_cap_ratio = getattr(args, "qadj_correction_cap_ratio", 0.5)
        self.enable_upper_bound = getattr(args, "qadj_enable_upper_bound", True)
        self.enable_lower_bound = getattr(args, "qadj_enable_lower_bound", True)
        self.control_bounds = getattr(args, "qadj_control_bounds", True)
        self.sync_age = 0
        self.bound_stats = {"over": 0, "under": 0, "within": 0}

        self.main_params = list(mac.parameters())
        if args.mixer == "dmaq":
            self.mixer = DMAQer(args)
        elif args.mixer == "dmaq_qatten":
            self.mixer = DMAQ_QattenMixer(args)
        else:
            raise ValueError(f"QADJ-QPLEX mixer {args.mixer} not recognised.")

        self.target_mixer = copy.deepcopy(self.mixer)
        for param in self.target_mixer.parameters():
            param.requires_grad_(False)
        self.main_params += list(self.mixer.parameters())

        self.target_mac = copy.deepcopy(mac)
        self.upper_mac = copy.deepcopy(mac)
        self.lower_mac = copy.deepcopy(mac)
        self.upper_mixer = copy.deepcopy(self.mixer)
        self.lower_mixer = copy.deepcopy(self.mixer)
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

        self._sync_auxiliaries()

    def _clamp_interval(self, value, low, high):
        return int(max(low, min(high, round(value))))

    def _set_train_mode(self, mac):
        if hasattr(mac, "set_train_mode"):
            mac.set_train_mode()
        elif hasattr(mac, "agent"):
            mac.agent.train()

    def _set_eval_mode(self, mac):
        if hasattr(mac, "set_evaluation_mode"):
            mac.set_evaluation_mode()
        elif hasattr(mac, "agent"):
            mac.agent.eval()

    def _forward_mac(self, mac, batch):
        outputs = []
        mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            outputs.append(mac.forward(batch, t=t))
        return th.stack(outputs, dim=1)

    def _mix(self, mixer, agent_qs, states, actions, max_q_i):
        agent_qs = agent_qs.contiguous()
        states = states.contiguous()
        actions = actions.contiguous()
        max_q_i = max_q_i.contiguous()
        if self.args.mixer == "dmaq_qatten":
            chosen, attend_reg, _ = mixer(agent_qs, states, is_v=True)
            adv, _, _ = mixer(agent_qs, states, actions=actions, max_q_i=max_q_i, is_v=False)
        else:
            chosen = mixer(agent_qs, states, is_v=True)
            adv = mixer(agent_qs, states, actions=actions, max_q_i=max_q_i, is_v=False)
            attend_reg = agent_qs.new_zeros(())
        return chosen + adv, attend_reg

    def _adjust_betas(self):
        if not self.control_bounds:
            return

        over = self.bound_stats["over"]
        under = self.bound_stats["under"]
        within = self.bound_stats["within"]
        total = max(1.0, over + under + within)
        over_frac = over / total
        under_frac = under / total
        within_frac = within / total
        upper_step = self.beta_upper_step
        lower_step = self.beta_lower_step
        if self.adaptive_schedule:
            violation_frac = over_frac + under_frac
            stable = max(0.0, within_frac - violation_frac)
            urgency = min(1.0, violation_frac)
            self.adjust_interval = self._clamp_interval(
                self.base_adjust_interval * (1.0 - 0.5 * urgency + 0.5 * stable),
                self.adaptive_adjust_min,
                self.adaptive_adjust_max,
            )
            self.sync_interval = self._clamp_interval(
                self.base_sync_interval * (1.0 - 0.6 * urgency + 0.6 * stable),
                self.adaptive_sync_min,
                self.adaptive_sync_max,
            )
            upper_step = self.beta_upper_step * self.adaptive_beta_gain * (1.0 + over_frac)
            lower_step = self.beta_lower_step * self.adaptive_beta_gain * (1.0 + under_frac)

        if over > within:
            self.beta_upper = min(self.beta_upper_max, self.beta_upper + upper_step)
        elif over > 0:
            self.beta_upper = max(self.beta_upper_min, self.beta_upper - upper_step)

        if under > within:
            self.beta_lower = min(self.beta_lower_max, self.beta_lower + lower_step)
        elif under > 0:
            self.beta_lower = max(self.beta_lower_min, self.beta_lower - lower_step)

        self.bound_stats = {"over": 0, "under": 0, "within": 0}

    def _sync_auxiliaries(self):
        self.upper_mac.load_state(self.target_mac)
        self.lower_mac.load_state(self.target_mac)
        self.upper_mixer.load_state_dict(self.target_mixer.state_dict())
        self.lower_mixer.load_state_dict(self.target_mixer.state_dict())
        self.sync_age = 0

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated target network")

    def _build_onehot(self, action_index):
        onehot = th.zeros(
            action_index.squeeze(3).shape + (self.n_actions,),
            device=action_index.device,
        )
        onehot.scatter_(3, action_index, 1)
        return onehot

    def _qadj_main_weight(self, t_env):
        if t_env < self.main_start_steps:
            return 0.0
        if self.main_ramp_steps <= 0:
            return self.main_max_weight
        ramp = float(t_env - self.main_start_steps) / float(self.main_ramp_steps)
        return min(self.main_max_weight, max(0.0, ramp * self.main_max_weight))

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int, save_data=None):
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]
        actions_onehot = batch["actions_onehot"][:, :-1]

        self.train_step += 1
        self.sync_age += 1
        qadj_active = t_env >= self.warmup_steps
        qadj_main_weight = self._qadj_main_weight(t_env) if qadj_active else 0.0

        for mac in (self.mac, self.upper_mac, self.lower_mac):
            self._set_train_mode(mac)
        self._set_eval_mode(self.target_mac)
        self.mixer.train()
        self.upper_mixer.train()
        self.lower_mixer.train()
        self.target_mixer.eval()

        mac_out = self._forward_mac(self.mac, batch)
        upper_out = self._forward_mac(self.upper_mac, batch)
        lower_out = self._forward_mac(self.lower_mac, batch)

        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)
        chosen_upper_qvals = th.gather(upper_out[:, :-1], dim=3, index=actions).squeeze(3)
        chosen_lower_qvals = th.gather(lower_out[:, :-1], dim=3, index=actions).squeeze(3)

        mac_out_detach = mac_out.clone().detach()
        mac_out_detach[avail_actions == 0] = -9999999
        max_action_qvals = mac_out_detach[:, :-1].max(dim=3)[0]
        if self.args.double_q:
            next_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
        else:
            next_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
        next_actions_onehot = self._build_onehot(next_actions)

        upper_out_detach = upper_out.clone().detach()
        upper_out_detach[avail_actions == 0] = -9999999
        upper_max_action_qvals = upper_out_detach[:, :-1].max(dim=3)[0]
        upper_next_actions = upper_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
        upper_next_actions_onehot = self._build_onehot(upper_next_actions)

        lower_out_detach = lower_out.clone().detach()
        lower_out_detach[avail_actions == 0] = -9999999
        lower_max_action_qvals = lower_out_detach[:, :-1].max(dim=3)[0]
        lower_next_actions = lower_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
        lower_next_actions_onehot = self._build_onehot(lower_next_actions)

        mixed_chosen, main_attend_reg = self._mix(
            self.mixer,
            chosen_action_qvals,
            batch["state"][:, :-1],
            actions_onehot,
            max_action_qvals,
        )
        current_upper_qs, upper_attend_reg = self._mix(
            self.upper_mixer,
            chosen_upper_qvals,
            batch["state"][:, :-1],
            actions_onehot,
            upper_max_action_qvals,
        )
        current_lower_qs, lower_attend_reg = self._mix(
            self.lower_mixer,
            chosen_lower_qvals,
            batch["state"][:, :-1],
            actions_onehot,
            lower_max_action_qvals,
        )

        with th.no_grad():
            target_mac_out = self._forward_mac(self.target_mac, batch)[:, 1:]
            target_mac_out[avail_actions[:, 1:] == 0] = -9999999

            if self.args.double_q:
                target_actions = next_actions
            else:
                target_actions = target_mac_out.max(dim=3, keepdim=True)[1]
                next_actions_onehot = self._build_onehot(target_actions)

            target_chosen_qvals = th.gather(target_mac_out, 3, target_actions).squeeze(3)
            target_max_qvals = target_mac_out.max(dim=3)[0]
            original_bootstrap, _ = self._mix(
                self.target_mixer,
                target_chosen_qvals,
                batch["state"][:, 1:],
                next_actions_onehot,
                target_max_qvals,
            )
            original_bootstrap = original_bootstrap.clamp(-self.target_cap, self.target_cap)

            upper_target_chosen_qvals = th.gather(
                upper_out_detach[:, 1:], 3, upper_next_actions
            ).squeeze(3)
            upper_target_max_qvals = upper_out_detach[:, 1:].max(dim=3)[0]
            upper_self_target_qs, _ = self._mix(
                self.upper_mixer,
                upper_target_chosen_qvals,
                batch["state"][:, 1:],
                upper_next_actions_onehot,
                upper_target_max_qvals,
            )
            upper_self_target_qs = upper_self_target_qs.clamp(-self.target_cap, self.target_cap)

            lower_target_chosen_qvals = th.gather(
                lower_out_detach[:, 1:], 3, lower_next_actions
            ).squeeze(3)
            lower_target_max_qvals = lower_out_detach[:, 1:].max(dim=3)[0]
            lower_self_target_qs, _ = self._mix(
                self.lower_mixer,
                lower_target_chosen_qvals,
                batch["state"][:, 1:],
                lower_next_actions_onehot,
                lower_target_max_qvals,
            )
            lower_self_target_qs = lower_self_target_qs.clamp(-self.target_cap, self.target_cap)

            if qadj_active:
                sorted_lower = th.minimum(lower_self_target_qs, upper_self_target_qs)
                sorted_upper = th.maximum(lower_self_target_qs, upper_self_target_qs)
                over_mask = (
                    original_bootstrap > sorted_upper
                    if self.enable_upper_bound
                    else th.zeros_like(original_bootstrap, dtype=th.bool)
                )
                under_mask = (
                    original_bootstrap < sorted_lower
                    if self.enable_lower_bound
                    else th.zeros_like(original_bootstrap, dtype=th.bool)
                )
                within_mask = ~(over_mask | under_mask)
                self.bound_stats["over"] += int(over_mask.sum().item())
                self.bound_stats["under"] += int(under_mask.sum().item())
                self.bound_stats["within"] += int(within_mask.sum().item())

                corrected_bootstrap = original_bootstrap.clone()
                if self.control_bounds:
                    sync_ratio = self.sync_age / max(1, self.sync_interval)
                    if self.enable_upper_bound:
                        lam_upper = min(
                            1.0,
                            max(
                                0.0,
                                (0.5 + sync_ratio) * (self.rho + self.eta * self.beta_upper),
                            ),
                        )
                        upper_correction = qadj_main_weight * lam_upper * (original_bootstrap - sorted_upper)
                        upper_cap = self.correction_cap_ratio * (original_bootstrap.abs() + 1.0)
                        upper_correction = th.minimum(upper_correction, upper_cap)
                        corrected_bootstrap = th.where(
                            over_mask,
                            original_bootstrap - upper_correction,
                            corrected_bootstrap,
                        )

                    if self.enable_lower_bound:
                        lam_lower = min(
                            1.0,
                            max(
                                0.0,
                                (0.5 + sync_ratio) * (self.rho + self.eta * self.beta_lower),
                            ),
                        )
                        lower_correction = qadj_main_weight * lam_lower * (sorted_lower - original_bootstrap)
                        lower_cap = self.correction_cap_ratio * (original_bootstrap.abs() + 1.0)
                        lower_correction = th.minimum(lower_correction, lower_cap)
                        corrected_bootstrap = th.where(
                            under_mask,
                            original_bootstrap + lower_correction,
                            corrected_bootstrap,
                        )

                bootstrap = corrected_bootstrap.clamp(-self.target_cap, self.target_cap)
                aux_center = (
                    rewards + self.args.gamma * (1 - terminated) * bootstrap
                ).detach()
                upper_margin = th.clamp(
                    (aux_center.abs() + 1.0) * self.beta_upper,
                    min=0.0,
                    max=self.margin_cap,
                )
                lower_margin = th.clamp(
                    (aux_center.abs() + 1.0) * self.beta_lower,
                    min=0.0,
                    max=self.margin_cap,
                )
                upper_targets = aux_center + upper_margin
                lower_targets = aux_center - lower_margin
            else:
                bootstrap = original_bootstrap
                over_mask = th.zeros_like(original_bootstrap, dtype=th.bool)
                under_mask = th.zeros_like(original_bootstrap, dtype=th.bool)
                within_mask = th.ones_like(original_bootstrap, dtype=th.bool)
                upper_targets = None
                lower_targets = None

            targets = rewards + self.args.gamma * (1 - terminated) * bootstrap

        td_error = mixed_chosen - targets.detach()
        mask = mask.expand_as(td_error)
        mask_sum = mask.sum()
        main_loss = ((td_error ** 2) * mask).sum() / mask_sum + main_attend_reg

        if qadj_active:
            upper_td_error = current_upper_qs - upper_targets.detach()
            lower_td_error = current_lower_qs - lower_targets.detach()
            aux_loss = (((upper_td_error ** 2) + (lower_td_error ** 2)) * mask).sum() / mask_sum

            current_sorted_lower = th.minimum(current_lower_qs.detach(), current_upper_qs.detach())
            current_sorted_upper = th.maximum(current_lower_qs.detach(), current_upper_qs.detach())
            upper_violation = (
                th.relu(mixed_chosen - current_sorted_upper)
                if self.enable_upper_bound
                else th.zeros_like(mixed_chosen)
            )
            lower_violation = (
                th.relu(current_sorted_lower - mixed_chosen)
                if self.enable_lower_bound
                else th.zeros_like(mixed_chosen)
            )
            constraint_loss = (((upper_violation ** 2) + (lower_violation ** 2)) * mask).sum() / mask_sum
            main_loss = main_loss + qadj_main_weight * self.constraint_weight * constraint_loss
            aux_objective = self.aux_loss_weight * aux_loss + upper_attend_reg + lower_attend_reg
        else:
            aux_loss = mixed_chosen.new_tensor(0.0)
            constraint_loss = mixed_chosen.new_tensor(0.0)
            aux_objective = mixed_chosen.new_tensor(0.0)

        self.main_optimiser.zero_grad()
        main_loss.backward()
        main_grad_norm = th.nn.utils.clip_grad_norm_(self.main_params, self.args.grad_norm_clip)
        self.main_optimiser.step()

        if qadj_active:
            self.aux_optimiser.zero_grad()
            aux_objective.backward()
            aux_grad_norm = th.nn.utils.clip_grad_norm_(self.aux_params, self.args.grad_norm_clip)
            self.aux_optimiser.step()
        else:
            aux_grad_norm = 0.0

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if self.control_bounds and qadj_active and self.train_step % self.adjust_interval == 0:
            self._adjust_betas()

        if self.train_step % self.sync_interval == 0:
            self._sync_auxiliaries()

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            mask_elems = mask.sum().item()
            self.logger.log_stat("loss", (main_loss + aux_objective).item(), t_env)
            self.logger.log_stat("grad_norm", main_grad_norm, t_env)
            self.logger.log_stat("qadj_aux_grad_norm", aux_grad_norm, t_env)
            self.logger.log_stat("td_error_abs", (td_error.abs() * mask).sum().item() / mask_elems, t_env)
            self.logger.log_stat(
                "q_taken_mean",
                (mixed_chosen * mask).sum().item() / (mask_elems * self.args.n_agents),
                t_env,
            )
            self.logger.log_stat(
                "target_mean",
                (targets * mask).sum().item() / (mask_elems * self.args.n_agents),
                t_env,
            )
            self.logger.log_stat("qadj_aux_loss", aux_loss.item(), t_env)
            self.logger.log_stat("qadj_constraint_mean", constraint_loss.item(), t_env)
            self.logger.log_stat("qadj_beta_upper", self.beta_upper, t_env)
            self.logger.log_stat("qadj_beta_lower", self.beta_lower, t_env)
            self.logger.log_stat("qadj_adjust_interval", self.adjust_interval, t_env)
            self.logger.log_stat("qadj_sync_interval", self.sync_interval, t_env)
            self.logger.log_stat("qadj_adaptive_schedule", float(self.adaptive_schedule), t_env)
            self.logger.log_stat(
                "qadj_active", 1.0 if qadj_active else 0.0, t_env
            )
            self.logger.log_stat("qadj_main_weight", qadj_main_weight, t_env)
            self.logger.log_stat(
                "qadj_over_frac", (over_mask.float() * mask).sum().item() / mask_elems, t_env
            )
            self.logger.log_stat(
                "qadj_under_frac", (under_mask.float() * mask).sum().item() / mask_elems, t_env
            )
            self.logger.log_stat(
                "qadj_within_frac", (within_mask.float() * mask).sum().item() / mask_elems, t_env
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
            },
            f"{path}/qadj_state.th",
        )
        th.save(self.main_optimiser.state_dict(), f"{path}/opt.th")
        th.save(self.main_optimiser.state_dict(), f"{path}/main_opt.th")
        th.save(self.aux_optimiser.state_dict(), f"{path}/aux_opt.th")

    def load_models(self, path):
        checkpoint_dir = Path(path)

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

        upper_mixer_path = checkpoint_dir / "upper_mixer.th"
        lower_mixer_path = checkpoint_dir / "lower_mixer.th"
        if upper_mixer_path.exists() and lower_mixer_path.exists():
            self.upper_mixer.load_state_dict(
                th.load(str(upper_mixer_path), map_location=lambda storage, loc: storage)
            )
            self.lower_mixer.load_state_dict(
                th.load(str(lower_mixer_path), map_location=lambda storage, loc: storage)
            )
        else:
            self.upper_mixer.load_state_dict(self.mixer.state_dict())
            self.lower_mixer.load_state_dict(self.mixer.state_dict())

        state = th.load(f"{path}/qadj_state.th", map_location=lambda storage, loc: storage)
        self.beta_upper = state.get("beta_upper", self.beta_upper)
        self.beta_lower = state.get("beta_lower", self.beta_lower)
        self.train_step = state.get("train_step", self.train_step)
        self.sync_age = state.get("sync_age", self.sync_age)
        self.adjust_interval = int(state.get("adjust_interval", self.adjust_interval))
        self.sync_interval = int(state.get("sync_interval", self.sync_interval))

        main_opt_path = checkpoint_dir / "main_opt.th"
        if main_opt_path.exists():
            self.main_optimiser.load_state_dict(
                th.load(str(main_opt_path), map_location=lambda storage, loc: storage)
            )
        elif (checkpoint_dir / "opt.th").exists():
            self.main_optimiser.load_state_dict(
                th.load(f"{path}/opt.th", map_location=lambda storage, loc: storage)
            )

        aux_opt_path = checkpoint_dir / "aux_opt.th"
        if aux_opt_path.exists():
            self.aux_optimiser.load_state_dict(
                th.load(str(aux_opt_path), map_location=lambda storage, loc: storage)
            )
