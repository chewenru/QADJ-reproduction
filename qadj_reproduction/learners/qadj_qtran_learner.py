import copy

import torch as th
from torch.optim import RMSprop

from components.episode_buffer import EpisodeBatch
from modules.mixers.qtran import QTranBase


class QADJQTranLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger

        self.last_target_update_episode = 0
        self.train_step = 0
        self.log_stats_t = -self.args.learner_log_interval - 1

        self.beta_upper = getattr(args, "qadj_beta_upper", getattr(args, "qadj_beta_init", 0.02))
        self.beta_lower = getattr(args, "qadj_beta_lower", getattr(args, "qadj_beta_init", 0.02))
        self.beta_step = getattr(args, "qadj_beta_step", 0.02)
        self.beta_min = getattr(args, "qadj_beta_min", 0.01)
        self.beta_max = getattr(args, "qadj_beta_max", 0.2)
        self.adjust_interval = getattr(args, "qadj_t1", 20)
        self.sync_interval = getattr(args, "qadj_t2", 400)
        self.rho = getattr(args, "qadj_rho", 0.3)
        self.eta = getattr(args, "qadj_eta", 1.0)
        self.aux_loss_weight = getattr(args, "qadj_aux_loss_weight", 0.02)
        self.constraint_weight = getattr(args, "qadj_constraint_weight", 0.005)
        self.margin_cap = getattr(args, "qadj_margin_cap", 10.0)
        self.target_cap = getattr(args, "qadj_target_cap", 100.0)
        self.warmup_steps = getattr(args, "qadj_warmup_steps", 0)
        self.enable_upper_bound = getattr(args, "qadj_enable_upper_bound", True)
        self.enable_lower_bound = getattr(args, "qadj_enable_lower_bound", True)
        self.control_bounds = getattr(args, "qadj_control_bounds", True)
        self.sync_age = 0
        self.bound_stats = {"over": 0, "under": 0, "within": 0}

        self.params = list(mac.parameters())

        if args.mixer != "qtran_base":
            raise ValueError(f"QADJ-QTRAN only supports mixer=qtran_base, got {args.mixer}")

        self.mixer = QTranBase(args)
        self.target_mixer = copy.deepcopy(self.mixer)
        for param in self.target_mixer.parameters():
            param.requires_grad_(False)
        self.params += list(self.mixer.parameters())

        self.target_mac = copy.deepcopy(mac)
        self.upper_mac = copy.deepcopy(mac)
        self.lower_mac = copy.deepcopy(mac)
        self.params += list(self.upper_mac.parameters())
        self.params += list(self.lower_mac.parameters())

        self.optimiser = RMSprop(
            params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps
        )

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

    def _adjust_betas(self):
        if not self.control_bounds:
            return

        over = self.bound_stats["over"]
        under = self.bound_stats["under"]
        within = self.bound_stats["within"]

        if over > within:
            self.beta_upper = min(self.beta_max, self.beta_upper + self.beta_step)
        elif over > 0:
            self.beta_upper = max(self.beta_min, self.beta_upper - self.beta_step)

        if under > within:
            self.beta_lower = min(self.beta_max, self.beta_lower + self.beta_step)
        elif under > 0:
            self.beta_lower = max(self.beta_min, self.beta_lower - self.beta_step)

        self.bound_stats = {"over": 0, "under": 0, "within": 0}

    def _sync_auxiliaries(self):
        self.upper_mac.load_state(self.target_mac)
        self.lower_mac.load_state(self.target_mac)
        self.sync_age = 0

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated target network")

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
        qadj_active = t_env >= self.warmup_steps

        mac_out, mac_hidden_states = self._forward_mac(self.mac, batch)

        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)

        # Keep the current-network greedy values on the autograd graph so the
        # QTRAN opt loss matches the baseline learner.
        mac_out_maxs = mac_out.masked_fill(avail_actions == 0, -9999999)
        max_actions_qvals, max_actions_current = mac_out_maxs.max(dim=3, keepdim=True)

        max_actions_current_ = th.zeros(
            size=(
                batch.batch_size,
                batch.max_seq_length,
                self.args.n_agents,
                self.args.n_actions,
            ),
            device=batch.device,
        )
        max_actions_current_onehot = max_actions_current_.scatter(3, max_actions_current, 1)

        with th.no_grad():
            target_mac_out, target_mac_hidden_states = self._forward_mac(self.target_mac, batch)

            target_mac_out = target_mac_out.masked_fill(avail_actions == 0, -9999999)
            target_max_actions = target_mac_out.max(dim=3, keepdim=True)[1]

            if self.args.double_q:
                max_actions_onehot = max_actions_current_onehot
            else:
                max_actions = th.zeros_like(max_actions_current_)
                max_actions_onehot = max_actions.scatter(3, target_max_actions, 1)

            target_joint_qs, target_vs = self.target_mixer(
                batch[:, 1:],
                hidden_states=target_mac_hidden_states[:, 1:],
                actions=max_actions_onehot[:, 1:],
            )
        joint_qs, vs = self.mixer(batch[:, :-1], mac_hidden_states[:, :-1])

        if qadj_active:
            upper_out, upper_hidden_states = self._forward_mac(self.upper_mac, batch)
            lower_out, lower_hidden_states = self._forward_mac(self.lower_mac, batch)

            with th.no_grad():
                upper_out_detach = upper_out.detach().masked_fill(avail_actions == 0, -9999999)
                lower_out_detach = lower_out.detach().masked_fill(avail_actions == 0, -9999999)
                upper_max_actions = upper_out_detach.max(dim=3, keepdim=True)[1]
                lower_max_actions = lower_out_detach.max(dim=3, keepdim=True)[1]
                upper_actions_onehot = th.zeros_like(max_actions_current_onehot).scatter(3, upper_max_actions, 1)
                lower_actions_onehot = th.zeros_like(max_actions_current_onehot).scatter(3, lower_max_actions, 1)

                upper_target_qs, _ = self.target_mixer(
                    batch[:, 1:],
                    hidden_states=upper_hidden_states[:, 1:],
                    actions=upper_actions_onehot[:, 1:],
                )
                lower_target_qs, _ = self.target_mixer(
                    batch[:, 1:],
                    hidden_states=lower_hidden_states[:, 1:],
                    actions=lower_actions_onehot[:, 1:],
                )

            current_upper_qs, _ = self.target_mixer(
                batch[:, :-1], hidden_states=upper_hidden_states[:, :-1], actions=actions_onehot[:, :-1]
            )
            current_lower_qs, _ = self.target_mixer(
                batch[:, :-1], hidden_states=lower_hidden_states[:, :-1], actions=actions_onehot[:, :-1]
            )

            with th.no_grad():
                original_bootstrap = target_joint_qs.clamp(-self.target_cap, self.target_cap)
                corrected_bootstrap = original_bootstrap.clone()

                target_sorted_lower = th.minimum(lower_target_qs, upper_target_qs)
                target_sorted_upper = th.maximum(lower_target_qs, upper_target_qs)

                target_over_mask = (
                    original_bootstrap > target_sorted_upper
                    if self.enable_upper_bound
                    else th.zeros_like(original_bootstrap, dtype=th.bool)
                )
                target_under_mask = (
                    original_bootstrap < target_sorted_lower
                    if self.enable_lower_bound
                    else th.zeros_like(original_bootstrap, dtype=th.bool)
                )
                target_within_mask = ~(target_over_mask | target_under_mask)

                if self.enable_upper_bound:
                    lam_upper = min(
                        1.0,
                        max(
                            0.0,
                            0.5
                            + (self.sync_age / max(1, self.sync_interval))
                            * (self.rho + self.eta * self.beta_upper),
                        ),
                    )
                    upper_correction = lam_upper * (original_bootstrap - target_sorted_upper)
                    corrected_bootstrap = th.where(
                        target_over_mask,
                        original_bootstrap - upper_correction,
                        corrected_bootstrap,
                    )

                if self.enable_lower_bound:
                    lam_lower = min(
                        1.0,
                        max(
                            0.0,
                            0.5
                            + (self.sync_age / max(1, self.sync_interval))
                            * (self.rho + self.eta * self.beta_lower),
                        ),
                    )
                    lower_correction = lam_lower * (target_sorted_lower - original_bootstrap)
                    corrected_bootstrap = th.where(
                        target_under_mask,
                        original_bootstrap + lower_correction,
                        corrected_bootstrap,
                    )

                bootstrap = corrected_bootstrap.clamp(-self.target_cap, self.target_cap)
                td_targets = rewards.reshape(-1, 1) + self.args.gamma * (
                    1 - terminated.reshape(-1, 1)
                ) * bootstrap

                current_sorted_lower = th.minimum(current_lower_qs.detach(), current_upper_qs.detach())
                current_sorted_upper = th.maximum(current_lower_qs.detach(), current_upper_qs.detach())
                over_mask = (
                    joint_qs.detach() > current_sorted_upper
                    if self.enable_upper_bound
                    else th.zeros_like(joint_qs, dtype=th.bool)
                )
                under_mask = (
                    joint_qs.detach() < current_sorted_lower
                    if self.enable_lower_bound
                    else th.zeros_like(joint_qs, dtype=th.bool)
                )
                within_mask = ~(over_mask | under_mask)
                self.bound_stats["over"] += int(over_mask.sum().item())
                self.bound_stats["under"] += int(under_mask.sum().item())
                self.bound_stats["within"] += int(within_mask.sum().item())

                aux_center = joint_qs.detach().clamp(-self.target_cap, self.target_cap)
                upper_margin = th.clamp(
                    (aux_center.abs() + 1.0) * self.beta_upper, min=0.0, max=self.margin_cap
                )
                lower_margin = th.clamp(
                    (aux_center.abs() + 1.0) * self.beta_lower, min=0.0, max=self.margin_cap
                )
                upper_targets = aux_center + upper_margin
                lower_targets = aux_center - lower_margin
        else:
            bootstrap = target_joint_qs.clamp(-self.target_cap, self.target_cap)
            td_targets = rewards.reshape(-1, 1) + self.args.gamma * (
                1 - terminated.reshape(-1, 1)
            ) * bootstrap
            aux_loss = joint_qs.new_tensor(0.0)
            constraint_loss = joint_qs.new_tensor(0.0)
            over_mask = th.zeros_like(joint_qs, dtype=th.bool)
            under_mask = th.zeros_like(joint_qs, dtype=th.bool)
            within_mask = th.ones_like(joint_qs, dtype=th.bool)

        td_error = joint_qs - td_targets.detach()
        masked_td_error = td_error * mask.reshape(-1, 1)
        td_loss = (masked_td_error ** 2).sum() / mask.sum()

        max_joint_qs, _ = self.mixer(
            batch[:, :-1],
            mac_hidden_states[:, :-1],
            actions=max_actions_current_onehot[:, :-1],
        )
        opt_error = (
            max_actions_qvals[:, :-1].sum(dim=2).reshape(-1, 1) - max_joint_qs.detach() + vs
        )
        masked_opt_error = opt_error * mask.reshape(-1, 1)
        opt_loss = (masked_opt_error ** 2).sum() / mask.sum()

        nopt_values = chosen_action_qvals.sum(dim=2).reshape(-1, 1) - joint_qs.detach() + vs
        nopt_error = nopt_values.clamp(max=0)
        masked_nopt_error = nopt_error * mask.reshape(-1, 1)
        nopt_loss = (masked_nopt_error ** 2).sum() / mask.sum()

        if qadj_active:
            upper_td_error = current_upper_qs - upper_targets.detach()
            lower_td_error = current_lower_qs - lower_targets.detach()
            aux_loss = (
                ((upper_td_error ** 2) + (lower_td_error ** 2)) * mask.reshape(-1, 1)
            ).sum() / mask.sum()

            current_sorted_lower = th.minimum(current_lower_qs.detach(), current_upper_qs.detach())
            current_sorted_upper = th.maximum(current_lower_qs.detach(), current_upper_qs.detach())
            upper_violation = (
                th.relu(joint_qs - current_sorted_upper) if self.enable_upper_bound else th.zeros_like(joint_qs)
            )
            lower_violation = (
                th.relu(current_sorted_lower - joint_qs) if self.enable_lower_bound else th.zeros_like(joint_qs)
            )
            constraint_loss = (
                ((upper_violation ** 2) + (lower_violation ** 2)) * mask.reshape(-1, 1)
            ).sum() / mask.sum()

        # Keep both QADJ-specific objectives off during warmup so the base
        # QTRAN objective can stabilize before auxiliary shaping starts.
        loss = (
            td_loss
            + self.args.opt_loss * opt_loss
            + self.args.nopt_min_loss * nopt_loss
            + (self.aux_loss_weight * aux_loss if qadj_active else 0.0)
            + (self.constraint_weight * constraint_loss if qadj_active else 0.0)
        )

        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if self.control_bounds and qadj_active and self.train_step % self.adjust_interval == 0:
            self._adjust_betas()

        if self.control_bounds and self.train_step % self.sync_interval == 0:
            self._sync_auxiliaries()

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            mask_elems = mask.sum().item()
            self.logger.log_stat("loss", loss.item(), t_env)
            self.logger.log_stat("td_loss", td_loss.item(), t_env)
            self.logger.log_stat("opt_loss", opt_loss.item(), t_env)
            self.logger.log_stat("nopt_loss", nopt_loss.item(), t_env)
            self.logger.log_stat("qadj_aux_loss", aux_loss.item(), t_env)
            self.logger.log_stat("qadj_constraint_mean", constraint_loss.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            self.logger.log_stat("td_error_abs", masked_td_error.abs().sum().item() / mask_elems, t_env)
            self.logger.log_stat("td_targets", (td_targets * mask.reshape(-1, 1)).sum().item() / mask_elems, t_env)
            self.logger.log_stat("td_chosen_qs", (joint_qs * mask.reshape(-1, 1)).sum().item() / mask_elems, t_env)
            self.logger.log_stat("v_mean", (vs * mask.reshape(-1, 1)).sum().item() / mask_elems, t_env)
            self.logger.log_stat(
                "agent_indiv_qs",
                ((chosen_action_qvals * mask).sum().item() / (mask_elems * self.args.n_agents)),
                t_env,
            )
            self.logger.log_stat("qadj_beta_upper", self.beta_upper, t_env)
            self.logger.log_stat("qadj_beta_lower", self.beta_lower, t_env)
            self.logger.log_stat(
                "qadj_over_frac", (over_mask.float() * mask.reshape(-1, 1)).sum().item() / mask_elems, t_env
            )
            self.logger.log_stat(
                "qadj_under_frac", (under_mask.float() * mask.reshape(-1, 1)).sum().item() / mask_elems, t_env
            )
            self.logger.log_stat(
                "qadj_within_frac", (within_mask.float() * mask.reshape(-1, 1)).sum().item() / mask_elems, t_env
            )
            self.log_stats_t = t_env

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        self.upper_mac.cuda()
        self.lower_mac.cuda()
        self.mixer.cuda()
        self.target_mixer.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        th.save(self.mixer.state_dict(), f"{path}/mixer.th")
        th.save(self.upper_mac.agent.state_dict(), f"{path}/upper_agent.th")
        th.save(self.lower_mac.agent.state_dict(), f"{path}/lower_agent.th")
        th.save(
            {
                "beta_upper": self.beta_upper,
                "beta_lower": self.beta_lower,
                "train_step": self.train_step,
                "sync_age": self.sync_age,
            },
            f"{path}/qadj_state.th",
        )
        th.save(self.optimiser.state_dict(), f"{path}/opt.th")

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
        state = th.load(f"{path}/qadj_state.th", map_location=lambda storage, loc: storage)
        self.beta_upper = state["beta_upper"]
        self.beta_lower = state["beta_lower"]
        self.train_step = state["train_step"]
        self.sync_age = state["sync_age"]
        self.optimiser.load_state_dict(th.load(f"{path}/opt.th", map_location=lambda storage, loc: storage))
