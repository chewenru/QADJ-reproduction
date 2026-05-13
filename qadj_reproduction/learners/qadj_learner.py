import copy

import torch as th
from torch.optim import RMSprop

from components.episode_buffer import EpisodeBatch
from modules.mixers.qmix import QMixer
from modules.mixers.vdn import VDNMixer


class QADJLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger
        self.last_target_update_episode = 0
        self.train_step = 0
        self.log_stats_t = -self.args.learner_log_interval - 1

        self.beta_upper = getattr(args, "qadj_beta_upper", getattr(args, "qadj_beta_init", 0.2))
        self.beta_lower = getattr(args, "qadj_beta_lower", getattr(args, "qadj_beta_init", 0.2))
        self.beta_step = getattr(args, "qadj_beta_step", 0.02)
        self.beta_min = getattr(args, "qadj_beta_min", 0.01)
        self.beta_max = getattr(args, "qadj_beta_max", 0.95)
        self.adjust_interval = getattr(args, "qadj_t1", 20)
        self.sync_interval = getattr(args, "qadj_t2", 400)
        self.rho = getattr(args, "qadj_rho", 0.3)
        self.eta = getattr(args, "qadj_eta", 1.0)
        self.aux_loss_weight = getattr(args, "qadj_aux_loss_weight", 0.1)
        self.enable_upper_bound = getattr(args, "qadj_enable_upper_bound", True)
        self.enable_lower_bound = getattr(args, "qadj_enable_lower_bound", True)
        self.control_bounds = getattr(args, "qadj_control_bounds", True)
        self.sync_age = 0
        self.bound_stats = {"over": 0, "under": 0, "within": 0}

        self.params = list(mac.parameters())
        self.mixer = self._build_mixer(args.mixer)
        self.target_mixer = copy.deepcopy(self.mixer)
        self.params += list(self.mixer.parameters())

        self.target_mac = copy.deepcopy(mac)
        self.upper_mac = copy.deepcopy(mac)
        self.lower_mac = copy.deepcopy(mac)
        self.upper_mixer = copy.deepcopy(self.mixer)
        self.lower_mixer = copy.deepcopy(self.mixer)

        self.params += list(self.upper_mac.parameters())
        self.params += list(self.lower_mac.parameters())
        self.params += list(self.upper_mixer.parameters())
        self.params += list(self.lower_mixer.parameters())

        self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)

    def _build_mixer(self, mixer_name):
        if mixer_name == "qmix":
            return QMixer(self.args)
        if mixer_name == "vdn":
            return VDNMixer()
        raise ValueError(f"QADJ does not support mixer={mixer_name}")

    def _forward_mac(self, mac, batch):
        outs = []
        mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            outs.append(mac.forward(batch, t=t))
        return th.stack(outs, dim=1)

    def _mix(self, mixer, chosen_qvals, states):
        return mixer(chosen_qvals, states)

    def _compute_aux_targets(self, next_qvals, rewards, terminated, beta, is_upper):
        multiplier = th.where(next_qvals > 0, 1.0 + beta, 1.0 - beta) if is_upper else th.where(next_qvals > 0, 1.0 - beta, 1.0 + beta)
        return rewards + self.args.gamma * (1 - terminated) * next_qvals * multiplier

    def _adjust_betas(self):
        if not self.control_bounds:
            return
        over = self.bound_stats["over"]
        under = self.bound_stats["under"]
        within = self.bound_stats["within"]

        if over > within:
            self.beta_upper = min(self.beta_max, self.beta_upper + self.beta_step)
        else:
            self.beta_upper = max(self.beta_min, self.beta_upper - self.beta_step)

        if under > within:
            self.beta_lower = min(self.beta_max, self.beta_lower + self.beta_step)
        else:
            self.beta_lower = max(self.beta_min, self.beta_lower - self.beta_step)

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

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        self.train_step += 1
        self.sync_age += 1

        for mac in (self.mac, self.upper_mac, self.lower_mac, self.target_mac):
            if hasattr(mac, "agent"):
                mac.agent.train()

        mac_out = self._forward_mac(self.mac, batch)
        upper_out = self._forward_mac(self.upper_mac, batch)
        lower_out = self._forward_mac(self.lower_mac, batch)

        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)
        chosen_upper_qvals = th.gather(upper_out[:, :-1], dim=3, index=actions).squeeze(3)
        chosen_lower_qvals = th.gather(lower_out[:, :-1], dim=3, index=actions).squeeze(3)

        with th.no_grad():
            target_mac_out = self._forward_mac(self.target_mac, batch)
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]

            target_next = th.gather(target_mac_out[:, 1:], 3, cur_max_actions).squeeze(3)

            upper_detach = upper_out.clone().detach()
            lower_detach = lower_out.clone().detach()
            upper_detach[avail_actions == 0] = -9999999
            lower_detach[avail_actions == 0] = -9999999
            upper_next = upper_detach[:, 1:].max(dim=3)[0]
            lower_next = lower_detach[:, 1:].max(dim=3)[0]

        mixed_chosen = self._mix(self.mixer, chosen_action_qvals, batch["state"][:, :-1])
        mixed_upper = self._mix(self.upper_mixer, chosen_upper_qvals, batch["state"][:, :-1])
        mixed_lower = self._mix(self.lower_mixer, chosen_lower_qvals, batch["state"][:, :-1])

        with th.no_grad():
            original_bootstrap = self._mix(self.target_mixer, target_next, batch["state"][:, 1:])
            upper_bound = self._mix(self.upper_mixer, upper_next, batch["state"][:, 1:])
            lower_bound = self._mix(self.lower_mixer, lower_next, batch["state"][:, 1:])

            over_mask = original_bootstrap > upper_bound if self.enable_upper_bound else th.zeros_like(original_bootstrap, dtype=th.bool)
            under_mask = original_bootstrap < lower_bound if self.enable_lower_bound else th.zeros_like(original_bootstrap, dtype=th.bool)
            within_mask = ~(over_mask | under_mask)

            self.bound_stats["over"] += int(over_mask.sum().item())
            self.bound_stats["under"] += int(under_mask.sum().item())
            self.bound_stats["within"] += int(within_mask.sum().item())

            corrected = original_bootstrap.clone()

            if self.enable_upper_bound:
                lam_upper = min(1.0, max(0.0, 0.5 + (self.sync_age / max(1, self.sync_interval)) * (self.rho + self.eta * self.beta_upper)))
                correction_upper = lam_upper * (original_bootstrap - upper_bound)
                corrected = th.where(over_mask, original_bootstrap - correction_upper, corrected)

            if self.enable_lower_bound:
                lam_lower = min(1.0, max(0.0, 0.5 + (self.sync_age / max(1, self.sync_interval)) * (self.rho + self.eta * self.beta_lower)))
                correction_lower = lam_lower * (lower_bound - original_bootstrap)
                corrected = th.where(under_mask, original_bootstrap + correction_lower, corrected)

            original_targets = rewards + self.args.gamma * (1 - terminated) * corrected
            upper_targets = self._compute_aux_targets(upper_bound, rewards, terminated, self.beta_upper, is_upper=True)
            lower_targets = self._compute_aux_targets(lower_bound, rewards, terminated, self.beta_lower, is_upper=False)

        td_error = mixed_chosen - original_targets.detach()
        upper_td_error = mixed_upper - upper_targets.detach()
        lower_td_error = mixed_lower - lower_targets.detach()

        mask = mask.expand_as(td_error)
        masked_td_error = (td_error ** 2) * mask
        masked_upper_error = (upper_td_error ** 2) * mask
        masked_lower_error = (lower_td_error ** 2) * mask

        main_loss = masked_td_error.sum() / mask.sum()
        upper_loss = masked_upper_error.sum() / mask.sum()
        lower_loss = masked_lower_error.sum() / mask.sum()
        aux_loss = upper_loss + lower_loss
        loss = main_loss + self.aux_loss_weight * aux_loss

        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if self.control_bounds and self.train_step % self.adjust_interval == 0:
            self._adjust_betas()

        if self.control_bounds and self.train_step % self.sync_interval == 0:
            self._sync_auxiliaries()

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            mask_elems = mask.sum().item()
            self.logger.log_stat("loss", loss.item(), t_env)
            self.logger.log_stat("main_loss", main_loss.item(), t_env)
            self.logger.log_stat("qadj_aux_loss", aux_loss.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            self.logger.log_stat("td_error_abs", (td_error.abs() * mask).sum().item() / mask_elems, t_env)
            self.logger.log_stat("q_taken_mean", (mixed_chosen * mask).sum().item() / (mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("target_mean", (original_targets * mask).sum().item() / (mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("qadj_beta_upper", self.beta_upper, t_env)
            self.logger.log_stat("qadj_beta_lower", self.beta_lower, t_env)
            self.logger.log_stat("qadj_over_frac", (over_mask.float() * mask).sum().item() / mask_elems, t_env)
            self.logger.log_stat("qadj_under_frac", (under_mask.float() * mask).sum().item() / mask_elems, t_env)
            self.logger.log_stat("qadj_within_frac", (within_mask.float() * mask).sum().item() / mask_elems, t_env)
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
            },
            f"{path}/qadj_state.th",
        )
        th.save(self.optimiser.state_dict(), f"{path}/opt.th")

    def load_models(self, path):
        self.mac.load_models(path)
        self.target_mac.load_models(path)
        self.upper_mac.agent.load_state_dict(th.load(f"{path}/upper_agent.th", map_location=lambda storage, loc: storage))
        self.lower_mac.agent.load_state_dict(th.load(f"{path}/lower_agent.th", map_location=lambda storage, loc: storage))
        self.mixer.load_state_dict(th.load(f"{path}/mixer.th", map_location=lambda storage, loc: storage))
        self.target_mixer.load_state_dict(th.load(f"{path}/mixer.th", map_location=lambda storage, loc: storage))
        self.upper_mixer.load_state_dict(th.load(f"{path}/upper_mixer.th", map_location=lambda storage, loc: storage))
        self.lower_mixer.load_state_dict(th.load(f"{path}/lower_mixer.th", map_location=lambda storage, loc: storage))
        state = th.load(f"{path}/qadj_state.th", map_location=lambda storage, loc: storage)
        self.beta_upper = state["beta_upper"]
        self.beta_lower = state["beta_lower"]
        self.train_step = state["train_step"]
        self.sync_age = state["sync_age"]
        self.optimiser.load_state_dict(th.load(f"{path}/opt.th", map_location=lambda storage, loc: storage))
