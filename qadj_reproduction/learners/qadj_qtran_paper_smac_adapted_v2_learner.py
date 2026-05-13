from .qadj_qtran_paper_learner import QADJQTranPaperLearner


class QADJQTranPaperSMACAdaptedV2Learner(QADJQTranPaperLearner):
    def __init__(self, mac, scheme, logger, args):
        self.qadj_warmup_steps = int(getattr(args, "qadj_warmup_steps", 100000))
        self.qadj_ramp_steps = int(getattr(args, "qadj_ramp_steps", 200000))
        self.current_t_env = 0
        super().__init__(mac, scheme, logger, args)

    def train(self, batch, t_env: int, episode_num: int):
        self.current_t_env = int(t_env)
        super().train(batch, t_env, episode_num)

    def _compute_stage_gate(self, t_env: int):
        base_gate = super()._compute_stage_gate(t_env)
        if t_env < self.qadj_warmup_steps:
            return 0.0
        if self.qadj_ramp_steps <= 0:
            return base_gate
        progress = (t_env - self.qadj_warmup_steps) / float(self.qadj_ramp_steps)
        ramp_gate = max(0.0, min(1.0, progress))
        return base_gate * ramp_gate

    def _adjust_betas(self):
        if self.current_t_env < self.qadj_warmup_steps:
            self.bound_stats = {"over": 0, "under": 0, "within": 0, "valid": 0.0, "total": 0.0}
            return
        super()._adjust_betas()
