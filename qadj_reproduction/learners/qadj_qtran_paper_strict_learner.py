from .qadj_qtran_paper_learner import QADJQTranPaperLearner


class QADJQTranPaperStrictLearner(QADJQTranPaperLearner):
    def __init__(self, mac, scheme, logger, args):
        # Neutralize engineering stabilizers so this path stays close to the
        # paper principle: auxiliary upper/lower estimators, beta self-adjustment,
        # and direct bound-based bootstrap correction.
        args.qadj_bias_abs_max = 0.0
        args.qadj_within_margin = 0.0
        args.qadj_bound_gap_gate_scale = 1e9
        args.qadj_gap_conf_threshold = -1.0
        args.qadj_skip_aux_when_no_conf = False
        args.qadj_beta_adjust_min_valid_frac = 0.0
        args.qadj_qtran_pressure_scale = 0.0
        args.qadj_decay_start_frac = 1.0
        args.qadj_decay_end_frac = 1.0
        args.qadj_min_apply_ratio = 1.0
        args.qadj_freeze_beta_decay_frac = 1.0
        args.qadj_correction_cap_ratio = 1e9
        args.qadj_upper_imbalance_min_gate = 1.0
        args.qadj_lower_imbalance_min_gate = 1.0
        args.qadj_aux_sync_tau = 1.0
        args.qadj_aux_sync_with_target = True
        args.qadj_aux_init_random = False
        super().__init__(mac, scheme, logger, args)
