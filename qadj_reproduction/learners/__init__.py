from .q_learner import QLearner
from .coma_learner import COMALearner
from .qtran_learner import QLearner as QTranLearner
from .qadj_learner import QADJLearner
from .qadj_dmaq_learner import QADJDMAQLearner
from .qadj_qtran_learner import QADJQTranLearner
from .qadj_qtran_paper_learner import QADJQTranPaperLearner
from .qadj_qtran_paper_strict_learner import QADJQTranPaperStrictLearner
from .qadj_qtran_paper_smac_adapted_learner import QADJQTranPaperSMACAdaptedLearner
from .qadj_qtran_paper_smac_adapted_v2_learner import QADJQTranPaperSMACAdaptedV2Learner
from .weighted_q_learner import WeightedQLearner
from .dmaq_qatten_learner import DMAQ_qattenLearner

REGISTRY = {}

REGISTRY["q_learner"] = QLearner
REGISTRY["coma_learner"] = COMALearner
REGISTRY["qtran_learner"] = QTranLearner
REGISTRY["qadj_learner"] = QADJLearner
REGISTRY["qadj_dmaq_learner"] = QADJDMAQLearner
REGISTRY["qadj_qtran_learner"] = QADJQTranLearner
REGISTRY["qadj_qtran_paper_learner"] = QADJQTranPaperLearner
REGISTRY["qadj_qtran_paper_strict_learner"] = QADJQTranPaperStrictLearner
REGISTRY["qadj_qtran_paper_smac_adapted_learner"] = QADJQTranPaperSMACAdaptedLearner
REGISTRY["qadj_qtran_paper_smac_adapted_v2_learner"] = QADJQTranPaperSMACAdaptedV2Learner
REGISTRY["weighted_q_learner"] = WeightedQLearner
REGISTRY["dmaq_qatten_learner"] = DMAQ_qattenLearner
