"""Answer grading: multiple-choice letters and option texts, and box answers by IoU."""
import pytest

from saturn.pipeline.evaluate import _parse_mcq_options, evaluate_answer

Q = "Which is closest? Options: A: chair, B: desk, C: lamp, D: bed"


# ---------------------------------------------------------------- multiple choice

def test_empty_prediction_is_wrong():
    assert evaluate_answer("", "A", Q) is False
    assert evaluate_answer("   ", "A", "Pick one. A. chair B. desk") is False


@pytest.mark.parametrize("pred,gt,expected", [
    ("chair", "C", False),   # option A text, not letter C
    ("chair", "A", True),
    ("desk", "D", False),    # option B text, not letter D
    ("desk", "B", True),
    ("bed", "D", True),      # option D text, not letter B
    ("lamp", "C", True),
])
def test_option_text_is_not_read_as_a_letter(pred, gt, expected):
    assert evaluate_answer(pred, gt, Q) is expected


@pytest.mark.parametrize("pred", ["C", "c", "C.", "C)", "(C)", "C: lamp", "C lamp", "C-"])
def test_letter_answers_score(pred):
    assert evaluate_answer(pred, "C", Q) is True
    assert evaluate_answer(pred, "A", Q) is False


def test_word_ending_in_option_letter_is_not_an_option():
    opts = _parse_mcq_options("Figure: the lamp. Options: A: chair, B: desk")
    assert opts == {"A": "chair", "B": "desk"}


# ---------------------------------------------------------------- box answers (3D-FORCE, non-GT)

def test_non_gt_grading_needs_a_predicted_box():
    """Without a predicted box, a non-GT 3D-FORCE answer is wrong: the option index is
    never looked up in the ground-truth boxes."""
    item = {"bboxes": [[0, 0, 10, 10], [100, 100, 200, 200]]}
    gt = [100, 100, 200, 200]
    assert evaluate_answer("1", gt, "q", item=item, result_record={"predicted_bbox": None}) is False
    assert evaluate_answer("1", gt, "q", item=item, result_record=None) is False
    # a real predicted box grades by IoU
    ok = {"predicted_bbox": [101, 101, 199, 199]}
    assert evaluate_answer("0", gt, "q", item=item, result_record=ok) is True
