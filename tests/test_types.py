from llm_debate_swarm.types import Question, Verdict


def test_question_neutral_defaults():
    q = Question(question="Will X happen?")
    assert q.yes_price == 0.5
    assert q.no_price == 0.5
    assert q.days_to_resolution == 30.0


def test_question_with_prior_and_horizon():
    q = Question(question="Q", prior=0.7, horizon_days=10)
    assert abs(q.yes_price - 0.7) < 1e-9
    assert abs(q.no_price - 0.3) < 1e-9
    assert q.days_to_resolution == 10


def test_verdict_fields():
    v = Verdict(question="Q", probability=0.6, confidence=0.5)
    assert v.probability == 0.6
    assert v.per_model == []
