from llm_debate_swarm.analysis.question_classifier import QuestionType, classify_question


def test_barrier_question():
    c = classify_question("Will Bitcoin reach $100,000 by December?")
    assert c.question_type == QuestionType.BARRIER


def test_head_to_head_question():
    c = classify_question("Lakers vs Celtics: who wins?")
    assert c.question_type == QuestionType.HEAD_TO_HEAD


def test_returns_classification_with_confidence():
    c = classify_question("Will it rain tomorrow?")
    assert 0.0 <= c.confidence <= 1.0
    assert isinstance(c.question_type, QuestionType)
