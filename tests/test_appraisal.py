from src.affectgpt.appraisal import analyze
def test_appraisal_keywords():
    label, _, _ = analyze('I feel so angry about this!')
    assert label in {'anger','sadness'}
    label2, _, _ = analyze('Thanks, I really appreciate your help.')
    assert label2 in {'gratitude','joy'}
