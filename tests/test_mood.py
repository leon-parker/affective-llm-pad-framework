from src.affectgpt.emotion import PAD, Mood
def test_mood_decay_and_blend():
    m = Mood(decay=0.8)
    for _ in range(5): s = m.update(PAD(0.8,0,0), blend=0.25)
    assert s.pleasure > 0.2
