import random

from meteor import prpatt


def test_expotas():
    s1 = 5
    t1 = 25
    assert prpatt.expotas(0, s1, t1) == 0
    assert prpatt.expotas(random.randint(0, 200), s1, t1) < s1
