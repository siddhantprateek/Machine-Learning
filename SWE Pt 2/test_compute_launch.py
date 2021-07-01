from compute_launch import days_until_launch

def test_days_until_launch_4():
    assert(days_until_launch(22, 26) == 4)

def test_days_until_launch_0():
    assert(days_until_launch(253, 253) == 0)

def test_days_until_launch_0_negative():
    assert(days_until_launch(83, 64) == -19)
    
def test_days_until_launch_10():
    assert(days_until_launch(10, 10) == 0)
    
def test_days_until_launch_12():
    assert(days_until_launch(10, 12) == 2)
    
def test_days_until_launch_100():
    assert(days_until_launch(99, 100) == 1)
    
def test_days_until_launch_90():
    assert(days_until_launch(90, 100) == 10)