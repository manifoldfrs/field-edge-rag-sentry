from src.ring_buffer import RingBuffer


def test_basic_ops():
    rb = RingBuffer(3)
    rb.append(1)
    rb.extend([2, 3, 4])  # buffer now holds 2,3,4
    assert len(rb) == 3
    assert rb.to_list() == [2.0, 3.0, 4.0]
    assert abs(rb.mean() - 3.0) < 1e-6


def test_capacity_check():
    try:
        RingBuffer(0)
    except ValueError:
        assert True
    else:
        assert False, "Expected ValueError for non‑positive capacity"
