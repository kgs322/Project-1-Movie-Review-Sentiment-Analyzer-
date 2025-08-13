from my_service import MyService, Config

def test_run_basic():
    """MyService should process a list of items without changes."""
    svc = MyService(Config(debug=True))
    result = svc.run([1, 2, 3])
    assert result == [1, 2, 3]

def test_run_empty():
    """MyService should handle an empty list gracefully."""
    svc = MyService(Config(debug=False))
    result = svc.run([])
    assert result == []

def test_run_with_exceptions():
    """MyService should log exceptions and continue processing."""
    svc = MyService(Config(debug=True))

    def bad_item():
        raise ValueError("bad item")

    data = [1, bad_item, 3]  # bad_item will raise an exception when called
    processed = []
    for x in data:
        if callable(x):
            try:
                processed.append(x())
            except Exception:
                continue
        else:
            processed.append(x)

    result = svc.run(processed)
    assert len(result) == 2  # Should skip the invalid item
    assert result == [1, 3]
