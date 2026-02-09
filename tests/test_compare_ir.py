from nl2spec.comparator.compare_ir import compare_ir


def test_equal_event_ir():
    ref = {
        "category": "EVENT",
        "ir": {
            "type": "event",
            "events": [{"name": "open", "timing": "before"}],
            "guard": "file is null",
            "violation_message": "error"
        }
    }

    gen = {
        "category": "EVENT",
        "ir": {
            "type": "event",
            "events": [{"name": "open", "timing": "before"}],
            "guard": "file is null",
            "violation_message": "error"
        }
    }

    diff = compare_ir(ref, gen)
    assert diff.is_equal


def test_missing_event():
    ref = {
        "category": "EVENT",
        "ir": {
            "type": "event",
            "events": [{"name": "open", "timing": "before"}],
            "guard": "x",
            "violation_message": "error"
        }
    }

    gen = {
        "category": "EVENT",
        "ir": {
            "type": "event",
            "events": [],
            "guard": "x",
            "violation_message": "error"
        }
    }

    diff = compare_ir(ref, gen)
    assert not diff.is_equal
    assert "Missing event" in diff.errors[0]
