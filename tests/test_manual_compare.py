from core.comparator.compare_ir import compare_ir


IR_A = {
    "category": "EVENT",
    "ir": {
        "type": "event",
        "events": [
            {"name": "File.close", "timing": "after"}
        ],
        "guard": "true",
        "violation_message": "File must be closed."
    }
}

IR_B = {
    "category": "EVENT",
    "ir": {
        "type": "event",
        "events": [
            {"name": "File.close", "timing": "before"}
        ],
        "guard": "true",
        "violation_message": "File must be closed."
    }
}


def main():
    print("=== Comparing identical IRs ===")
    result_same = compare_ir(IR_A, IR_A)
    print(result_same)

    print("\n=== Comparing different IRs ===")
    result_diff = compare_ir(IR_A, IR_B)
    print(result_diff)


if __name__ == "__main__":
    main()
