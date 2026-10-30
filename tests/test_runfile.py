from s1d1_bench.runfile import find_json_objects, strip_reasoning


def test_strip_reasoning_keeps_code_fences():
    assert strip_reasoning("<think>x</think>```cpp\nint a;\n```") == " ```cpp\nint a;\n```"


def test_strip_reasoning_takes_the_text_after_a_prefilled_close_tag():
    assert strip_reasoning("</think>\n\nBonjour") == "\n\nBonjour"


def test_find_json_objects_skips_undecodable_braces():
    assert find_json_objects('{ not json } {"a": 1} tail {"b": [2]}') == [{"a": 1}, {"b": [2]}]
