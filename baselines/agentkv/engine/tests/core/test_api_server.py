from minisgl.server.api_server import _parse_qwen3_tool_calls


def test_parse_qwen3_tool_call_accepts_tool_parameters_shape() -> None:
    """Tau2 traces show Qwen3 sometimes emits Gorilla-style tool/parameters JSON."""
    text = (
        "<think>need a lookup</think>\n\n"
        "<tool_call>\n"
        '{"tool":"get_reservation_details","parameters":{"reservation_id":"ABC123"}}'
        "\n</tool_call>"
    )

    content, tool_calls = _parse_qwen3_tool_calls(text, uid=17)

    assert content == "<think>need a lookup</think>"
    assert tool_calls == [
        {
            "id": "call_17_0",
            "type": "function",
            "function": {
                "name": "get_reservation_details",
                "arguments": '{"reservation_id":"ABC123"}',
            },
        }
    ]


def test_parse_qwen3_tool_call_accepts_nested_function_parameters_shape() -> None:
    """Nested function objects can also use parameters instead of arguments."""
    text = (
        "<tool_call>"
        '{"function":{"name":"cancel_reservation","parameters":{"reservation_id":"ABC123"}}}'
        "</tool_call>"
    )

    content, tool_calls = _parse_qwen3_tool_calls(text, uid=3)

    assert content is None
    assert tool_calls[0]["function"] == {
        "name": "cancel_reservation",
        "arguments": '{"reservation_id":"ABC123"}',
    }
