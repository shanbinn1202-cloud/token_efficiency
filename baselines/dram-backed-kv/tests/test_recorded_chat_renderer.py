from __future__ import annotations

import unittest

from scripts.run_oracle_budget_sweep import build_recorded_trajectory, chat_renderer


class _CharacterTokenizer:
    @staticmethod
    def _encode(text: str) -> list[int]:
        return [ord(character) for character in text]

    def __call__(self, text: str, *, add_special_tokens: bool = False):
        del add_special_tokens
        return {"input_ids": self._encode(text)}

    def apply_chat_template(self, messages, **kwargs):
        del kwargs
        return {
            "input_ids": self._encode("".join(
                f"<|im_start|>{message['role']}\n{message.get('content', '')}<|im_end|>\n"
                for message in messages
            ))
        }


class RecordedChatRendererTests(unittest.TestCase):
    def test_tool_observation_is_qwen_wrapped_and_prefix_stable(self) -> None:
        messages = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "task"},
            {"role": "assistant", "content": "action"},
            {"role": "tool", "content": "result"},
            {"role": "assistant", "content": "next action"},
        ]
        render = chat_renderer(_CharacterTokenizer())
        action = render(messages[:3], False)
        observation = render(messages[:4], False)
        context = render(messages[:4], True)
        continuation = render(messages[:5], False)
        self.assertEqual(observation[:len(action)], action)
        self.assertEqual(context[:len(observation)], observation)
        self.assertEqual(continuation[:len(context)], context)
        decoded = "".join(chr(value) for value in observation)
        self.assertIn(
            "<|im_start|>user\n<tool_response>\nresult\n</tool_response><|im_end|>\n",
            decoded,
        )

    def test_builder_accepts_multiple_input_messages_before_action(self) -> None:
        messages = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "task"},
            {"role": "assistant", "content": "first"},
            {"role": "tool", "content": "tool result"},
            {"role": "user", "content": "retry notice"},
            {"role": "assistant", "content": "second"},
            {"role": "tool", "content": "done"},
        ]
        annotation = {
            "trajectory": {"instance_id": "multi-input"},
            "message_map": {"terminal_exit_message_id": 6},
            "turns": [
                {"turn_id": 1, "assistant_message_id": 2, "input_message_ids": [0, 1]},
                {"turn_id": 2, "assistant_message_id": 5, "input_message_ids": [3, 4]},
            ],
        }

        trajectory = build_recorded_trajectory(
            messages, annotation, chat_renderer(_CharacterTokenizer())
        )

        second = trajectory.turns[1]
        self.assertEqual(second.prefill_spans[0].source_message_ids, (3, 4))
        self.assertEqual(second.observation_text, "tool result\nretry notice")
        decoded = "".join(chr(value) for value in second.prefill_spans[0].token_ids)
        self.assertIn("tool result", decoded)
        self.assertIn("retry notice", decoded)

    def test_builder_keeps_canonical_token_when_bpe_crosses_action_boundary(self) -> None:
        messages = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "task"},
            {"role": "assistant", "content": "first"},
            {"role": "tool", "content": "result"},
            {"role": "user", "content": "notice"},
            {"role": "assistant", "content": "second"},
            {"role": "tool", "content": "done"},
        ]
        complete = {
            2: [1, 2],
            3: [1, 2, 3, 4],
            4: [1, 2, 3, 4, 5],
            5: [1, 2, 3, 4, 5, 6],
            6: [1, 2, 3, 4, 5, 6, 8, 9],
            7: [1, 2, 3, 4, 5, 6, 8, 9, 10],
        }

        def render(selected, add_generation_prompt):
            count = len(selected)
            if add_generation_prompt:
                return [1, 2, 3] if count == 2 else [1, 2, 3, 4, 5, 6, 7]
            return complete[count]

        annotation = {
            "trajectory": {"instance_id": "bpe-boundary"},
            "message_map": {"terminal_exit_message_id": 6},
            "turns": [
                {"turn_id": 1, "assistant_message_id": 2, "input_message_ids": [0, 1]},
                {"turn_id": 2, "assistant_message_id": 5, "input_message_ids": [3, 4]},
            ],
        }

        trajectory = build_recorded_trajectory(messages, annotation, render)

        first = trajectory.turns[0]
        second = trajectory.turns[1]
        emitted = [
            token
            for turn in (first, second)
            for span in (*turn.prefill_spans, *turn.generated_spans)
            for token in span.token_ids
        ]
        self.assertEqual(emitted, complete[6])
        self.assertEqual(second.prefill_spans[0].token_ids, (5, 6, 8))
        self.assertEqual(second.generated_spans[0].token_ids, (9,))


if __name__ == "__main__":
    unittest.main()
