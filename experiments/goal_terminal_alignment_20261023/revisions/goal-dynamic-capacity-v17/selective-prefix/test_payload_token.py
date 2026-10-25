"""TensorIO.payload_digest is an identity token of a captured CPU payload, not a content hash. CPU only."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent / 'source-overlay'))
import torch  # noqa: E402
from remote_ra_growing_worker.tensors import TensorIO  # noqa: E402


def payload(value):
    return {'hybrid': ((torch.full((4, 8), value, dtype=torch.bfloat16), torch.zeros(3)),),
            'attention': ((torch.ones(2, 2),),)}


class PayloadTokenTests(unittest.TestCase):
    def test_same_object_same_token_and_equal_content_different_token(self):
        a, b = payload(1.0), payload(1.0)
        token = TensorIO.payload_digest(a)
        self.assertRegex(token, r'^[0-9a-f]{64}$')
        self.assertEqual(TensorIO.payload_digest(a), token)            # stable for the object
        self.assertNotEqual(TensorIO.payload_digest(b), token)         # another copy, same bytes
        # A new dict over the same captured tensors is the same payload.
        self.assertEqual(TensorIO.payload_digest({k: v for k, v in a.items()}), token)

    def test_tokens_do_not_read_the_bytes(self):
        a = payload(2.0)
        token = TensorIO.payload_digest(a)
        a['attention'][0][0].add_(1)          # in-place change: identity, not content, is tracked
        self.assertEqual(TensorIO.payload_digest(a), token)

    def test_empty_payload_is_rejected(self):
        with self.assertRaisesRegex(Exception, 'empty tensor payload'):
            TensorIO.payload_digest({'hybrid': ()})


if __name__ == '__main__':
    unittest.main()
