"""Admission CPU path: plain_copy (the phase-1 lease and payload copies) and the token
checks (render binding, Qwen request) against their former forms; CPU only.

plain_copy replaced copy.deepcopy of the phase-1 request, lease and payload (a
prompt list of up to 256K ints, ~1 us per element through deepcopy); the token
checks replaced all(type(x) is int and x >= 0 for x in tokens) with a C-speed
set/min check, and a binding remembers the tuple it checked.
"""
import copy
import json
from pathlib import Path
import random
import sys
from types import SimpleNamespace as NS
import unittest

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'source-overlay'))
from remote_ra_live_v2.contract import plain_copy, token_ids

class CopyAndTokenTests(unittest.TestCase):
    def test_plain_copy_is_deepcopy_of_json_data(self):
        rng = random.Random(13)

        def value(depth):
            kind = rng.randrange(9 if depth < 4 else 5)
            if kind == 0:
                return rng.randint(-5, 10 ** 6)
            if kind == 1:
                return rng.random()
            if kind == 2:
                return rng.choice((None, True, False, 'x', ''))
            if kind == 3:
                return [rng.randint(0, 9) for _ in range(rng.randint(0, 5))]
            if kind == 4:
                return tuple(rng.randint(0, 9) for _ in range(rng.randint(0, 3)))
            if kind in (5, 6):
                return {str(rng.randint(0, 99)): value(depth + 1) for _ in range(rng.randint(0, 4))}
            if kind == 7:
                return [value(depth + 1) for _ in range(rng.randint(0, 4))]
            return tuple(value(depth + 1) for _ in range(rng.randint(0, 3)))
        for _ in range(300):
            v = value(0)
            c = plain_copy(v)
            self.assertEqual(c, copy.deepcopy(v))
            self.assertEqual(type(c), type(v))
            self.assertEqual(json.dumps(c, sort_keys=True), json.dumps(v, sort_keys=True))

            def containers(x, out):
                if type(x) in (list, dict):
                    out.append(x)
                for y in (x.values() if type(x) is dict else x if type(x) in (list, tuple) else ()):
                    containers(y, out)
                return out
            for a, b in zip(containers(v, []), containers(c, [])):
                self.assertIsNot(a, b)          # every mutable container is new
        tokens = list(range(1000))
        payload = dict(tokens=tokens, lease=dict(rows=[[1, 2], [3]]))
        clone = plain_copy(payload)
        clone['tokens'].append(5); clone['lease']['rows'][0].append(9)
        self.assertEqual((len(tokens), payload['lease']['rows'][0]), (1000, [1, 2]))
        atoms = (1, 2, 'x')
        self.assertIs(plain_copy(atoms), atoms)
        odd = NS(a=[1])
        self.assertIsNot(plain_copy([odd])[0], odd)     # other objects: deepcopy

    def test_token_checks_match_the_generator_form(self):
        from remote_ra_native_growing import contract as native
        cases = [(), (0,), (1, 2, 3), (True,), (1, False), (-1,), (1, -5, 2), (1.0,), ('1',), (None,),
                 tuple(range(10 ** 5)), (2 ** 70,), (1, 2 ** 70, -1)]
        for tokens in cases:
            want = all(type(x) is int and x >= 0 for x in tokens)
            self.assertEqual(token_ids(tokens), want, tokens)
            self.assertEqual(token_ids(list(tokens)), want, tokens)
            if tokens:
                b = native.RenderBinding('e' * 64, 's', 'r', 0, 0, False, 'g', 'm', 'x', tokens, 0, None, 'p', 'n')
                if want:
                    native.binding_shape(b)
                    self.assertIs(b.__dict__['_checked_tokens'], tokens)
                    native.binding_shape(b)            # remembered: no scan
                    bad = tokens[:-1] + (-1,)
                    object.__setattr__(b, 'tokens', bad)
                    with self.assertRaisesRegex(Exception, 'invalid tokens'):
                        native.binding_shape(b)        # a replaced tuple is scanned again
                else:
                    with self.assertRaisesRegex(Exception, 'invalid tokens'):
                        native.binding_shape(b)
                    self.assertNotIn('_checked_tokens', b.__dict__)


if __name__ == '__main__':
    unittest.main()
