# SPDX-License-Identifier: GPL-3.0-only
# search-bot — grounded scientific RAG engine
# Copyright (C) 2026 raaaas
# This program comes with ABSOLUTELY NO WARRANTY; it is free software, and you
# are welcome to redistribute it under GNU GPL-3.0-only terms. See LICENSE.

"""The shipped UI file as a contract: what it may claim, and how it may embed data.

The Python tests cover the API; nothing covered the page that consumes it, and both
ways it broke are cheap to check from here. The server reads this file from disk on
every request, so these assertions are about the artefact users actually get.
"""
import pathlib
import re
import shutil
import subprocess
import unittest

UI = (pathlib.Path(__file__).resolve().parent.parent / "web" / "index.html") \
    .read_text(encoding="utf-8")

# Interpolations that legitimately produce a number or a static fragment rather
# than carrying text: a .map() index. Anything else in a handler must be escaped.
PLAIN_VALUES = {"i"}


class HandlerEscapingTest(unittest.TestCase):
    """Data reaching an inline handler must be a JS literal, not HTML-escaped text.

    esc() is the wrong tool here even though it is the right one everywhere else:
    the JS of an event-handler attribute is read after entity decoding, so a value
    containing a backtick or ${ closes the template literal it was poured into and
    the remainder runs as code. Corpus passages quote markdown, so answer text can
    carry both without anyone meaning to.
    """

    def test_every_handler_interpolation_is_a_js_literal(self):
        bad = []
        for m in re.finditer(r'\bon\w+\s*=\s*"[^"]*?\$\{([^{}]*)\}', UI):
            expr = m.group(1).strip()
            if expr.startswith("jsAttr(") or expr in PLAIN_VALUES:
                continue
            bad.append((UI[:m.start()].count("\n") + 1, expr))
        self.assertEqual(bad, [], f"inline handlers need jsAttr(), not esc(): {bad}")

    def test_jsattr_is_a_json_literal_wrapped_in_the_html_escape(self):
        body = re.search(r"function jsAttr\(value\)\s*\{(.*?)\}", UI, re.S).group(1)
        self.assertIn("JSON.stringify", body)
        self.assertIn("esc(", body)


class ClaimsTest(unittest.TestCase):
    """The engine has no re-ranker and gets no usage figures, so the page says so
    once and offers no control that pretends otherwise."""

    def test_cross_encoder_appears_only_as_a_denial(self):
        hits = re.findall(r"[^.]*cross-encoder[^.]*\.", UI, re.I)
        self.assertEqual(len(hits), 1, hits)
        self.assertRegex(hits[0], r"[Nn]o cross-encoder", hits[0])

    def test_no_token_usage_is_displayed(self):
        for label in ("Session Tokens", "Token Analytics", "tokens this", "prompt_tokens"):
            self.assertNotIn(label, UI, label)


@unittest.skipUnless(shutil.which("node"), "node not installed")
class ScriptParsesTest(unittest.TestCase):
    def test_the_inline_script_is_valid_javascript(self):
        js = re.search(r"<script>(.*)</script>", UI, re.S).group(1)
        proc = subprocess.run(["node", "--check"], input=js.encode(), capture_output=True)
        self.assertEqual(proc.returncode, 0, proc.stderr.decode()[:2000])


if __name__ == "__main__":
    unittest.main()
