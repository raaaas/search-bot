# SPDX-License-Identifier: GPL-3.0-only
# search-bot — grounded scientific RAG engine
# Copyright (C) 2026 raaaas
# This program comes with ABSOLUTELY NO WARRANTY; it is free software, and you
# are welcome to redistribute it under GNU GPL-3.0-only terms. See LICENSE.

"""Control-token scrubbing and the chat retry ladder — the grounding last line.

requests is replaced wholesale inside searchbot.llm, so nothing here opens a
socket or needs a model server.
"""
import json
import sys
sys.path.insert(0, __file__.rsplit("/", 1)[0])
import support  # noqa: E402
from searchbot import config, llm  # noqa: E402


class FakeResponse:
    def __init__(self, payload, stream_lines=None, status=200, text=None):
        self._payload = payload
        self.status_code = status
        self._lines = stream_lines
        self.text = text if text is not None else str(payload)

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def iter_lines(self, decode_unicode=False):
        return iter(self._lines or [])


def chat_body(text, finish="stop"):
    return {"choices": [{"message": {"content": text}, "finish_reason": finish}]}


class FakeRequests:
    """Plugs a scripted sequence of responses; records every payload sent."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.sent = []
        self.headers = []

    def post(self, url, json=None, timeout=None, stream=False, **kw):
        self.sent.append(json)
        self.headers.append(kw.get("headers"))
        if not self.responses:
            raise AssertionError("FakeRequests ran out of canned responses")
        r = self.responses.pop(0)
        return r if isinstance(r, FakeResponse) else FakeResponse(r)

    def get(self, url, timeout=None, **kw):
        self.headers.append(kw.get("headers"))
        return FakeResponse({"data": [{"id": "test-model"}]})


class LLMRequestCase(support.TempCase):
    """Restores the real llm.embed so these tests exercise the actual request
    builder (TempCase swaps it out for a deterministic stand-in)."""

    def setUp(self):
        super().setUp()
        self._real_requests = llm.requests
        llm.embed = self._embed          # self._embed is the real function saved by TempCase
        llm._embed_dim = None

    def tearDown(self):
        llm.requests = self._real_requests
        super().tearDown()

    def plug(self, responses):
        fake = FakeRequests(responses)
        llm.requests = fake
        return fake


class SanitizeTest(support.TempCase):
    def test_strips_full_tool_block(self):
        out = llm.sanitize('Result <|tool_call_start|>[google("x")]<|tool_call_end|> clean')
        self.assertNotIn("<|", out)
        self.assertIn("clean", out)

    def test_strips_stray_control_token(self):
        self.assertNotIn("<|", llm.sanitize("alpha <|padding|> beta"))

    def test_strips_pseudo_calls(self):
        for bad in ('google(query="ephedrine")', "search('x')", "functions.tool(1)"):
            self.assertNotIn(bad, llm.sanitize(f"prose {bad} more prose"), bad)

    def test_keeps_plain_prose_untouched(self):
        text = "Ephedrine raises blood pressure [E1]."
        self.assertEqual(llm.sanitize(text), text)

    def test_empty_and_none(self):
        self.assertEqual(llm.sanitize(""), "")
        self.assertEqual(llm.sanitize(None), "")

    def test_does_not_eat_normal_brackets(self):
        """Citation tags are the product — scrubbing must leave [E1] alone."""
        self.assertIn("[E1]", llm.sanitize("Answer [E1] and [E2]."))


class ChatRetryTest(LLMRequestCase):
    def test_accepts_first_good_answer(self):
        fake = self.plug([FakeResponse(chat_body("clean answer [E1]"))])
        out = llm.chat_llm([{"role": "user", "content": "q"}])
        self.assertEqual(out, "clean answer [E1]")
        self.assertEqual(len(fake.sent), 1, "a good first pass must not retry")

    def test_grows_budget_when_thinking_ate_it(self):
        """finish=length with almost no text is the failure the ladder exists for."""
        fake = self.plug([FakeResponse(chat_body("", finish="length")),
                          FakeResponse(chat_body("real answer", finish="stop"))])
        out = llm.chat_llm([{"role": "user", "content": "q"}], max_tokens=100)
        self.assertEqual(out, "real answer")
        self.assertEqual(len(fake.sent), 2)
        self.assertGreater(fake.sent[1]["max_tokens"], fake.sent[0]["max_tokens"])
        self.assertEqual(fake.sent[1]["temperature"], 0.0, "retries must go greedy")

    def test_salvages_prose_from_control_token_bleed(self):
        long_prose = "x" * 200
        fake = self.plug([FakeResponse(chat_body(f"{long_prose} <|tool_call_start|>junk"))])
        out = llm.chat_llm([{"role": "user", "content": "q"}])
        self.assertNotIn("<|", out)
        self.assertIn("x" * 50, out, "usable prose is kept, not thrown away")
        self.assertEqual(len(fake.sent), 1)

    def test_all_garbage_returns_empty_after_three_attempts(self):
        fake = self.plug([FakeResponse(chat_body("<|tool_call_start|>g", finish="length"))] * 3)
        out = llm.chat_llm([{"role": "user", "content": "q"}])
        self.assertEqual(out, "")
        self.assertEqual(len(fake.sent), 3, "the ladder is capped at 3 attempts")

    def test_short_but_complete_answer_is_not_retried(self):
        fake = self.plug([FakeResponse(chat_body("Yes [E1].", finish="stop"))])
        self.assertEqual(llm.chat_llm([{"role": "user", "content": "q"}]), "Yes [E1].")
        self.assertEqual(len(fake.sent), 1)

    def test_on_retry_callback_sees_growing_budgets(self):
        self.plug([FakeResponse(chat_body("", finish="length")),
                   FakeResponse(chat_body("answer", finish="stop"))])
        budgets = []
        llm.chat_llm([{"role": "user", "content": "q"}], max_tokens=100,
                     on_retry=lambda b: budgets.append(b))
        self.assertEqual(len(budgets), 1)
        self.assertGreater(budgets[0], 100)


class ChatStreamTest(LLMRequestCase):
    def sse(self, delta, finish=None):
        import json
        return "data: " + json.dumps({"choices": [{"delta": {"content": delta},
                                                   "finish_reason": finish}]})

    def test_deltas_reach_the_ui_in_order(self):
        lines = [self.sse("Ephedrine "), self.sse("raises "), self.sse("pressure"),
                 "data: [DONE]"]
        self.plug([FakeResponse(None, stream_lines=lines)])
        seen = []
        out = llm.chat_llm([{"role": "user", "content": "q"}], on_delta=seen.append)
        self.assertEqual("".join(seen), "Ephedrine raises pressure")
        self.assertEqual(out, "Ephedrine raises pressure")

    def test_control_token_split_across_deltas_is_scrubbed(self):
        """The tokenizer can emit '<|tool' then '_call_start|>'; the UI must never see it.

        Prose is >80 chars because chat_llm discards a SHORT stream that was
        mostly control tokens (that is the retry path, tested separately).
        """
        head = ("Ephedrine raises systolic blood pressure and heart rate in anaesthetised "
                "patients, which is why it is used to treat spinal hypotension. ")
        tail = "It is contraindicated in patients with severe hypertension."
        lines = [self.sse(head + "<|tool"),
                 self.sse("_call_start|>junk<|tool_call_end|> " + tail),
                 "data: [DONE]"]
        self.plug([FakeResponse(None, stream_lines=lines)])
        seen = []
        out = llm.chat_llm([{"role": "user", "content": "q"}], on_delta=seen.append)
        joined = "".join(seen)
        self.assertNotIn("<|", joined)
        self.assertNotIn("|>", joined)
        self.assertNotIn("junk", joined)
        self.assertIn("Ephedrine raises systolic blood pressure", joined)
        self.assertIn(tail, joined)
        self.assertEqual(out, joined.strip())

    def test_pure_control_output_is_discarded_after_the_ladder(self):
        """A stream that is only tool tokens must not be shown as an answer."""
        lines = [self.sse("<|tool_call_start|>"), self.sse("[google(q)]"),
                 self.sse("<|tool_call_end|>"), "data: [DONE]"]
        fake = self.plug([FakeResponse(None, stream_lines=lines)] * 3)
        seen = []
        out = llm.chat_llm([{"role": "user", "content": "q"}], on_delta=seen.append)
        self.assertEqual(out, "")
        self.assertEqual(len(fake.sent), 3, "empty verdict triggers the full retry ladder")

    def test_falls_back_to_non_stream_when_streaming_breaks(self):
        """Some OpenAI servers reject stream:true; the answer must still arrive."""
        fake = self.plug([])
        calls = []

        def post(url, json=None, timeout=None, stream=False, **kw):
            calls.append(bool(stream))
            if stream:
                raise RuntimeError("no stream support")
            return FakeResponse(chat_body("fallback answer [E1]"))
        fake.post = post
        seen = []
        out = llm.chat_llm([{"role": "user", "content": "q"}], on_delta=seen.append)
        self.assertEqual(out, "fallback answer [E1]")
        self.assertTrue(any(calls), "streaming was attempted first")
        self.assertIn("fallback answer", "".join(seen))


class AuthTest(LLMRequestCase):
    """A keyed provider is unreachable without a Bearer header, and a local
    server must not receive one it did not ask for."""

    def test_no_key_sends_no_header(self):
        fake = self.plug([chat_body(" Ephedrine raises pressure.")])
        llm.chat_llm([{"role": "user", "content": "q"}])
        self.assertEqual(fake.headers, [{}])

    def test_key_goes_to_chat_and_embeddings(self):
        config.API_KEY = "sk-test"
        fake = self.plug([chat_body("Ephedrine raises blood pressure.")])
        llm.chat_llm([{"role": "user", "content": "q"}])
        self.assertEqual(fake.headers[-1], {"Authorization": "Bearer sk-test"})
        fake = self.plug([{"data": [{"index": 0, "embedding": [1.0, 0.0, 0.0]}]}])
        llm.embed(["ephedrine"])
        self.assertEqual(fake.headers[-1], {"Authorization": "Bearer sk-test"})

    def test_streaming_carries_the_key(self):
        config.API_KEY = "sk-test"
        lines = ["data: " + json.dumps({"choices": [{"delta": {"content": "Ephedrine works."},
                                                     "finish_reason": None}]}),
                 "data: [DONE]"]
        fake = self.plug([FakeResponse(None, stream_lines=lines)])
        llm.chat_llm([{"role": "user", "content": "q"}], on_delta=lambda t: None)
        self.assertEqual(fake.headers[-1], {"Authorization": "Bearer sk-test"})

    def test_model_autodetect_is_authenticated(self):
        """A 401 on /v1/models used to be swallowed into model id "default"."""
        config.API_KEY = "sk-test"
        config.CHAT_MODEL = ""
        llm._chat_model = None
        fake = self.plug([])
        self.assertEqual(llm.chat_model(), "test-model")
        self.assertEqual(fake.headers, [{"Authorization": "Bearer sk-test"}])

class ChatModelTest(LLMRequestCase):
    def setUp(self):
        super().setUp()
        config.CHAT_MODEL = ""      # TempCase pins a model; these test detection

    def test_auto_detects_from_models_endpoint(self):
        llm._chat_model = None
        self.plug([])
        self.assertEqual(llm.chat_model(), "test-model")

    def test_config_override_wins(self):
        support.config.CHAT_MODEL = "pinned-model"
        try:
            llm._chat_model = None
            self.plug([])
            self.assertEqual(llm.chat_model(), "pinned-model")
        finally:
            support.config.CHAT_MODEL = ""

    def test_embedding_dimension_comes_from_response(self):
        """The vec table is sized from the first vector, so any embedder dim works."""
        dim = 5
        self.plug([FakeResponse({"data": [{"index": 0,
                                           "embedding": [1.0] + [0.0] * (dim - 1)}]})])
        llm._embed_dim = None
        try:
            self.assertEqual(llm.embed_dim(), dim)
        finally:
            llm._embed_dim = support.DIM


class EmbedRequestTest(LLMRequestCase):
    """The embeddings call must stay spec-minimal so non-llama.cpp servers accept it."""

    def test_no_model_field_unless_configured(self):
        support.config.EMBED_MODEL = ""
        fake = self.plug([FakeResponse({"data": [{"index": 0, "embedding": [0.0] * 8 + [1.0]}]})])
        llm.embed(["x"])
        self.assertNotIn("model", fake.sent[0])

    def test_model_field_sent_when_configured(self):
        support.config.EMBED_MODEL = "some-embedder"
        fake = self.plug([FakeResponse({"data": [{"index": 0, "embedding": [0.0] * 8 + [1.0]}]})])
        llm.embed(["x"])
        self.assertEqual(fake.sent[0]["model"], "some-embedder")

    def test_no_llama_only_normalize_field(self):
        support.config.EMBED_MODEL = ""
        fake = self.plug([FakeResponse({"data": [{"index": 0, "embedding": [0.0] * 8 + [1.0]}]})])
        llm.embed(["x"])
        self.assertNotIn("normalize", fake.sent[0])

    def test_vectors_are_l2_normalized_client_side(self):
        import numpy
        fake = self.plug([FakeResponse({"data": [{"index": 0, "embedding": [3.0, 4.0]}]})])
        v = llm.embed(["x"])[0]
        self.assertAlmostEqual(float(numpy.linalg.norm(v)), 1.0)

    def test_zero_vector_is_rejected_not_silently_used(self):
        self.plug([FakeResponse({"data": [{"index": 0, "embedding": [0.0, 0.0]}]})])
        with self.assertRaises(RuntimeError):
            llm.embed(["x"])

    def test_oversized_input_is_split_and_pooled_not_dropped(self):
        """llama.cpp answers 500 — not truncation — when one input exceeds its
        physical batch, so a long chunk must not be able to kill an index build."""
        import numpy
        support.config.EMBED_MAX_CHARS = 10
        one = lambda v: FakeResponse({"data": [{"index": 0, "embedding": v}]})
        too_long = FakeResponse({"error": {"message": "too large"}}, status=500,
                                text="input (535 tokens) is too large to process")
        fake = self.plug([too_long, one([1.0, 0.0]), one([0.0, 1.0]), one([1.0, 0.0])])
        try:
            v = llm.embed(["x" * 25])[0]
        finally:
            support.config.EMBED_MAX_CHARS = 800
        self.assertEqual(len(fake.sent), 4)                   # 1 refused + 3 parts
        self.assertAlmostEqual(float(numpy.linalg.norm(v)), 1.0, places=5)
        self.assertEqual([round(x, 3) for x in v.tolist()], [0.894, 0.447])

    def test_per_item_retry_keeps_the_query_instruction(self):
        """A poisoned batch must fall back to the same query semantics, or the
        fused lane compares instructed and uninstructed vectors against one index."""
        support.config.QUERY_INSTRUCT = "Instruct: q\nQuery: "
        one = {"data": [{"index": 0, "embedding": [1.0, 0.0]}]}
        fake = self.plug([FakeResponse(one, status=500, text="batch poisoned"),
                          FakeResponse(one), FakeResponse(one)])
        llm.embed(["alpha", "beta"], is_query=True)
        self.assertEqual([s["input"][0] for s in fake.sent[1:]],
                         ["Instruct: q\nQuery: alpha", "Instruct: q\nQuery: beta"])

    def test_query_instruction_prefix(self):
        support.config.QUERY_INSTRUCT = "Instruct: q\nQuery: "
        one = {"data": [{"index": 0, "embedding": [1.0, 0.0]}]}
        fake = self.plug([FakeResponse(one), FakeResponse(one)])
        llm.embed(["hello"], is_query=True)
        self.assertTrue(fake.sent[0]["input"][0].startswith("Instruct: "))
        llm.embed(["hello"], is_query=False)
        self.assertEqual(fake.sent[-1]["input"][0], "hello", "passages must never get the prefix")
