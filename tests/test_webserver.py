# SPDX-License-Identifier: GPL-3.0-only
# search-bot — grounded scientific RAG engine
# Copyright (C) 2026 raaaas
# This program comes with ABSOLUTELY NO WARRANTY; it is free software, and you
# are welcome to redistribute it under GNU GPL-3.0-only terms. See LICENSE.

"""HTTP layer: routing, JSON shapes, and the wiring between the UI and the agent.

agent.ask_agentic is stubbed, so these pin the server's own behaviour — including
the acquire:false path, which used to die on an unbound `events` list before the
answer thread ever started.
"""
import json
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import support  # noqa: E402
from searchbot import agent, trace as trace_mod, webserver  # noqa: E402


class ServerCase(support.TempCase):
    def setUp(self):
        super().setUp()
        self._saved_conn = webserver._conn
        webserver._conn = self.c                       # never open the user's DB
        self.calls = []
        self._real_ask = agent.ask_agentic
        agent.ask_agentic = self._stub_ask
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), webserver.H)
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        agent.ask_agentic = self._real_ask
        webserver._conn = self._saved_conn
        super().tearDown()

    def _stub_ask(self, c, slug, session_id, question, **kw):
        self.calls.append({"slug": slug, "session_id": session_id,
                           "question": question, **kw})
        return {"answer": f"grounded answer for {question} [E1]", "evidence": [],
                "facts_stored": 0, "rounds": []}

    def get(self, path):
        try:
            with urllib.request.urlopen(self.base + path, timeout=10) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    def post(self, path, obj):
        req = urllib.request.Request(self.base + path, data=json.dumps(obj).encode(),
                                     headers={"Content-Type": "application/json"},
                                     method="POST")
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    def run_ask(self, body):
        """POST /api/ask then follow the trace to its result."""
        code, started = self.post("/api/ask", body)
        self.assertEqual(code, 200)
        rid = started["run_id"]
        for _ in range(200):
            _, t = self.get(f"/api/trace?id={rid}")
            if t.get("done"):
                return t
            time.sleep(0.05)
        self.fail("ask never finished")


class AskEndpointTest(ServerCase):
    def test_acquire_false_returns_an_answer(self):
        """Regression: this path raised NameError before the answer thread ran."""
        trace = self.run_ask({"slug": "ephedra", "question": "ephedrine hypotension?",
                              "acquire": False})
        self.assertNotIn("error", trace["result"], json.dumps(trace["events"][-3:]))
        self.assertIn("[E1]", trace["result"]["answer"])
        self.assertEqual(self.calls[0]["acquire"], False)

    def test_acquire_defaults_to_true(self):
        """Omitting the field must not silently disable gathering."""
        self.run_ask({"slug": "ephedra", "question": "ephedrine hypotension?"})
        self.assertTrue(self.calls[0].get("acquire", True))

    def test_ranking_options_reach_the_agent_filtered(self):
        self.run_ask({"slug": "ephedra", "question": "ephedrine?", "acquire": False,
                      "year_after": "2015", "recency": 0.4, "session_id": "s1",
                      "topk": 5})
        rank = self.calls[0]["rank"]
        self.assertEqual(rank, {"year_after": 2015, "recency": 0.4})
        self.assertEqual(self.calls[0]["topk"], 5)
        self.assertEqual(self.calls[0]["session_id"], "s1")

    def test_trace_carries_the_phases_the_ui_renders(self):
        trace = self.run_ask({"slug": "ephedra", "question": "ephedrine?", "acquire": False})
        phases = [e["phase"] for e in trace["events"]]
        self.assertIn("done", phases)
        self.assertNotIn("error", phases)

    def test_agent_failure_becomes_an_error_result_not_a_hang(self):
        def boom(*a, **kw):
            raise RuntimeError("model server down")
        agent.ask_agentic = boom
        trace = self.run_ask({"slug": "ephedra", "question": "ephedrine?", "acquire": False})
        self.assertIn("error", trace["result"])
        self.assertIn("model server down", trace["result"]["error"])


class ReadEndpointsTest(ServerCase):
    def test_searches_listing_includes_doc_and_chunk_counts(self):
        from searchbot import db
        self.add_doc("ephedra", "Ephedrine and hypotension", "2001")
        db.upsert_search(self.c, "ephedra", "Ephedra studies", "")
        code, rows = self.get("/api/searches")
        self.assertEqual(code, 200)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["slug"], "ephedra")
        self.assertEqual((rows[0]["docs"], rows[0]["chunks"]), (1, 1))

    def test_creating_a_search_folder_slugifies_the_name(self):
        code, body = self.post("/api/searches", {"slug": "New Topic!", "title": "New Topic"})
        self.assertEqual(code, 200)
        self.assertEqual(body["slug"], "new-topic")
        _, rows = self.get("/api/searches")
        self.assertIn("new-topic", [r["slug"] for r in rows])

    def test_status_lists_jobs(self):
        from searchbot import db
        db.job_set(self.c, "ephedra", "index", "queued")
        code, rows = self.get("/api/status?slug=ephedra")
        self.assertEqual(code, 200)
        self.assertEqual(rows[0]["state"], "queued")

    def test_history_round_trips_turns(self):
        from searchbot import memory
        memory.add_turn(self.c, "sess1", "ephedra", "user", "ephedrine?")
        code, rows = self.get("/api/history?session=sess1")
        self.assertEqual(code, 200)
        self.assertEqual(rows[0]["role"], "user")
        self.assertEqual(rows[0]["text"], "ephedrine?")

    def test_unknown_run_id_is_404(self):
        code, body = self.get("/api/trace?id=nope")
        self.assertEqual(code, 404)
        self.assertIn("error", body)

    def test_unknown_path_is_404(self):
        self.assertEqual(self.get("/api/nope")[0], 404)


class SlugifyTest(support.TempCase):
    def test_slugs_cannot_escape_the_search_directory(self):
        for bad in ("../../etc/passwd", "a b c", "Ephedra!", "", "-----"):
            s = webserver._slugify(bad)
            self.assertNotIn("/", s)
            self.assertNotIn("..", s)
            self.assertTrue(s.replace("-", "").replace("_", "") or s == "untitled", s)
            self.assertLessEqual(len(s), 48)

    def test_blank_input_gets_a_usable_name(self):
        self.assertEqual(webserver._slugify("///"), "untitled")


class StatsEndpointTest(ServerCase):
    def setUp(self):
        super().setUp()
        from searchbot import db
        self.add_doc("ephedra", "Ephedrine and hypotension", "2001")
        db.upsert_search(self.c, "ephedra", "Ephedra studies", "")

    def test_stats_counts_the_corpus_it_actually_holds(self):
        code, s = self.get("/api/stats")
        self.assertEqual(code, 200)
        self.assertEqual((s["total_folders"], s["total_docs"], s["total_chunks"]), (1, 1, 1))
        self.assertEqual(s["vector_dimension"], support.DIM)
        self.assertEqual(s["folders"][0]["slug"], "ephedra")
        self.assertGreater(s["folders"][0]["bytes"], 0)
        self.assertRegex(s["formatted_size"], r"^\d+(\.\d+)? (B|KB|MB|GB)$")

    def test_stats_reports_the_embedder_without_inventing_a_name(self):
        _, s = self.get("/api/stats")
        e = s["embedder"]
        # Nothing has stamped an embedder, but the table itself says 32d and so
        # does the active embedder: that is a match, not a missing fact.
        self.assertEqual(e["indexed_dim"], support.DIM)
        self.assertEqual(e["indexed_model"], None)
        self.assertEqual(e["active_model"], "unknown")   # no server, no name
        self.assertFalse(e["dim_mismatch"])
        self.assertFalse(e["model_mismatch"])

    def test_stats_flags_a_index_built_at_a_different_width(self):
        from searchbot import db
        db.set_setting(self.c, "indexed_embedder", {"model": "old-embedder", "dim": 768})
        _, s = self.get("/api/stats")
        e = s["embedder"]
        self.assertTrue(e["dim_mismatch"], "32d active vs a 768d stamp must be reported")
        # The other half of the truth: with no name from the embed server, a
        # model swap cannot be asserted — only the width is provable.
        self.assertFalse(e["model_mismatch"])

    def test_an_unstamped_index_is_still_compared_on_its_table_width(self):
        """Old databases predate stamping; the vec table's own width is the fact
        that matters, because inserting a different width is what would fail."""
        from searchbot import db
        db.reset_vec_table(self.c, 768)
        _, s = self.get("/api/stats")
        self.assertTrue(s["embedder"]["dim_mismatch"])
        self.assertEqual(s["vector_dimension"], 768)

    def test_stats_counts_chunks_that_lost_their_vectors(self):
        self.c.execute("DELETE FROM chunks_vec")
        self.c.commit()
        _, s = self.get("/api/stats")
        self.assertEqual(s["embedder"]["chunks_missing_vectors"], 1)

    def test_stats_shares_the_knobs_the_settings_panel_edits(self):
        _, s = self.get("/api/stats")
        self.assertEqual(set(s["settings"]) >= set(webserver.DEFAULT_SETTINGS), True)
        self.assertEqual(s["engine"]["chunk_chars"], webserver.config.CHUNK_CHARS)


class SettingsEndpointTest(ServerCase):
    def test_get_returns_defaults_and_the_server_is_the_authority(self):
        code, body = self.get("/api/settings")
        self.assertEqual(code, 200)
        self.assertEqual(body["settings"], webserver.DEFAULT_SETTINGS)
        self.assertEqual(body["defaults"], webserver.DEFAULT_SETTINGS)

    def test_saved_settings_become_the_defaults_for_ask(self):
        code, body = self.post("/api/settings", {"topk": 3, "temperature": 0.7, "recency": 0.05})
        self.assertEqual(code, 200)
        self.assertEqual(body["settings"]["topk"], 3)
        self.run_ask({"slug": "ephedra", "question": "ephedrine?", "acquire": False})
        self.assertEqual(self.calls[0]["topk"], 3)
        self.assertEqual(self.calls[0]["temperature"], 0.7)
        self.assertEqual(self.calls[0]["rank"]["recency"], 0.05)

    def test_a_value_in_the_ask_body_still_beats_the_saved_setting(self):
        self.post("/api/settings", {"topk": 3})
        self.run_ask({"slug": "ephedra", "question": "ephedrine?", "acquire": False, "topk": 9})
        self.assertEqual(self.calls[0]["topk"], 9)

    def test_choosing_a_chat_model_is_picked_up_by_the_llm_client(self):
        _, body = self.post("/api/settings", {"chat_model": "other-model"})
        self.assertEqual(body["settings"]["chat_model"], "other-model")
        self.assertEqual(webserver.config.CHAT_MODEL, "other-model")

    def test_embedding_model_change_reports_whether_reindex_is_due(self):
        from searchbot import db
        db.set_setting(self.c, "indexed_embedder", {"model": "a", "dim": support.DIM})
        _, body = self.post("/api/settings", {"embed_model": "b"})
        self.assertTrue(body["needs_reindex"])
        self.assertFalse(body["embedder"]["dim_mismatch"], "same width, different model")

    def test_out_of_range_numbers_are_rejected_not_stored(self):
        for bad in ({"topk": 0}, {"topk": 999}, {"temperature": 5}, {"topk": "many"}):
            code, body = self.post("/api/settings", bad)
            self.assertEqual(code, 400, bad)
            self.assertIn("error", body)
        self.assertEqual(self.get("/api/settings")[1]["settings"]["topk"],
                         webserver.DEFAULT_SETTINGS["topk"])

    def test_acquire_false_survives_the_settings_merge(self):
        """A stored `acquire: true` must not override an explicit false in the body."""
        self.post("/api/settings", {"acquire": True})
        self.run_ask({"slug": "ephedra", "question": "ephedrine?", "acquire": False})
        self.assertEqual(self.calls[0]["acquire"], False)


class ReindexEndpointTest(ServerCase):
    def setUp(self):
        super().setUp()
        from searchbot import db
        self.add_doc("ephedra", "Ephedrine and hypotension", "2001")
        db.upsert_search(self.c, "ephedra", "Ephedra studies", "")

    def _wait(self, job_id):
        for _ in range(200):
            _, rows = self.get("/api/status")          # unfiltered: a whole-corpus
            job = next((j for j in rows if j["id"] == job_id), None)   # reindex is
            if job and job["state"] in ("done", "error"):              # filed under
                return job                               # slug "all"
            time.sleep(0.05)
        self.fail("reindex job never finished")

    def test_reindex_rebuilds_the_vector_table_and_stamps_the_embedder(self):
        code, body = self.post("/api/reindex", {"slug": "ephedra"})
        self.assertEqual(code, 200)
        job = self._wait(body["job"])
        self.assertEqual(job["state"], "done", job["info"])
        import json as _json
        out = _json.loads(job["info"])
        self.assertEqual((out["chunks"], out["docs"], out["dim"]), (1, 1, support.DIM))
        _, s = self.get("/api/stats")
        self.assertEqual(s["embedder"]["chunks_missing_vectors"], 0)
        self.assertFalse(s["embedder"]["dim_mismatch"])

    def test_unknown_slug_is_rejected_before_a_job_starts(self):
        code, body = self.post("/api/reindex", {"slug": "nope"})
        self.assertEqual(code, 404)
        self.assertIn("error", body)

    def test_null_slug_re_embeds_every_folder(self):
        _, body = self.post("/api/reindex", {"slug": None})
        self.assertEqual(body["slug"], "all")
        self.assertEqual(self._wait(body["job"])["state"], "done")


class DeleteSearchTest(ServerCase):
    def delete(self, path):
        req = urllib.request.Request(self.base + path, method="DELETE")
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    def setUp(self):
        super().setUp()
        from searchbot import db, memory
        self.slug = "ephedra"
        d = webserver.config.SEARCH_DIR / self.slug / "pdf"
        d.mkdir(parents=True, exist_ok=True)
        (d / "ephedrine.txt").write_text("a paper")
        self.add_doc(self.slug, "Ephedrine and hypotension", "2001")
        db.upsert_search(self.c, self.slug, "Ephedra studies", str(d))
        memory.add_turn(self.c, "sess1", self.slug, "user", "ephedrine?")

    def test_forgetting_a_folder_drops_its_index_rows(self):
        code, body = self.delete(f"/api/searches/{self.slug}")
        self.assertEqual(code, 200)
        self.assertEqual((body["docs"], body["chunks"]), (1, 1))
        self.assertEqual(self.get("/api/searches")[1], [])
        for table in ("docs", "chunks", "chunks_fts"):
            self.assertEqual(self.c.execute(f"SELECT COUNT(*) n FROM {table}").fetchone()["n"],
                             0, table)
        self.assertEqual(self.c.execute("SELECT COUNT(*) n FROM chunks_vec").fetchone()["n"], 0)

    def test_forgetting_a_folder_keeps_the_files_it_was_built_from(self):
        _, body = self.delete(f"/api/searches/{self.slug}")
        self.assertTrue((webserver.config.SEARCH_DIR / self.slug / "pdf" / "ephedrine.txt").exists())
        self.assertEqual(body["kept_files"], str(webserver.config.SEARCH_DIR / self.slug))

    def test_forgetting_a_folder_keeps_the_answers_already_given(self):
        self.delete(f"/api/searches/{self.slug}")
        self.assertEqual(self.get("/api/history?session=sess1")[1][0]["text"], "ephedrine?")

    def test_unknown_or_malformed_slug_is_not_deleted(self):
        self.assertEqual(self.delete("/api/searches/nope")[0], 404)
        self.assertEqual(self.delete("/api/searches/../etc")[0], 404)
        self.assertEqual(self.delete("/api/searches")[0], 404)
        self.assertEqual(self.c.execute("SELECT COUNT(*) n FROM searches").fetchone()["n"], 1)
