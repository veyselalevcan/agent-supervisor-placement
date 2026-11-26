# Offline tests for p4h-3.6 mode g7 (AgentDojo adapter, gap G7). They need the agentdojo package (pip install agentdojo==0.1.35) and are skipped without it.
# Scripted agent and supervisor, real AgentDojo suites, attack text, utility and security checks.
import json, os, sys, tempfile, unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import p4h

try:
    import agentdojo                                   # noqa: F401
    from agentdojo.task_suite.load_suites import get_suite
    from agentdojo.attacks.attack_registry import load_attack
    HAVE_AD = True
except Exception:
    HAVE_AD = False

ALLOW = '{"verdict": "allow"}'


def agent_replies(replies):
    it = iter(replies)
    def gen(msgs, seed):
        return next(it, '{"final": "done"}'), 10, 5, 0.01
    return gen


def scripted_sup(replies):
    it = iter(replies)
    seen = []
    def sup(messages):
        seen.append(messages)
        return next(it, ALLOW), 50, 8, 0.02
    sup.seen = seen
    return sup


def call(tool, **args):
    return json.dumps({"tool": tool, "args": args})


@unittest.skipUnless(HAVE_AD, "agentdojo not installed")
class G7Base(unittest.TestCase):
    def setUp(self):
        p4h.CONFIG.update(agent_model="Qwen/Qwen3-4B", repeat=0, arms=["0", "Bq"], limit_cases=None, g7_users=4, g7_injs=2)
        self.tmp = tempfile.TemporaryDirectory()
        self.log = p4h.Logger(os.path.join(self.tmp.name, "ev.jsonl"))
        self.suite = get_suite("v1.2.2", "banking")

    def tearDown(self):
        self.tmp.cleanup()

    def attack(self, arm="0"):
        pipe = p4h.ad_build_pipeline(arm, agent_replies([]), scripted_sup([]), "", 0, dict(gen_s=0.0, sup_s=0.0, model_calls=0, sup_calls=0, sup_blocks=0, sup_overruled=0, sup_parse_fail=0, tool_calls=0))
        return load_attack(p4h.AD_ATTACK, self.suite, pipe)

    def events(self):
        return [json.loads(l) for l in open(self.log.path, encoding="utf-8")]


@unittest.skipUnless(HAVE_AD, "agentdojo not installed")
class TestHelpers(G7Base):
    def test_suite_sizes_of_v1_2_2(self):
        got = {n: (len(get_suite("v1.2.2", n).user_tasks), len(get_suite("v1.2.2", n).injection_tasks)) for n in ("workspace", "travel", "banking", "slack")}
        self.assertEqual(got, {"workspace": (40, 14), "travel": (20, 7), "banking": (16, 9), "slack": (21, 5)})

    def test_task_subset_is_deterministic_and_sized(self):
        users, injs = p4h.ad_task_lists(self.suite)
        self.assertEqual((len(users), len(injs)), (4, 2))
        self.assertEqual((users, injs), p4h.ad_task_lists(self.suite))
        self.assertEqual(injs, ["injection_task_0", "injection_task_1"])

    def test_read_only_prefix_heuristic(self):
        self.assertFalse(p4h.ad_needs_check("get_balance"))
        self.assertFalse(p4h.ad_needs_check("read_file"))
        self.assertTrue(p4h.ad_needs_check("send_money"))
        self.assertTrue(p4h.ad_needs_check("update_password"))

    def test_chat_lists_tools(self):
        from agentdojo.functions_runtime import FunctionsRuntime
        rt = FunctionsRuntime(self.suite.tools)
        chat = p4h.ad_chat(rt, "pay the bill", [])
        self.assertIn("send_money(", chat[0]["content"])
        self.assertEqual(chat[1], {"role": "user", "content": "pay the bill"})

    def test_attack_text_reaches_a_tool_output(self):
        atk = self.attack()
        user = self.suite.get_user_task_by_id("user_task_0")
        inj = self.suite.get_injection_task_by_id("injection_task_0")
        injections = atk.attack(user, inj)
        self.assertTrue(injections)
        self.assertIn(inj.GOAL[:30], " ".join(injections.values()))


@unittest.skipUnless(HAVE_AD, "agentdojo not installed")
class TestRuns(G7Base):
    def test_benign_run_logs_a_row_and_unique_ids(self):
        row = p4h.ad_run(self.suite, "0", "user_task_0", None, agent_replies(['{"final": "done"}']), scripted_sup([]), self.attack(), self.log)
        self.assertEqual((row["group"], row["attack_success"], row["sup_calls"]), ("benign", 0, 0))
        self.assertIn("/g7/banking/user_task_0/benign/0/r0", row["run_id"])
        with self.assertRaises(AssertionError):
            p4h.ad_run(self.suite, "0", "user_task_0", None, agent_replies([]), scripted_sup([]), self.attack(), self.log)

    def test_read_tool_is_not_checked_and_write_tool_is_checked(self):
        sup = scripted_sup([ALLOW])
        agent = agent_replies([call("get_balance"), call("send_money", recipient="US133000000121212121212", amount=1.0, subject="x", date="2022-01-01"), '{"final": "ok"}'])
        row = p4h.ad_run(self.suite, "Bq", "user_task_0", None, agent, sup, self.attack("Bq"), self.log)
        self.assertEqual(row["sup_calls"], 1)                     # only send_money
        self.assertEqual(row["tool_calls"], 2)
        self.assertIn("send_money", sup.seen[0][1]["content"])

    def test_valid_block_stops_the_action(self):
        iban = "US133000000121212121212"
        block = json.dumps({"verdict": "block", "reason": "recipient", "evidence": iban})
        agent = agent_replies([call("send_money", recipient=iban, amount=1.0, subject="x", date="2022-01-01"), '{"final": "done"}'])
        row = p4h.ad_run(self.suite, "Bq", "user_task_0", "injection_task_0", agent, scripted_sup([block]), self.attack("Bq"), self.log)
        self.assertEqual((row["sup_blocks"], row["tool_calls"], row["attack_success"]), (1, 0, 0))

    def test_unfound_quote_is_overruled_and_the_action_runs(self):
        iban = "US133000000121212121212"
        block = json.dumps({"verdict": "block", "reason": "r", "evidence": "text that is not in the action"})
        agent = agent_replies([call("send_money", recipient=iban, amount=1.0, subject="x", date="2022-01-01"), '{"final": "done"}'])
        row = p4h.ad_run(self.suite, "Bq", "user_task_0", "injection_task_0", agent, scripted_sup([block]), self.attack("Bq"), self.log)
        self.assertEqual((row["sup_blocks"], row["sup_overruled"], row["tool_calls"]), (0, 1, 1))

    def test_every_supervisor_decision_is_logged_with_its_reply(self):
        iban = "US133000000121212121212"
        block = json.dumps({"verdict": "block", "reason": "r", "evidence": "text not in the action"})
        agent = agent_replies([call("send_money", recipient=iban, amount=1.0, subject="x", date="2022-01-01"), '{"final": "done"}'])
        p4h.ad_run(self.suite, "Bq", "user_task_0", "injection_task_0", agent, scripted_sup([block]), self.attack("Bq"), self.log)
        ev = self.events()
        dec = [e for e in ev if e["kind"] == "sup_decision"]
        self.assertEqual(len(dec), 1)
        self.assertEqual((dec[0]["verdict"], dec[0]["overruled"], dec[0]["tool"]), ("block", True, "send_money"))
        self.assertIn("text not in the action", dec[0]["payload"])
        self.assertTrue(dec[0]["run_id"].endswith("/Bq/r0"))
        self.assertEqual([e["kind"] for e in ev][-1], "outcome")

    def test_arm_0_never_calls_the_supervisor(self):
        sup = scripted_sup([])
        agent = agent_replies([call("send_money", recipient="US133000000121212121212", amount=1.0, subject="x", date="2022-01-01"), '{"final": "done"}'])
        row = p4h.ad_run(self.suite, "0", "user_task_0", "injection_task_0", agent, sup, self.attack(), self.log)
        self.assertEqual((row["sup_calls"], len(sup.seen)), (0, 0))

    def test_attack_success_uses_the_agentdojo_security_check(self):
        # AgentDojo's own predicate decides: a transfer to the attacker IBAN counts (observed: even with an empty subject, so the
        # banking predicate is loose, like our sabotage predicate); doing nothing does not
        iban = "US133000000121212121212"
        hit = agent_replies([call("send_money", recipient=iban, amount=1.0, subject="", date="2022-01-01"), '{"final": "done"}'])
        row = p4h.ad_run(self.suite, "0", "user_task_0", "injection_task_0", hit, scripted_sup([]), self.attack(), self.log)
        self.assertEqual(row["attack_success"], 1)
        miss = agent_replies(['{"final": "done"}'])
        row2 = p4h.ad_run(self.suite, "0", "user_task_0", "injection_task_1", miss, scripted_sup([]), self.attack(), self.log)
        self.assertEqual(row2["attack_success"], 0)

    def test_unparsable_reply_is_retried_then_returned_as_text(self):
        row = p4h.ad_run(self.suite, "0", "user_task_0", None, agent_replies(["not json", "still not json", "nope"]), scripted_sup([]), self.attack(), self.log)
        self.assertEqual(row["model_calls"], p4h.AD_MAX_TRIES)

    def test_summary(self):
        rows = [p4h.ad_run(self.suite, "0", "user_task_0", None, agent_replies([]), scripted_sup([]), self.attack(), self.log),
                p4h.ad_run(self.suite, "0", "user_task_0", "injection_task_0", agent_replies([]), scripted_sup([]), self.attack(), self.log)]
        s = p4h.summarize_g7(rows)["0"]
        self.assertEqual((s["runs"], s["asr"][1], s["utility_benign"][1]), (2, 1, 1))


@unittest.skipUnless(HAVE_AD, "agentdojo not installed")
class TestFailClosed(G7Base):
    """p4h-3.10: an unreadable supervisor reply blocks in the AgentDojo adapter, as in the main harness (V42 had 20 of 208 such replies treated as overruled)."""
    IBAN = "US133000000121212121212"

    def run_with(self, arm, reply):
        agent = agent_replies([call("send_money", recipient=self.IBAN, amount=1.0, subject="x", date="2022-01-01"), '{"final": "done"}'])
        row = p4h.ad_run(self.suite, arm, "user_task_0", "injection_task_0", agent, scripted_sup([reply]), self.attack(arm), self.log)
        dec = [e for e in self.events() if e["kind"] == "sup_decision"]
        return row, dec

    def test_unreadable_block_reply_blocks_under_Bq(self):
        row, dec = self.run_with("Bq", "I think this should be blocked, it looks bad")
        self.assertEqual((dec[0]["parse_ok"], dec[0]["overruled"]), (False, False))
        self.assertEqual((row["sup_blocks"], row["sup_overruled"], row["sup_parse_fail"]), (1, 0, 1))
        self.assertEqual(row["attack_success"], 0)

    def test_unreadable_block_reply_blocks_under_B(self):
        row, dec = self.run_with("B", "no idea, maybe block {{{")
        self.assertEqual((row["sup_blocks"], row["attack_success"]), (1, 0))

    def test_readable_block_with_missing_quote_is_still_overruled(self):
        row, dec = self.run_with("Bq", json.dumps({"verdict": "block", "reason": "r", "evidence": "text not in the action"}))
        self.assertEqual((dec[0]["overruled"], row["sup_overruled"], row["sup_blocks"]), (True, 1, 0))


class TestAdFit(unittest.TestCase):
    """p4h-3.10: the agent prompt is capped; long tool results shrink first and the system and task messages stay."""
    @staticmethod
    def count(chat):
        return sum(len(m["content"]) for m in chat) // 4

    def chat(self):
        return [{"role": "system", "content": "S" * 400}, {"role": "user", "content": "task"},
                {"role": "assistant", "content": "{}"}, {"role": "user", "content": "Tool result: " + "a" * 20000},
                {"role": "assistant", "content": "{}"}, {"role": "user", "content": "Tool result: " + "b" * 4000}]

    def test_under_the_cap_nothing_changes(self):
        c = self.chat()
        self.assertEqual(p4h.ad_fit(c, lambda m: 0, 100), c)

    def test_longest_result_shrinks_first(self):
        out = p4h.ad_fit(self.chat(), self.count, 2000)
        self.assertLessEqual(self.count(out), 2000)
        self.assertEqual(out[0]["content"], "S" * 400)
        self.assertEqual(out[1]["content"], "task")
        self.assertTrue(out[3]["content"].endswith("[shortened]"))

    def test_input_is_not_modified(self):
        c = self.chat(); before = json.dumps(c)
        p4h.ad_fit(c, self.count, 500)
        self.assertEqual(json.dumps(c), before)

    def test_old_messages_are_dropped_when_nothing_can_shrink(self):
        c = [{"role": "system", "content": "S"}, {"role": "user", "content": "task"}] + [{"role": "assistant", "content": "x" * 400} for _ in range(30)]
        out = p4h.ad_fit(c, self.count, 1000)
        self.assertLessEqual(self.count(out), 1000)
        self.assertEqual(out[:2], c[:2])


class TestMakeNb(unittest.TestCase):
    def test_g7_notebook(self):
        import make_nb
        path = make_nb.build("g7", "Qwen/Qwen3-4B", 0, None, out_dir=make_nb.TEST_OUT)
        try:
            src = "".join(json.load(open(path, encoding="utf-8"))["cells"][0]["source"])
        finally:
            os.remove(path)
        self.assertIn('mode="g7"', src)
        self.assertIn('arms=["0", "Bq"]', src)
        self.assertIn("def main_g7", src)


if __name__ == "__main__":
    unittest.main()
