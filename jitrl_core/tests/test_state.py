"""state: signature replacements, CJK bigrams, Jaccard hand cases, intent order."""
from __future__ import annotations

import unittest

from jitrl_core.state import intent_class, jaccard, signature_token_set, task_signature


class TestSignature(unittest.TestCase):
    def test_code_fence_replaced(self):
        toks = task_signature("看这段：```x = 1\ny = 2```然后改")
        self.assertIn("<code>", toks)
        self.assertNotIn("x", toks)

    def test_indented_code_replaced(self):
        toks = task_signature("说明如下：\n    alpha = 1\n    beta = 2\n完事")
        self.assertIn("<code>", toks)
        self.assertNotIn("alpha", toks)

    def test_path_replaced(self):
        toks = task_signature("check src/main.py for details")
        self.assertIn("<path>", toks)
        self.assertNotIn("main", toks)

    def test_url_replaced(self):
        toks = task_signature("see https://example.com/docs?x=1 today")
        self.assertIn("<url>", toks)
        self.assertNotIn("https", toks)

    def test_number_replaced(self):
        toks = task_signature("run it 42 times")
        self.assertIn("<num>", toks)
        self.assertNotIn("42", toks)

    def test_long_quoted_string_replaced(self):
        toks = task_signature('他说的内容是 "a very long quoted string here" 没错')
        self.assertIn("<str>", toks)

    def test_cjk_bigram(self):
        self.assertEqual(task_signature("写测试"), ["写测", "测试"])

    def test_cjk_single_char(self):
        self.assertEqual(task_signature("好"), ["好"])

    def test_lowercase(self):
        self.assertEqual(task_signature("ABC def"), ["abc", "def"])

    def test_placeholder_is_atomic_token(self):
        toks = task_signature("```print(1)```")
        self.assertEqual(toks, ["<code>"])

    def test_jaccard_hand_case(self):
        a = {"a", "b", "c"}
        b = {"b", "c", "d"}
        self.assertAlmostEqual(jaccard(a, b), 2 / 4)

    def test_jaccard_empty_cases(self):
        self.assertEqual(jaccard(set(), set()), 1.0)
        self.assertEqual(jaccard({"a"}, set()), 0.0)

    def test_signature_set_dedups(self):
        # "测试测试" -> bigrams [测试, 试测, 测试] -> set {测试, 试测}
        self.assertEqual(signature_token_set("测试测试"), {"测试", "试测"})


class TestIntentClass(unittest.TestCase):
    def test_spec_examples(self):
        cases = {
            "帮我写单元测试": "code_gen",
            "分析一下这批数据": "data_analysis",
            "重构这段代码": "refactor",
            "你好，谢谢": "chat_qa",
            "继续": "continuation",
            "帮我检索相关论文": "info_retrieval",
            "写一封通知": "doc_writing",
        }
        for msg, expected in cases.items():
            self.assertEqual(intent_class(msg), expected, msg)

    def test_first_match_wins_specificity_order(self):
        # contains both code_gen ("写一个函数") and data_analysis ("分析")
        self.assertEqual(intent_class("分析这段需求，然后写一个函数"), "code_gen")

    def test_fallback_other(self):
        self.assertEqual(intent_class("zzz qqq unusual input"), "other")


if __name__ == "__main__":
    unittest.main()
