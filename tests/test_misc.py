import unittest

from src.misc import Json2Obj, generate_player_names, timeit


class TestJson2Obj(unittest.TestCase):
    def test_flat_dict_access(self):
        obj = Json2Obj({"name": "Alice", "score": 10})
        self.assertEqual(obj.name, "Alice")
        self.assertEqual(obj.score, 10)

    def test_nested_dict_becomes_obj(self):
        obj = Json2Obj({"player": {"name": "Bob", "wins": 3}})
        self.assertEqual(obj.player.name, "Bob")
        self.assertEqual(obj.player.wins, 3)

    def test_list_of_dicts(self):
        obj = Json2Obj({"items": [{"val": 1}, {"val": 2}]})
        self.assertEqual(obj.items[0].val, 1)
        self.assertEqual(obj.items[1].val, 2)

    def test_list_of_primitives(self):
        obj = Json2Obj({"nums": [1, 2, 3]})
        self.assertEqual(obj.nums, [1, 2, 3])


class TestGeneratePlayerNames(unittest.TestCase):
    def test_returns_correct_count(self):
        names = generate_player_names(10)
        self.assertEqual(len(names), 10)

    def test_returns_list_of_strings(self):
        names = generate_player_names(5)
        self.assertIsInstance(names, list)
        for name in names:
            self.assertIsInstance(name, str)

    def test_names_are_unique(self):
        names = generate_player_names(20)
        self.assertEqual(len(names), len(set(names)))

    def test_zero_names(self):
        names = generate_player_names(0)
        self.assertEqual(names, [])


class TestTimeit(unittest.TestCase):
    def test_returns_function_result(self):
        @timeit
        def add(a, b):
            return a + b

        result = add(2, 3)
        self.assertEqual(result, 5)

    def test_works_with_no_args(self):
        @timeit
        def constant():
            return 42

        self.assertEqual(constant(), 42)
