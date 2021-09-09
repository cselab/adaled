from base import TestCase
from adaled.utils.dataclasses_ import DataclassMixin, ParseError, dataclass, field

from typing import Any, Dict, List, Tuple

@dataclass
class InnerInner(DataclassMixin):
    a: int = 1


@dataclass
class Inner(DataclassMixin):
    b: int = 2
    inner: InnerInner = field(InnerInner)


@dataclass
class Outer(DataclassMixin):
    c: int = 3
    # Test same names.
    inner: Inner = field(Inner)


class TestDataclasses(TestCase):
    def test_format_argv(self):
        fmt = Outer().format_argv
        self.assertEqual(fmt('color', 'red'), 'color=red')
        self.assertEqual(fmt('color', 4 / 3), f'color={4 / 3!r}')
        self.assertEqual(fmt('color', {'aa': 5}), 'color:json={"aa":5}')
        self.assertEqual(fmt('color', [10, 20, 30]), 'color:json=[10,20,30]')

        # Not this function's job to escape characters.
        self.assertEqual(fmt('color', '\'\"'), 'color=\'\"')

    def test_update(self):
        outer = Outer()
        outer.update({'c': 7, 'inner.b': 8, 'inner.inner.a': 9})
        self.assertEqual(outer, Outer(c=7, inner=Inner(b=8, inner=InnerInner(a=9))))

    def test_update_nonexistent(self):
        outer = Outer()
        with self.assertRaises((AttributeError, KeyError, ParseError)):
            outer.update({'aaaaaaaaaaa.bbbbbbbb': 5})
        with self.assertRaises((AttributeError, KeyError, ParseError)):
            outer.update({'aaaaaaaaaaa': 5})
        with self.assertRaises((AttributeError, KeyError, ParseError)):
            outer.update({'inner.aaaaaaaaaa': 5})


class TestDataclassApplyArgv(TestCase):
    def test_basic(self):
        outer = Outer()
        outer.apply_argv(['c=8', 'inner.inner.a=9'])
        self.assertEqual(outer, Outer(c=8, inner=Inner(b=2, inner=InnerInner(a=9))))

        # Test that value must match the existing type and that nothing changes
        # in case of an error.
        outer = Outer()
        with self.assertRaises(ParseError):
            outer.apply_argv(['c=10', 'inner.b=not_an_integer'])
        self.assertEqual(outer, Outer())

    def test_supported_types(self):
        """Test all dataclass member types for which argv parsing is supported."""
        @dataclass
        class Klass(DataclassMixin):
            i: int = 0
            f: float = 0.0
            s: str = ""
            tif: Tuple[int, float] = (0, 0.0)
            li: List[int] = field(lambda: [0])

        k = Klass()
        k.apply_argv(['i=1', 'f=2.0', 's=abc', 'tif=3,4.5', 'li=5,6,7'])
        self.assertEqual(k, Klass(i=1, f=2.0, s="abc", tif=(3, 4.5), li=[5, 6, 7]))

    def test_list_item(self):
        @dataclass
        class Klass(DataclassMixin):
            li: List[int] = field(lambda: [0])

        k = Klass()
        k.apply_argv(['li.0=5'])
        self.assertEqual(k, Klass(li=[5]))

    def test_dict_item(self):
        @dataclass
        class Klass(DataclassMixin):
            d: Dict[str, Any] = field(lambda: {'d1': [10, 20, 30], 'd2': 5})

        k = Klass()
        k.apply_argv(['d.d1=5,6,7'])
        self.assertEqual(k, Klass(d={'d1': [5, 6, 7], 'd2': 5}))

    def test_json(self):
        i = Inner()
        i.apply_argv(['b:json=[10,20]', 'inner.a:json={"a":5}'])
        self.assertEqual(i, Inner(b=[10, 20], inner=InnerInner(a={'a': 5})))

        with self.assertRaises(ValueError):
            i.apply_argv(['b:i_am_not_defined=5'])

    def test_special_delete(self):
        @dataclass
        class Klass(DataclassMixin):
            l: List[int] = field(lambda: [10, 20, 30])
            d: Dict[str, Any] = field(lambda: {'d1': [10, 20, 30], 'd2': 5})

        k = Klass()
        k.apply_argv(['l.1:delete=', 'd.d1.-1:delete=', 'd.d2:delete='])
        self.assertEqual(k, Klass(l=[10, 30], d={'d1': [10, 20]}))

    def test_special_extend(self):
        @dataclass
        class Klass(DataclassMixin):
            l: List[Any] = field(lambda: [10, 'abc'])

        # Test one modification.
        k = Klass()
        k.apply_argv(['l:extend:json=[{}, 5.0]'])
        self.assertEqual(k, Klass(l=[10, 'abc', {}, 5.0]))

        # The test below fails.
        # # Test two modifications.
        # k = Klass()
        # k.apply_argv(['l:extend:json=["def"]', 'l:extend:json=[{}, 5.0]'])
        # self.assertEqual(k, Klass(l=[10, 'abc', 'def', {}, 5.0]))
