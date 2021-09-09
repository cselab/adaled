"""Dataclass-related utils."""

import numpy as np

from collections.abc import Collection, Mapping
from typing import Any, Dict, Optional, Sequence, Union
import dataclasses
import json
import re

class ParseError(ValueError):
    pass


class NoValue:
    pass


class SpecifiedLater:
    """A non-`None` value used to mark that a dataclass field value must be
    provided either by the user or by one of the outer dataclasses in a nested
    dataclass scenario.
    """
    def __repr__(self):
        return "SPECIFIED_LATER"

    def __hash__(self):
        return hash(repr(self))

    def __eq__(self, other):
        if isinstance(other, SpecifiedLater):
            return True
        else:
            return NotImplemented

    def __ne__(self, other):
        if isinstance(other, SpecifiedLater):
            return False
        else:
            return NotImplemented

SPECIFIED_LATER = SpecifiedLater()


class PrettyPrintJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if callable(o):
            return str(o)
        elif o is SPECIFIED_LATER or isinstance(o, SpecifiedLater):
            return str(o)
        elif isinstance(o, slice):
            return repr(o)
        elif isinstance(o, DataclassMixin):
            return dataclasses.asdict(o)
        elif hasattr(o, 'asdict'):
            return o.asdict()
        elif isinstance(o, np.ndarray):
            return o.tolist()
        elif isinstance(o, np.generic):
            return repr(o)
        else:
            return super().default(o)


def _parse_like(arg: str, like: Any):
    if isinstance(like, int):
        return int(arg)
    elif isinstance(like, float):
        return float(arg)
    elif isinstance(like, bool):
        if arg in ['true', 'True']:
            return True
        if arg in ['false', 'False']:
            return False
        arg = int(arg)
        if not (0 <= arg <= 1):
            raise ValueError(f"expected 0 or 1, got {arg}")
        return bool(arg)
    elif isinstance(like, str):
        return arg
    elif isinstance(like, tuple):
        args = arg.split(',')
        if len(args) != len(like):
            raise ParseError(f"Expected {len(like)} comma-separated elements "
                             f"like {like!r}, got {arg!r}. If the number "
                             f"of elements can vary, use a list instead.")
        return tuple(_parse_like(a, l) for a, l in zip(args, like))
    elif isinstance(like, list):
        assert len(like) > 0, "cannot determine type from an empty list"
        assert all(el.__class__ == like[0].__class__ for el in like), \
               "list must have uniform element types for now"
        like = like[0]
        return [_parse_like(el, like) for el in arg.split(',')]
    else:
        raise Exception(f"type {like.__class__.__name__} not (yet) supported")


class DataclassMixin:
    """Auxiliary extensions of the built-in dataclass type."""
    __slots__ = ()

    @classmethod
    def filter_field_names(cls, *patterns):
        """Return a list of fields whose names match any of the given patterns."""
        patterns = [re.compile(pattern) for pattern in patterns]
        fields = [
            field.name
            for field in dataclasses.fields(cls)
            if any(pattern.match(field.name) for pattern in patterns)
        ]
        return fields

    @staticmethod
    def format_argv(key: str, value: Any):
        """Reverse of `parse_argv`. Generate an equivalent `key[:fmt]=value`
        command line argument for the given key and value."""
        # Note: in principle we should always generate JSON, to ensure that
        # types are indeed changed if needed.
        fmt = ''
        if isinstance(value, (list, dict)):
            fmt = ':json'
            value = json.dumps(value, separators=(',', ':'))
        elif isinstance(value, str):
            pass
        else:
            value = repr(value)
        return f'{key}{fmt}={value}'

    def parse_argv(self, argv: Sequence[str]):
        """Parse command line arguments of the format `some.key[:special]=value`,
        where `:special` is an optional part denoting special format or meaning
        of `value`.

        Supported formats:
            unspecified or empty: `value` is parsed as the type of the
                                  current value of the corresponding attribute
            'delete': delete the element `key`, `value` must be empty
            'json': `value` is parsed as a JSON
            'extend:json': `value` is parsed as a JSON list and extend to the
                    existing value. The current value and given value must be
                    either both strings or both lists or both dictionaries. (*)

        (*) Note: operations currently cannot be stacked!

        Returns a dictionary `{'some.key': <parsed value>, ...}` compatible
        with `update()`.
        """
        # TODO: To support stacked operations, we need to take a deep copy of
        # `self`, operate on it and then either accept the replacement or
        # reject it.

        def _parse(arg):
            if '=' not in arg:
                raise ValueError(f"expected 'some.key=value', got {arg!r}")
            key, value = arg.split('=', maxsplit=1)
            if ':' in key:
                key, fmt = key.split(':', maxsplit=1)
            else:
                fmt = ''
            parts = key.split('.')
            current = self
            for part in parts[:-1]:
                current = self._getattr(current, part)

            if fmt == 'delete':
                if value:
                    raise ParseError(f"expected empty value, got {value!r}")
                try:
                    del current[parts[-1]]
                except TypeError:
                    del current[int(parts[-1])]
                value = NoValue
            elif fmt == 'extend:json':
                current = self._getattr(current, parts[-1])
                value = json.loads(value)
                if isinstance(value, (str, list, dict)):
                    if not isinstance(current, value.__class__):
                        raise TypeError(f"cannot extend {current!r} with a value {value!r}")
                else:
                    raise TypeError(f"extend does not support value {value!r}")
                value = current + value
            elif fmt == 'json':
                value = json.loads(value)
            elif fmt == '':
                like = self._getattr(current, parts[-1])
                value = _parse_like(value, like)
            else:
                raise ValueError(f"unrecognized format {fmt!r}")

            return key, value

        out = {}
        for arg in argv:
            try:
                key, value = _parse(arg)
            except ParseError:
                raise
            except Exception as e:
                raise ParseError(f"error while parsing {arg!r} for {self.__class__}") from e
            if value is not NoValue:
                out[key] = value
        return out

    def apply_argv(self, argv: Sequence[str]):
        """Parse command line arguments and update the dataclass."""
        kwargs = self.parse_argv(argv)
        self.update(kwargs)

    def update(self, dict: Dict[str, Any]):
        """Recursively update the dataclass values.

        Arguments:
            dict: dictionary of fields names and their new values

        Nested fields can be updated by passing dot-separated names.
        Updating a field that does not exist is an error.

        Example:
            >>> @dataclass
                class Inner:
                    a: int = 5
                @dataclass
                class Outer(DataclassMixin):
                    b: int = 5
                    inner: Inner = dataclasses.field(default_factory=Inner)
                outer = Outer()
                outer.update({'b': 7, 'inner.a': 9})
                outer
            Outer(b=7, inner=Inner(a=9))
        """
        for key, value in dict.items():
            current = self
            parts = key.split('.')
            try:
                for part in parts[:-1]:
                    current = self._getattr(current, part)
                self._setattr(current, parts[-1], value)
            except Exception as e:
                raise ParseError(f"exception while updating key {key!r} to a value "
                                   f"of type {value.__class__} for {self.__class__}")

    def _getattr(self, obj, name: str):
        if isinstance(obj, Mapping):
            return obj[name]
        elif isinstance(obj, Collection):
            return obj[int(name)]
        else:
            return getattr(obj, name)

    def _setattr(self, obj: Any, name: str, value: Any):
        if isinstance(obj, Mapping):
            obj[name] = value
        elif isinstance(obj, Collection):
            obj[int(name)] = value
        else:
            if not hasattr(obj, name):
                raise AttributeError(
                        f"{obj.__class__.__name__} has no attribute {name}")
            setattr(obj, name, value)

    def pretty_print(self, flush=False):
        """Pretty-print the config. Currently prints in the JSON format."""
        print(json.dumps(dataclasses.asdict(self), indent=4,
                         cls=PrettyPrintJSONEncoder), flush=flush)

    def assert_equal_length(self, *field_patterns):
        """Assert that given list-like fields have same length.

        Arguments:
            regex patterns of field names
        """
        fields = self.filter_field_names(*field_patterns)
        sizes = [len(getattr(self, field)) for field in fields]
        if any(size != sizes[0] for size in sizes):
            msg = "\n".join(f"    {field} = {getattr(self, field)}"
                            for field, size in zip(fields, sizes))
            raise ValueError(f"The following fields of a dataclass "
                             f"`{self.__class__.__name__}` are expected "
                             f"to have equal length:\n{msg}")

    def validate(self, prefix: Optional[str] = None):
        if prefix is None:
            prefix = self.__class__.__name__
        def rec(key, item):
            if item is SPECIFIED_LATER:
                raise TypeError(f"`{prefix}.{key}` not specified")
            if isinstance(item, DataclassMixin):
                item.validate(f'{prefix}.{key}')
            elif isinstance(item, (list, tuple)):
                for i, child in enumerate(item):
                    rec(f'{key}.{i}', child)
            elif isinstance(item, dict):
                for k, v in item.items():
                    rec(f'{key}.{k}', v)

        for key, item in self.__dict__.items():
            rec(key, item)


dataclass = dataclasses.dataclass

def field(default_factory=None, **kwargs):
    """Wrapper around `dataclasses.field` that accepts `default_factory` as the
    (first and only) positional argument."""
    if default_factory is not None:
        kwargs['default_factory'] = default_factory
    return dataclasses.field(**kwargs)
