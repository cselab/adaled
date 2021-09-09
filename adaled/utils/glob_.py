from typing import Iterable, Union

def glob_ex(include: Union[str, Iterable[str]],
            exclude: Union[str, Iterable[str]] = []):
    """Evaluate file globs.

    If a pattern has no asterisk `*`, it is accepted as is, otherwise the
    pattern is passed to `glob.glob`.
    """
    if isinstance(include, str):
        include = (include,)
    if isinstance(exclude, str):
        exclude = (exclude,)

    from glob import glob

    def find(patterns):
        paths = []
        for pattern in patterns:
            if '*' in pattern:
                paths.extend(sorted(glob(pattern)))
            else:
                paths.append(pattern)
        return paths

    exclude = set(find(exclude))
    include = find(include)
    return [path for path in include if path not in exclude]
