import contextlib

__all__ = ['Tracer', 'default_tracer', 'get_tracer']

# TODO: Rename to Hook? Torch already has something like that.
class Tracer:
    def __enter__(self):
        _tracer_stack.append(self)
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        old = _tracer_stack.pop()
        assert old is self

    def evaluate(self, model, input_):
        return model(input_)

    def __call__(self, key):
        return _nullcontext


default_tracer = Tracer()
_tracer_stack = [default_tracer]
_nullcontext = contextlib.nullcontext()

def get_tracer():
    return _tracer_stack[-1]
