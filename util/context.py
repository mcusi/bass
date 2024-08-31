class Context():
    """ Context
    This is a context manager used to set global variables that are referenced in model sampling. Specifically we use it to track:
    - The audio sampling rate
    - The reference db
    - The scene object
    """
    def __init__(self, **kwargs):
        self.stack = []
        self.warned = []

    def push(self, **kwargs):
        previous_values = {}
        for k, v in kwargs.items():
            if hasattr(self, k):
                previous_values[k] = getattr(self, k)
                if k not in self.warned:
                    print(f"Warning: overriding context.{k}")
                    self.warned.append(k)
            else:
                previous_values[k] = None
            setattr(self, k, v)
        self.stack.append(previous_values)

        return self

    def __enter__(self):
        pass

    def __exit__(self, type, value, traceback):
        previous_values = self.stack.pop()
        for k, v in previous_values.items():
            if v is None:
                delattr(self, k)
            else:
                setattr(self, k, v)

    def __call__(self, *args, **kwargs):
        push_dict = {k:v for d in [*args, kwargs] for k,v in d.items()}
        return self.push(**push_dict)

context = Context()