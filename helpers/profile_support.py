try:
    from line_profiler import LineProfiler

    profile = LineProfiler()
except ImportError:
    def profile(func):
        return func
