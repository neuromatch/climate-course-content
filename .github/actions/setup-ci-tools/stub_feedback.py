# Stub vibecheck and datatops for headless/CI execution.
# Notebooks install and import these packages for student feedback widgets.
# In CI we replace them with no-ops so notebook execution is not blocked.
#
# Installed into ~/.ipython/profile_default/startup/ by the setup-ci-tools action.
import sys
import types


class _NoOpContainer:
    def __init__(self, *args, **kwargs):
        pass

    def render(self):
        pass


class _VibecheckStub(types.ModuleType):
    DatatopsContentReviewContainer = _NoOpContainer

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return _NoOpContainer


class _DatatopsStub(types.ModuleType):
    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return _NoOpContainer


if "vibecheck" not in sys.modules:
    sys.modules["vibecheck"] = _VibecheckStub("vibecheck")
    print("vibecheck stubbed for headless CI execution")

if "datatops" not in sys.modules:
    sys.modules["datatops"] = _DatatopsStub("datatops")
    print("datatops stubbed for headless CI execution")
