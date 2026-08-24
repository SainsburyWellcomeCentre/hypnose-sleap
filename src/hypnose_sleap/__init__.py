"""SLEAP pose tracking for hypnose experiments.

Pipeline verbs, in order: ``fetch``, ``infer``, ``extract``, ``combine``, ``push``,
``annotate``. ``run`` does ``extract`` + ``combine`` in one process.

Nothing is imported eagerly -- `infer` needs sleap and torch, `combine` and `annotate`
need hypnose_behavior, and a session that only runs `extract` should not pay for them.
"""

__version__ = "0.1.0"
