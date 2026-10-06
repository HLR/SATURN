"""In-process cache of freshly built scenes, keyed by what determines them.

Several questions can share one image set; perception (SAM3, Orient-Anything,
VGGT, fusion) is identical for all of them, and each question then mutates its
own Scene (planner context, detect(), constraints). So the cache keeps the
PRE-MUTATION scene as a serialisable dict and hands every question a fresh
`Scene.from_dict`.
A per-key lock makes concurrent questions on one image set wait for the first
build instead of repeating it.
"""
from __future__ import annotations
import hashlib
import json
import threading
from collections import OrderedDict
from typing import Any, Callable, Hashable, Tuple


def scene_key(image_paths, keywords, bboxes, **flags) -> Tuple[Hashable, ...]:
    """Everything the built scene depends on. Image identity is the path; two
    questions with the same images but different keywords/boxes never share."""
    kw = tuple(sorted(keywords)) if isinstance(keywords, (list, tuple, set)) else (
        tuple(sorted((k, tuple(v)) for k, v in keywords.items())) if isinstance(keywords, dict) else keywords)
    bb = hashlib.sha1(json.dumps(bboxes, sort_keys=True, default=str).encode()).hexdigest() if bboxes is not None else None
    return (tuple(image_paths), kw, bb, tuple(sorted(flags.items())))


class SceneCache:
    def __init__(self, max_entries: int = 8):
        self.max_entries = max_entries
        self._entries: "OrderedDict[Hashable, Any]" = OrderedDict()
        self._locks: dict = {}
        self._mu = threading.Lock()
        self.hits = 0
        self.builds = 0

    def _lock_for(self, key):
        with self._mu:
            return self._locks.setdefault(key, threading.Lock())

    def get_or_build(self, key: Hashable, build: Callable[[], Any],
                     cacheable: Callable[[Any], bool] = lambda v: True) -> Any:
        """Return the cached value for ``key`` or build it. A value for which
        ``cacheable(value)`` is False goes back to its builder only and is NOT
        stored, so every later call builds its own."""
        with self._lock_for(key):           # one build per key, others wait then hit
            with self._mu:
                if key in self._entries:
                    self._entries.move_to_end(key)
                    self.hits += 1
                    return self._entries[key]
            value = build()
            if not cacheable(value):
                return value
            with self._mu:
                self._entries[key] = value
                self.builds += 1
                while len(self._entries) > self.max_entries:
                    old, _ = self._entries.popitem(last=False)
                    self._locks.pop(old, None)
            return value
