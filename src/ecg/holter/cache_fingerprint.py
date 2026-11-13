"""Fingerprint of the code a cached result depends on.

A result cached next to a recording (ischemia_cache.json, ...) is valid only
for the code that made it, so its key holds a hash of that code and it is
recomputed when the code changes. A hand-written list of files drifts: the
ischemia / axis cache listed five files and missed the one that held the
mains filter (a UI module), so a changed filter left old QRS axis results in
the cache (code review, 2026-10).

code_fingerprint() takes the entry files and follows their imports itself --
statically (AST), lazy imports inside functions included -- through the `ecg`
package, so a file the result depends on cannot be left out, and a file it
does not import (the UI, the arrhythmia detection) does not invalidate it.
Package __init__ files on the way are hashed too (not followed).

Code is not the only input: the measurements read the clinical configuration
(signal-quality limit, QRS validator, mains notch ...), which is a YAML file,
not code -- clinical_config_fingerprint() hashes the values this process
loaded, for a cache key to hold next to the code hash.

'' means the code cannot be read (a frozen build has no .py files, or a file
does not parse): there is then no way to tell whether a saved result is
stale, and the caller must not use its disk cache (auto_segment_arrthymia_
detection._code_fingerprint does the same).
"""

import ast
import hashlib
import json
import os
from typing import Dict, Iterable, List, Optional, Set, Tuple

_ECG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # .../ecg
_SRC_DIR = os.path.dirname(_ECG_DIR)                                      # holds the `ecg` package
_memo: Dict[Tuple[str, ...], str] = {}


def _norm(path: str) -> str:
    return os.path.normcase(os.path.abspath(path))


def _in_ecg(path: str) -> bool:
    return _norm(path).startswith(_norm(_ECG_DIR) + os.sep)


def _module_files(root: str, parts: List[str]) -> List[str]:
    """The file(s) module `parts` under `root` stands for: parts.py and / or
    parts/__init__.py."""
    base = os.path.join(root, *parts) if parts else root
    out = []
    if parts and os.path.isfile(base + '.py'):
        out.append(base + '.py')
    init = os.path.join(base, '__init__.py')
    if os.path.isfile(init):
        out.append(init)
    return out


def _imports_of(path: str) -> Optional[Set[str]]:
    """Local (ecg package) files `path` imports, anywhere in the file; None
    when it cannot be read or parsed."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            tree = ast.parse(f.read(), filename=path)
    except (OSError, SyntaxError, ValueError):
        return None
    here = os.path.dirname(os.path.abspath(path))
    found: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.level:
                root = here
                for _ in range(node.level - 1):
                    root = os.path.dirname(root)
            else:
                if not (node.module or '').split('.')[0] == 'ecg':
                    continue
                root = _SRC_DIR
            parts = (node.module or '').split('.') if node.module else []
            found.update(_module_files(root, parts) if parts else [])
            for alias in node.names:          # `from pkg import submodule`
                found.update(_module_files(root, parts + [alias.name]))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split('.')[0] == 'ecg':
                    found.update(_module_files(_SRC_DIR, alias.name.split('.')))
    return {p for p in found if _in_ecg(p)}


def source_closure(entry_files: Iterable[str]) -> Optional[List[str]]:
    """Every file of the ecg package the entry files import, transitively,
    plus the entry files and the __init__ of every package on the way, sorted;
    None when any of them cannot be read."""
    todo = [os.path.abspath(p) for p in entry_files]
    seen: Dict[str, str] = {}
    while todo:
        p = todo.pop()
        if _norm(p) in seen:
            continue
        seen[_norm(p)] = p
        deps = _imports_of(p)
        if deps is None:
            return None
        todo.extend(d for d in deps if _norm(d) not in seen)
    for p in list(seen.values()):             # package __init__ files, hashed only
        d = os.path.dirname(os.path.abspath(p))
        while _in_ecg(os.path.join(d, 'x')) or _norm(d) == _norm(_ECG_DIR):
            init = os.path.join(d, '__init__.py')
            if os.path.isfile(init):
                seen.setdefault(_norm(init), init)
            if _norm(d) == _norm(_ECG_DIR):
                break
            d = os.path.dirname(d)
    return sorted(seen.values(), key=_norm)


def code_fingerprint(entry_files: Iterable[str]) -> str:
    """sha1 over the closure's relative paths and contents ('' when the code
    cannot be read -- see the module note). Computed once per process."""
    entries = tuple(sorted(_norm(p) for p in entry_files))
    if entries in _memo:
        return _memo[entries]
    files = source_closure(entries)
    digest = ''
    if files:
        h = hashlib.sha1()
        try:
            for p in files:
                h.update(os.path.relpath(p, _SRC_DIR).replace('\\', '/').encode('utf-8'))
                with open(p, 'rb') as f:
                    h.update(f.read())
            digest = h.hexdigest()
        except OSError:
            digest = ''
    _memo[entries] = digest
    return digest


_CONFIG_UNREADABLE = 'config-unreadable'


def clinical_config_fingerprint() -> str:
    """Hash of the clinical configuration this process is running with (the
    values load_clinical_config() returned -- the file, an environment
    override and the defaults all included; it is cached per process, so a
    file edited while the app runs counts from the next start, as it takes
    effect then). A fixed marker when it cannot be loaded: the code then runs
    on the HolterConfig defaults, the same every time."""
    try:
        from .clinical_config import load_clinical_config
        text = json.dumps(load_clinical_config().to_dict(), sort_keys=True, default=str)
    except Exception:
        return _CONFIG_UNREADABLE
    return hashlib.sha1(text.encode('utf-8')).hexdigest()
