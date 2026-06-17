"""Lightweight run/decision audit (SQLite).

Optional: the per-decision logger and cost tracker are only engaged when a
``_decision_id`` is attached to the input object. The engine runs fine — and
writes nothing to disk — without one.
"""
