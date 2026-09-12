"""Multi-turn trajectory task grouping and conversation building (M2).

Task-row convention (eval/multiturn_tasks.jsonl):
    rows carrying the same `traj_id` form ONE trajectory, ordered by
    `turn_index` (file order breaks ties, stable sort); rows without
    `traj_id` are single-turn trajectories — exactly the M1 behavior.

Conversation convention:
    turn k executes with messages [user_1, assistant_1, ..., user_k];
    a prior turn whose execution failed contributes its user message only.
    Assistant history is capped at HISTORY_PREVIEW_CHARS per turn to
    control context cost; the current user message is never truncated.
    A single-turn trajectory therefore produces the legacy
    [{"role": "user", ...}] list byte-for-byte.
"""
from __future__ import annotations

HISTORY_PREVIEW_CHARS = 2000


def group_trajectories(tasks: list[dict]) -> list[list[dict]]:
    """Group task rows into trajectories, preserving file order of groups."""
    out: list[list[dict]] = []
    by_traj_id: dict[str, list[dict]] = {}
    for t in tasks:
        tid = t.get("traj_id")
        if tid is None:
            out.append([t])
        else:
            group = by_traj_id.get(tid)
            if group is None:
                group = []
                by_traj_id[tid] = group
                out.append(group)
            group.append(t)
    for group in out:
        if group[0].get("traj_id") is not None:
            group.sort(key=lambda t: t.get("turn_index", 0))
    return out


def build_conversation(steps: list, current_message: str) -> list[dict]:
    """[user_1, assistant_1, ..., user_k] from completed prior steps."""
    msgs: list[dict] = []
    for s in steps:
        msgs.append({"role": "user", "content": s.message})
        if s.exec_result is not None:
            content = s.exec_result.content
            if len(content) > HISTORY_PREVIEW_CHARS:
                content = content[:HISTORY_PREVIEW_CHARS]
            msgs.append({"role": "assistant", "content": content})
    msgs.append({"role": "user", "content": current_message})
    return msgs
