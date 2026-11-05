"""Read or append the one shared team chat, across all Git worktrees.

python scripts/team_chat.py read
python scripts/team_chat.py post --sender codex --to gemini --message "Please review the audit"
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

AGENTS = ("codex", "claude", "gemini")
HEADER = "# Team chat — Noesis\n\nShared append-only conversation. Read before work and before a full simulation.\nReply using the message ID. Never remove or rewrite another message.\n\n"


def shared_root():
    checkout = Path(__file__).resolve().parents[1]
    common = subprocess.check_output([
        "git", "-C", str(checkout), "rev-parse", "--path-format=absolute", "--git-common-dir",
    ], encoding="utf-8").strip()
    return Path(common).resolve().parent


@contextmanager
def locked(root, timeout=10.):
    # Persistent sentinel; OS locks are released even if a writer crashes.
    with (root / ".team-chat.lock").open("a+b") as handle:
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()
        deadline = time.monotonic() + timeout
        while True:
            handle.seek(0)
            try:
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                if time.monotonic() >= deadline:
                    raise TimeoutError("Team chat is busy; retry without editing it directly") from None
                time.sleep(.05)
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)


def post(root, sender, recipient, kind, message, reply_to=None):
    if not message.strip():
        raise ValueError("Message cannot be empty")
    identifier = uuid.uuid4().hex[:12]
    stamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
    body = "\n".join("> " + line for line in message.strip().splitlines())
    reply = f"\nReply to: `{reply_to}`\n" if reply_to else ""
    entry = f"## {stamp} | {sender} → {recipient} | {kind} | {identifier}\n{reply}\n{body}\n\n"
    with locked(root):
        path = root / "TEAM_CHAT.md"
        with path.open("a", encoding="utf-8", newline="\n") as handle:
            if path.stat().st_size == 0:
                handle.write(HEADER)
            handle.write(entry)
            handle.flush()
            os.fsync(handle.fileno())
    return identifier


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="action", required=True)
    read = sub.add_parser("read")
    read.add_argument("--tail", type=int, default=120)
    snapshot = sub.add_parser("snapshot")
    snapshot.add_argument("--output", type=Path, required=True)
    send = sub.add_parser("post")
    send.add_argument("--sender", choices=AGENTS, required=True)
    send.add_argument("--to", choices=(*AGENTS, "all"), default="all")
    send.add_argument("--kind", choices=("status", "question", "answer", "result"), default="status")
    send.add_argument("--reply-to")
    text = send.add_mutually_exclusive_group(required=True)
    text.add_argument("--message")
    text.add_argument("--message-file", type=Path)
    args = ap.parse_args()
    root = shared_root()
    if args.action == "snapshot":
        if args.output.resolve() == (root / "TEAM_CHAT.md").resolve():
            ap.error("A snapshot cannot replace the live log")
        with locked(root):
            content = (root / "TEAM_CHAT.md").read_text(encoding="utf-8")
        args.output.write_text(content, encoding="utf-8")
        print(str(args.output.resolve()))
    elif args.action == "read":
        if args.tail <= 0:
            ap.error("--tail must be positive")
        with locked(root):
            path = root / "TEAM_CHAT.md"
            content = path.read_text(encoding="utf-8") if path.exists() else HEADER
        print(str(path))
        print("\n".join(content.splitlines()[-args.tail:]))
    else:
        if args.reply_to and (len(args.reply_to) != 12 or any(c not in "0123456789abcdef" for c in args.reply_to)):
            ap.error("--reply-to must be a 12-character message ID")
        message = args.message_file.read_text(encoding="utf-8-sig") if args.message_file else args.message
        print(post(root, args.sender, args.to, args.kind, message, args.reply_to))


if __name__ == "__main__":
    main()
