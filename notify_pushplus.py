"""Durable PushPlus terminal-event delivery using the user's existing token source."""

import ast
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import urllib.request


TOKEN_SOURCE = Path("C:/ZemaxAuto/dji_zemax_checker/src/notify_pushplus.py")


def token():
    value = os.environ.get("PUSHPLUS_TOKEN")
    if value:
        return value
    tree = ast.parse(TOKEN_SOURCE.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "PUSHPLUS_TOKEN"
            for target in node.targets
        ):
            value = ast.literal_eval(node.value)
            if isinstance(value, str) and value:
                return value
    raise RuntimeError("PushPlus token unavailable")


def _deliver_terminal_unlocked(directory, state):
    """Retry an unacknowledged terminal event. Successful deliveries are deduplicated."""
    directory = Path(directory)
    event_id = state["event_id"]
    receipt = directory / f"notification_{event_id}.json"
    previous = json.loads(receipt.read_text()) if receipt.exists() else {}
    if previous.get("accepted"):
        return previous
    result = {
        "event_id": event_id, "accepted": False,
        "attempts": previous.get("attempts", 0) + 1,
        "utc": datetime.now(timezone.utc).isoformat(),
    }
    try:
        content = (
            f'Zemax 后台任务 {directory.name}\n状态：{state["status"]}\n'
            f'已验收 {state.get("completed", 0)}/{state.get("total", 0)} {state.get("unit_label", "组")}\n'
            f'原因：{state.get("reason", "")}\n事件：{event_id}'
        )
        payload = {"token": token(), "title": state.get("notification_title", f'Zemax 任务{state["status"]}'),
                   "content": content, "template": "txt"}
        request = urllib.request.Request(
            "https://www.pushplus.plus/send",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json; charset=utf-8"}, method="POST",
        )
        with urllib.request.urlopen(request, timeout=15) as response:
            body = json.loads(response.read().decode("utf-8"))
        result.update({"code": body.get("code"), "message": body.get("msg"),
                       "accepted": body.get("code") == 200})
    except Exception as error:
        result["error"] = str(error)
    temporary = receipt.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temporary.replace(receipt)
    return result


def deliver_terminal(directory, state):
    """Serialize watchdog and manual delivery for the same terminal event."""
    import msvcrt
    lock_path = Path(directory)/f'notification_{state["event_id"]}.lock'
    with lock_path.open("a+b") as lock:
        lock.seek(0)
        if not lock.read(1):
            lock.write(b"0")
            lock.flush()
        lock.seek(0)
        try:
            msvcrt.locking(lock.fileno(), msvcrt.LK_LOCK, 1)
        except OSError:
            return {"event_id":state["event_id"], "accepted":False,
                    "error":"another notification delivery is in progress"}
        try:
            return _deliver_terminal_unlocked(directory, state)
        finally:
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
