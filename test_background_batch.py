"""Targeted regressions for notification, resume validation and timeout cleanup."""

import json
import multiprocessing
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, Mock

import background_batch as batch
import notify_pushplus as push


def write_state_repeatedly(path):
    for number in range(100):
        batch.save(Path(path), {"sequence":number,"payload":"x"*10000})


class BackgroundBatchTests(unittest.TestCase):
    def test_atomic_state_read_during_another_process_writes(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"state.json"
            batch.save(path,{"sequence":-1,"payload":"x"*10000})
            writer = multiprocessing.get_context("spawn").Process(
                target=write_state_repeatedly,args=(str(path),))
            writer.start()
            for _ in range(300):
                state = batch.read(path)
                self.assertEqual(len(state["payload"]),10000)
            writer.join(timeout=15)
            if writer.is_alive():
                writer.terminate()
                writer.join()
                self.fail("Concurrent writer timed out")
            self.assertEqual(writer.exitcode,0)
            self.assertEqual(batch.read(path)["sequence"],99)

    def test_startup_admission_is_serial_and_stops_on_low_memory(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            state = {"jobs":{}}
            for key in ("first","second"):
                (directory/key).mkdir()
                state["jobs"][key]={"attempt":key}
            keys = ["first","second"]
            self.assertIsNone(batch.admit_start(directory,keys,state,2*1024**3,None))
            self.assertTrue((directory/"first/START_PERMIT").exists())
            self.assertFalse((directory/"second/START_PERMIT").exists())
            batch.admit_start(directory,keys,state,2*1024**3,None)
            self.assertFalse((directory/"second/START_PERMIT").exists())
            batch.save(directory/"first/model_status.json",{"phase":"ready"})
            reason = batch.admit_start(directory,keys,state,512*1024**2,None)
            self.assertEqual(reason,"startup_admission_requires_1_GiB_available")
            self.assertFalse((directory/"second/START_PERMIT").exists())
            batch.admit_start(directory,keys,state,2*1024**3,None)
            self.assertTrue((directory/"second/START_PERMIT").exists())

    def test_stopped_admission_grants_no_permit(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            (directory/"first").mkdir()
            state = {"jobs":{"first":{"attempt":"first"}}}
            batch.admit_start(directory,["first"],state,2*1024**3,"stopped")
            self.assertFalse((directory/"first/START_PERMIT").exists())
    def test_state_replace_retries_transient_windows_reader_lock(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"state.json"
            original = Path.replace
            calls = []
            def replace(source, target):
                calls.append(1)
                if len(calls) == 1:
                    raise PermissionError("transient reader lock")
                return original(source, target)
            with patch.object(Path,"replace",replace):
                batch.save(path,{"status":"running"})
            self.assertEqual(batch.read(path)["status"],"running")
            self.assertEqual(len(calls),2)
    def test_interrupted_group_gets_a_new_attempt_without_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            info = {"case_id":3,"rule":1,"status":"interrupted"}
            plan = {"cases":[{"id":3}],"installation":"example","group_timeout_seconds":1800}
            first = batch.prepare_attempt(directory, "case_003_rule_1", info, plan)
            (first/"evidence.txt").write_text("keep")
            second = batch.prepare_attempt(directory, "case_003_rule_1", info, plan)
            self.assertEqual(first.name,"attempt_001")
            self.assertEqual(second.name,"attempt_002")
            self.assertEqual((first/"evidence.txt").read_text(),"keep")
            self.assertEqual(batch.read(second/"job.json")["case_id"],3)

    def test_completed_batch_is_revalidated_without_launching(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            batch.save(directory/"state.json", {
                "status":"completed","jobs":{"job":{
                    "case_id":3,"rule":1,"attempt":"attempt",
                    "acceptance":{"result_sha256":"result","lens_sha256":"lens"},
                }},
            })
            with patch.object(batch,"validate_group",return_value={
                "result_sha256":"result","lens_sha256":"lens",
            }), patch.object(batch,"detached") as launch:
                batch.launch(directory)
                launch.assert_not_called()
    def test_pushplus_success_is_deduplicated(self):
        with tempfile.TemporaryDirectory() as folder:
            response = Mock()
            response.read.return_value = b'{"code":200,"msg":"ok"}'
            context = Mock()
            context.__enter__ = Mock(return_value=response)
            context.__exit__ = Mock(return_value=False)
            state = {"status":"completed","event_id":"event1","completed":1,"total":1}
            with patch.object(push, "token", return_value="test-token"), patch.object(
                push.urllib.request, "urlopen", return_value=context
            ) as send:
                self.assertTrue(push.deliver_terminal(folder, state)["accepted"])
                self.assertTrue(push.deliver_terminal(folder, state)["accepted"])
                self.assertEqual(send.call_count, 1)
            self.assertNotIn("test-token", next(Path(folder).glob("notification_*.json")).read_text())

    def test_api_rejection_is_not_marked_delivered(self):
        with tempfile.TemporaryDirectory() as folder:
            response = Mock()
            response.read.return_value = b'{"code":500,"msg":"failed"}'
            context = Mock()
            context.__enter__ = Mock(return_value=response)
            context.__exit__ = Mock(return_value=False)
            with patch.object(push, "token", return_value="test-token"), patch.object(
                push.urllib.request, "urlopen", return_value=context
            ):
                result = push.deliver_terminal(folder, {"status":"stopped","event_id":"event2"})
                self.assertFalse(result["accepted"])
                self.assertEqual(result["code"], 500)

    def test_result_identity_and_budget_checked_before_resume(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            (directory/"experiment").mkdir()
            result = {
                "case_id":1,"rule":2,"steps":[{}],"cost":{"requested_dls_cycles":50},
                "audit":{"case_identity_verified":True},"final_accepted":None,
                "stop_reasons":{"aperture":"initial_rejected","field":"initial_rejected"},
                "target_reached":False,
            }
            batch.save(directory/"experiment/result.json", result)
            self.assertIsNotNone(batch.validate_group(directory, 1, 2))
            with self.assertRaises(ValueError):
                batch.validate_group(directory, 2, 2)
            result["cost"]["requested_dls_cycles"] = 5
            batch.save(directory/"experiment/result.json", result)
            with self.assertRaises(ValueError):
                batch.validate_group(directory, 1, 2)

    def test_pid_reuse_cannot_kill_unrelated_process(self):
        identity = {"pid":os.getpid(),"created":0}
        with patch.object(batch.subprocess, "run") as kill:
            batch.terminate(identity)
            kill.assert_not_called()

    def test_timeout_terminates_owned_child(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            job = {"run_directory":folder,"group_timeout_seconds":10}
            batch.save(directory/"job.json", job)
            process = Mock(pid=123456, returncode=None)
            process.poll.side_effect = lambda: process.returncode
            def kill(identity):
                self.assertEqual(identity["pid"], 123456)
                process.returncode = 1
            with patch.object(batch.subprocess, "Popen", return_value=process), patch.object(
                batch.psutil, "Process", return_value=Mock(create_time=lambda:1)
            ), patch.object(batch, "time", side_effect=[0,20,21]), patch.object(
                batch, "terminate", side_effect=kill
            ):
                result = batch.execute(str(directory/"job.json"))
                self.assertEqual(result["reason"], "group_timeout")
                self.assertEqual(result["returncode"], 1)

    def test_watchdog_reports_unexpected_daemon_exit(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            batch.save(directory/"daemon.json", {"pid":123456,"created":0})
            batch.save(directory/"state.json", {
                "status":"running","event_id":"crash","heartbeat":0,"completed":0,"total":1,
            })
            with patch.object(batch,"alive",return_value=False), patch.object(
                batch,"terminate"
            ), patch.object(push,"deliver_terminal",return_value={"accepted":True}) as send:
                batch.watchdog(directory)
                self.assertEqual(batch.read(directory/"state.json")["status"],"failed")
                send.assert_called_once()


if __name__ == "__main__":
    unittest.main()
