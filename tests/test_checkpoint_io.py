import errno
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from enntity_drafter.checkpoint_io import WrittenBack


class DiskHeadroomTests(unittest.TestCase):
    def test_refuses_before_opening_when_headroom_is_unavailable(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'rank0.pt.tmp'
            with patch('enntity_drafter.checkpoint_io.shutil.disk_usage', return_value=SimpleNamespace(free=99)):
                with self.assertRaises(OSError) as caught:
                    WrittenBack(path, min_free_bytes=100)
            self.assertEqual(caught.exception.errno, errno.ENOSPC)
            self.assertFalse(path.exists())

    def test_rechecks_each_write_when_another_writer_consumes_space(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'rank0.pt.tmp'
            with patch('enntity_drafter.checkpoint_io.shutil.disk_usage', side_effect=[
                SimpleNamespace(free=200), SimpleNamespace(free=200), SimpleNamespace(free=102)
            ]):
                stream = WrittenBack(path, min_free_bytes=100)
                self.assertEqual(stream.write(b'first'), 5)
                with self.assertRaisesRegex(OSError, 'disk headroom'):
                    stream.write(b'next')
                stream.close()
            self.assertEqual(path.read_bytes(), b'first')

    def test_write_must_leave_the_reserve_even_when_it_starts_above_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch('enntity_drafter.checkpoint_io.shutil.disk_usage', return_value=SimpleNamespace(free=105)):
                stream = WrittenBack(Path(tmp) / 'rank0.pt.tmp', min_free_bytes=100)
                with self.assertRaises(OSError):
                    stream.write(b'123456')
                stream.close()

    def test_negative_reserve_is_invalid(self):
        with self.assertRaises(ValueError):
            WrittenBack(Path('/unused'), min_free_bytes=-1)


class CheckpointIntegrationTests(unittest.TestCase):
    def test_low_space_keeps_last_good_checkpoint_and_removes_partial_file(self):
        import torch
        from enntity_drafter.train import save_checkpoint
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            model = torch.nn.Linear(8, 8)
            optimizer = torch.optim.AdamW(model.parameters())
            model(torch.ones(1, 8)).sum().backward()
            optimizer.step()
            old = save_checkpoint(output, 1, model, optimizer, 0, 1, {}, min_disk_free_gb=0)
            previous = (old / 'rank0.pt').read_bytes()
            calls = 0
            def space(_):
                nonlocal calls
                calls += 1
                return SimpleNamespace(free=10**12 if calls < 8 else 0)
            with patch('enntity_drafter.checkpoint_io.shutil.disk_usage', side_effect=space):
                with self.assertRaises((OSError, RuntimeError)):
                    save_checkpoint(output, 2, model, optimizer, 0, 1, {}, keep=1)
            self.assertEqual((output / 'checkpoints/latest-rank0').read_text(), old.name)
            self.assertEqual((old / 'rank0.pt').read_bytes(), previous)
            self.assertFalse((output / 'checkpoints/step-0000002/rank0.pt.tmp').exists())
            self.assertFalse((output / 'checkpoints/step-0000002/rank0.pt').exists())

    def test_successful_save_retains_the_requested_number(self):
        import torch
        from enntity_drafter.train import save_checkpoint
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            model = torch.nn.Linear(2, 2)
            optimizer = torch.optim.AdamW(model.parameters())
            for step in range(1, 4):
                save_checkpoint(output, step, model, optimizer, 0, 1, {}, keep=2, min_disk_free_gb=0)
            self.assertEqual(sorted(p.name for p in (output / 'checkpoints').glob('step-*')),
                             ['step-0000002', 'step-0000003'])
