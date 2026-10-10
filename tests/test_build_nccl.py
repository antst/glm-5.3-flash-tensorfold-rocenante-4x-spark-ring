"""CPU-only builder guards; no Docker, SSH, CUDA, RDMA or compiler invocation."""
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tarfile
import tempfile
import types
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
ROOT=Path(__file__).resolve().parent.parent
SCRIPT=ROOT/'scripts/build-nccl.sh'
body=SCRIPT.read_text().split("<<'SPARK_RING_NCCL_PYTHON'\n",1)[1].rsplit('\nSPARK_RING_NCCL_PYTHON',1)[0]
builder=types.ModuleType('native_nccl_builder');builder.__file__=str(SCRIPT)
with patch.dict(os.environ,{'SPARK_RING_BUILDER_FILE':str(SCRIPT)}):
    exec(compile(body,str(SCRIPT),'exec'),builder.__dict__)

class BuildTests(unittest.TestCase):
    def test_current_fixed_source_and_patch(self):
        pins=builder.manifest()
        self.assertEqual(pins['commit'],'12df1a11afad322be5a204a2db890161cbf8131d')
        self.assertEqual(pins['nccl_get_version'],23203)
        self.assertEqual(pins['max_jobs'],2)
        self.assertFalse(pins['marker']['execution_permitted_by_builder'])

    def test_plan_does_not_run_process_or_create_paths(self):
        with patch.object(builder,'run',side_effect=AssertionError('unexpected process')), \
             patch.object(Path,'mkdir',side_effect=AssertionError('unexpected write')), \
             contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(builder.main(['--plan','--fetch','--build-dir','/tmp/new-native','--publish','/tmp/new-lib','--loader-image',json.loads((ROOT/'config.example.json').read_text())['image']]),0)
        plan=json.loads(output.getvalue())
        self.assertEqual(plan['loader_image'],json.loads((ROOT/'config.example.json').read_text())['image'])
        for field in ('writes','network','ssh','gpu'): self.assertFalse(plan[field])

    def test_more_than_two_jobs_refused(self):
        with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
            builder.main(['--plan','--jobs','3'])

    def test_wrong_loader_image_refused_before_docker(self):
        with patch.object(builder,'run',side_effect=AssertionError('unexpected process')),self.assertRaises(ValueError):
            builder.main(['--plan','--loader-image','unrelated:latest'])

    def test_wrong_or_dirty_source_refused(self):
        for answers in ([builder.COMMIT,'dirty'],['wrong','']):
            with patch.object(builder,'git',side_effect=answers),self.assertRaises(ValueError):
                builder.source_proof(Path('/not-written'))

    def test_git_source_queries_disable_optional_index_writes(self):
        with patch.object(builder,'run',return_value='value') as run:
            builder.git(Path('/source'),'status','--porcelain=v1')
        self.assertEqual(run.call_args.kwargs['env']['GIT_OPTIONAL_LOCKS'],'0')
        self.assertIn('safe.directory=/source',run.call_args.args[0])

    def test_existing_or_network_compile_paths_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory=Path(temporary)
            with self.assertRaises(ValueError): builder.validate_paths(directory,None)
            with patch.object(builder,'run',return_value='nfs4\n'),self.assertRaises(ValueError):
                builder.validate_paths(directory/'new',None)
            self.assertFalse((directory/'new').exists())

    def test_existing_publish_directory_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory=Path(temporary)
            with patch.object(builder,'run',return_value='ext4\n'),self.assertRaises(ValueError):
                builder.validate_paths(directory/'new',directory)

    def test_changed_archive_refused_without_extraction(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory=Path(temporary);archive=directory/'bad.tar';archive.write_bytes(b'bad')
            with self.assertRaises(ValueError): builder.extract(archive,directory/'source')
            self.assertFalse((directory/'source').exists())

    def test_unsafe_archive_paths_refused_before_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory=Path(temporary);archive=directory/'bad.tar'
            with tarfile.open(archive,'w') as tar:
                entry=tarfile.TarInfo('../escape');entry.size=1;tar.addfile(entry,io.BytesIO(b'x'))
            with patch.object(builder,'sha',return_value=builder.ARCHIVE_SHA),self.assertRaises(ValueError):
                builder.extract(archive,directory/'source')
            self.assertFalse((directory/'source').exists())

    def test_safe_source_license_symlink_preserves_git_target(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory=Path(temporary);archive=directory/'source.tar'
            with tarfile.open(archive,'w') as tar:
                file=tarfile.TarInfo('LICENSE.txt');file.size=1;tar.addfile(file,io.BytesIO(b'x'))
                link=tarfile.TarInfo('pkg/debian/copyright');link.type=tarfile.SYMTYPE;link.linkname='../../LICENSE.txt';tar.addfile(link)
            with patch.object(builder,'sha',return_value=builder.ARCHIVE_SHA): builder.extract(archive,directory/'source')
            self.assertEqual(os.readlink(directory/'source/pkg/debian/copyright'),'../../LICENSE.txt')
            self.assertEqual((directory/'source/pkg/debian/copyright').read_bytes(),b'x')

    def test_archive_symlink_escape_refused_before_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory=Path(temporary);archive=directory/'source.tar'
            with tarfile.open(archive,'w') as tar:
                link=tarfile.TarInfo('escape');link.type=tarfile.SYMTYPE;link.linkname='../outside';tar.addfile(link)
            with patch.object(builder,'sha',return_value=builder.ARCHIVE_SHA),self.assertRaises(ValueError):
                builder.extract(archive,directory/'source')
            self.assertFalse((directory/'source').exists())

    def test_mesh_publish_requires_explicit_marker_input(self):
        with patch.object(builder,'run',side_effect=AssertionError('unexpected process')),self.assertRaises(ValueError):
            builder.main(['--plan','--publish-mesh','/tmp/new-marker'])

if __name__=='__main__': unittest.main()
