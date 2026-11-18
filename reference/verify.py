"""Verify reference tables, source snapshots and optional release artifacts.

Uses only the Python standard library; no model fitting or data download.
"""
from pathlib import Path, PurePosixPath
import argparse
import hashlib
import json
import urllib.request
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group()
    choice.add_argument('--download', action='store_true', help='Download and verify the versioned partition/record archive')
    choice.add_argument('--archive', type=Path, help='Verify an already downloaded archive')
    args = parser.parse_args()
    manifest = json.loads((HERE / 'manifest.json').read_text())
    for name, item in manifest['tables'].items():
        if sha((HERE / 'tables' / name).read_bytes()) != item['sha256']:
            raise RuntimeError('Reference table checksum mismatch: ' + name)
    for name, digest in manifest['source_and_data_sha256'].items():
        if name.startswith('data/'):
            continue  # run_revision.py --check-only verifies the downloaded inputs.
        if sha((ROOT / 'revision' / name).read_bytes()) != digest:
            raise RuntimeError('Frozen source checksum mismatch: ' + name)
    print(f"Verified {len(manifest['tables'])} reference tables and frozen experiment sources.")
    if not args.download and args.archive is None:
        print('Use --download to check the archived partitions and run records as well.')
        return
    info = manifest['archive']
    archive = args.archive or ROOT / 'results/reference' / info['filename']
    if args.download and not archive.exists():
        archive.parent.mkdir(parents=True, exist_ok=True)
        temporary = archive.with_suffix('.zip.downloading')
        urllib.request.urlretrieve(info['url'], temporary)
        if sha(temporary.read_bytes()) != info['sha256']:
            raise RuntimeError('Downloaded archive checksum mismatch')
        temporary.replace(archive)
    if archive.stat().st_size != info['bytes'] or sha(archive.read_bytes()) != info['sha256']:
        raise RuntimeError('Archive does not match this version of the reference manifest')
    with zipfile.ZipFile(archive) as z:
        internal = json.loads(z.read('ARTIFACT_MANIFEST.json'))
        names = z.namelist()
        expected = set(internal['files_sha256']) | {'ARTIFACT_MANIFEST.json'}
        if len(names) != info['members'] or len(names) != len(set(names)) or set(names) != expected:
            raise RuntimeError('Unexpected archive contents')
        for name, digest in internal['files_sha256'].items():
            path = PurePosixPath(name)
            if path.is_absolute() or '..' in path.parts or sha(z.read(name)) != digest:
                raise RuntimeError('Invalid archive member: ' + name)
        for name, item in manifest['tables'].items():
            if z.read(item['source']) != (HERE / 'tables' / name).read_bytes():
                raise RuntimeError('Archive and browsable table differ: ' + name)
    print(f'Verified all {len(names)} archive entries. No files were extracted or models fitted.')


if __name__ == '__main__':
    main()
