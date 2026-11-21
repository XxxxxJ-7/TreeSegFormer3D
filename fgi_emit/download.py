"""Download the pinned Zenodo release; large point clouds require an explicit flag."""
import argparse
import hashlib
from pathlib import Path
import shutil
import urllib.request
import zipfile

from .protocol import FILES, RECORD


def checksum(path):
    digest = hashlib.md5()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def safe_extract(archive, target):
    target = Path(target).resolve()
    with zipfile.ZipFile(archive) as zip_file:
        for member in zip_file.infolist():
            path = (target / member.filename).resolve()
            if not path.is_relative_to(target):
                raise ValueError(f'Unsafe archive member: {member.filename}')
        zip_file.extractall(target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-root', type=Path, default=Path('data/FGI_EMIT/raw'))
    parser.add_argument('--point-clouds', action='store_true',
                        help='Also download/extract training.zip and test.zip (~2.45 GB).')
    args = parser.parse_args()
    args.raw_root.mkdir(parents=True, exist_ok=True)
    for name, expected in FILES.items():
        if name in ('training.zip', 'test.zip') and not args.point_clouds:
            continue
        dest = args.raw_root / name
        if not dest.exists() or checksum(dest) != expected:
            print(f'Downloading {name} ...', flush=True)
            url = f'https://zenodo.org/api/records/{RECORD}/files/{name}/content'
            request = urllib.request.Request(url, headers={'User-Agent': 'FGI-EMIT-research/1.0'})
            partial = dest.with_suffix(dest.suffix + '.part')
            with urllib.request.urlopen(request, timeout=120) as response, partial.open('wb') as out:
                shutil.copyfileobj(response, out, length=8 * 1024 * 1024)
            if checksum(partial) != expected:
                raise ValueError(f'Checksum failed: {partial}; rerun to download again.')
            partial.replace(dest)
        print(f'Checksum OK: {name}', flush=True)
        if dest.suffix == '.zip':
            safe_extract(dest, args.raw_root)


if __name__ == '__main__':
    main()
