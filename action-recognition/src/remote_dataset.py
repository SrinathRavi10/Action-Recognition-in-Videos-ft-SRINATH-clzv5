"""
Lets the frontend browse and fetch individual UCF101 clips on demand,
WITHOUT downloading the full ~7GB dataset — the listing and each
single-clip fetch only pull the bytes actually needed, using HTTP
Range requests against the remote zip file.

How this works: a zip file's list of contents (the "central directory")
sits at the END of the file, and is small (a few hundred KB even for a
huge archive) — reading it doesn't require downloading everything before
it. Python's zipfile module already knows how to do this IF given a
file-like object that supports seek()/read() — it will naturally seek to
the end first. RemoteZipReader below implements exactly that minimal
file-like interface over HTTP Range requests, so zipfile does the rest:
listing costs a handful of small requests, and extracting one member
only fetches that member's own bytes (its local header + compressed
data), not the whole archive.

Usage:
    from src.remote_dataset import list_remote_clips, fetch_clip_to_temp

    clips = list_remote_clips()  # {class_name: [member_name, ...]}
    local_path = fetch_clip_to_temp(member_name)
    ...
    os.remove(local_path)  # caller's responsibility to clean up
"""

import io
import os
import re
import tempfile
import zipfile

import requests
from huggingface_hub import hf_hub_url

HF_REPO = "quchenyuan/UCF101-ZIP"
HF_FILENAME = "UCF-101.zip"

# Matches paths like "UCF-101/JugglingBalls/v_JugglingBalls_g01_c01.avi"
_CLIP_PATH_RE = re.compile(r"([^/]+)/([^/]+\.(?:avi|mp4))$", re.IGNORECASE)


class RemoteZipReader(io.RawIOBase):
    """
    Minimal seekable, readable file-like object backed by HTTP Range
    requests, just enough for zipfile.ZipFile to treat a remote URL like
    a local file — without ever downloading the whole thing.
    """

    def __init__(self, url: str, session: requests.Session = None):
        self.url = url
        self.session = session or requests.Session()
        self._pos = 0
        self._size = self._fetch_size()

    def _fetch_size(self) -> int:
        resp = self.session.head(self.url, allow_redirects=True, timeout=30)
        resp.raise_for_status()
        size = resp.headers.get("Content-Length")
        if size is None:
            raise RuntimeError(f"Server did not report Content-Length for {self.url}")
        return int(size)

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._pos

    def seek(self, offset: int, whence: int = 0) -> int:
        if whence == io.SEEK_SET:
            self._pos = offset
        elif whence == io.SEEK_CUR:
            self._pos += offset
        elif whence == io.SEEK_END:
            self._pos = self._size + offset
        else:
            raise ValueError(f"Unsupported whence: {whence}")
        return self._pos

    def readinto(self, b) -> int:
        n = len(b)
        if n == 0 or self._pos >= self._size:
            return 0
        end = min(self._pos + n, self._size) - 1
        headers = {"Range": f"bytes={self._pos}-{end}"}
        resp = self.session.get(self.url, headers=headers, timeout=60)
        resp.raise_for_status()
        data = resp.content
        b[:len(data)] = data
        self._pos += len(data)
        return len(data)


def _get_remote_zip_url() -> str:
    return hf_hub_url(repo_id=HF_REPO, filename=HF_FILENAME, repo_type="dataset")


def list_remote_clips() -> dict:
    """
    Returns {class_name: [member_name, ...]} for every clip in the
    dataset, by reading only the zip's central directory — not the
    dataset's actual video content.
    """
    url = _get_remote_zip_url()
    reader = RemoteZipReader(url)
    with zipfile.ZipFile(reader) as zf:
        names = zf.namelist()

    by_class = {}
    for name in names:
        match = _CLIP_PATH_RE.search(name)
        if match:
            class_name, _ = match.groups()
            by_class.setdefault(class_name, []).append(name)
    for class_name in by_class:
        by_class[class_name].sort()
    return by_class


def fetch_clip_to_temp(member_name: str) -> str:
    """
    Extracts exactly one clip's bytes from the remote zip into a local
    temp file, without downloading anything else. Caller is responsible
    for deleting the returned path when done with it.
    """
    url = _get_remote_zip_url()
    reader = RemoteZipReader(url)
    with zipfile.ZipFile(reader) as zf:
        data = zf.read(member_name)

    suffix = os.path.splitext(member_name)[1] or ".avi"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(data)
    tmp.close()
    return tmp.name
