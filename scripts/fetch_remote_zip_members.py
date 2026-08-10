#!/usr/bin/env python3
"""Fetch selected members from a remote ZIP using explicit HTTP byte ranges."""

from __future__ import annotations

import argparse
import binascii
import json
import struct
import zlib
from pathlib import Path

import requests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--suffix", action="append", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def fetch_range(session: requests.Session, url: str, start: int, end: int) -> bytes:
    response = session.get(
        url,
        headers={"Range": f"bytes={start}-{end}"},
        timeout=120,
        allow_redirects=True,
    )
    response.raise_for_status()
    if response.status_code != 206:
        raise RuntimeError(f"Expected HTTP 206 for range {start}-{end}")
    expected = end - start + 1
    if len(response.content) != expected:
        raise RuntimeError(
            f"Range {start}-{end} returned {len(response.content)} bytes, "
            f"expected {expected}"
        )
    return response.content


def content_length(session: requests.Session, url: str) -> int:
    response = session.head(url, timeout=120, allow_redirects=True)
    response.raise_for_status()
    value = response.headers.get("Content-Length")
    if value is None:
        raise RuntimeError("Remote server did not provide Content-Length")
    return int(value)


def central_directory(
    session: requests.Session, url: str, archive_size: int
) -> tuple[bytes, int, int]:
    tail_size = min(archive_size, 65_557)
    tail_start = archive_size - tail_size
    tail = fetch_range(session, url, tail_start, archive_size - 1)
    marker = b"PK\x05\x06"
    position = tail.rfind(marker)
    if position < 0:
        raise RuntimeError("ZIP end-of-central-directory record was not found")
    fields = struct.unpack_from("<4s4H2LH", tail, position)
    total_entries = int(fields[4])
    directory_size = int(fields[5])
    directory_offset = int(fields[6])
    directory = fetch_range(
        session,
        url,
        directory_offset,
        directory_offset + directory_size - 1,
    )
    return directory, total_entries, directory_offset


def parse_members(directory: bytes, expected_entries: int) -> list[dict[str, object]]:
    members: list[dict[str, object]] = []
    position = 0
    header_format = "<4s6H3L5H2L"
    header_size = struct.calcsize(header_format)
    while position + header_size <= len(directory):
        fields = struct.unpack_from(header_format, directory, position)
        if fields[0] != b"PK\x01\x02":
            raise RuntimeError(f"Invalid central-directory header at byte {position}")
        flags = int(fields[3])
        method = int(fields[4])
        crc32 = int(fields[7])
        compressed_size = int(fields[8])
        uncompressed_size = int(fields[9])
        filename_length = int(fields[10])
        extra_length = int(fields[11])
        comment_length = int(fields[12])
        local_offset = int(fields[16])
        name_start = position + header_size
        name_bytes = directory[name_start : name_start + filename_length]
        encoding = "utf-8" if flags & 0x800 else "cp437"
        name = name_bytes.decode(encoding)
        members.append(
            {
                "name": name,
                "flags": flags,
                "method": method,
                "crc32": crc32,
                "compressed_size": compressed_size,
                "uncompressed_size": uncompressed_size,
                "local_offset": local_offset,
            }
        )
        position = name_start + filename_length + extra_length + comment_length
    if len(members) != expected_entries:
        raise RuntimeError(
            f"Parsed {len(members)} central entries, expected {expected_entries}"
        )
    return members


def fetch_member(
    session: requests.Session, url: str, member: dict[str, object]
) -> bytes:
    local_offset = int(member["local_offset"])
    header = fetch_range(session, url, local_offset, local_offset + 29)
    fields = struct.unpack("<4s5H3L2H", header)
    if fields[0] != b"PK\x03\x04":
        raise RuntimeError(f"Invalid local header for {member['name']}")
    filename_length = int(fields[9])
    extra_length = int(fields[10])
    compressed_size = int(member["compressed_size"])
    data_start = local_offset + 30 + filename_length + extra_length
    compressed = fetch_range(
        session,
        url,
        data_start,
        data_start + compressed_size - 1,
    )
    method = int(member["method"])
    if method == 0:
        data = compressed
    elif method == 8:
        data = zlib.decompress(compressed, -zlib.MAX_WBITS)
    else:
        raise RuntimeError(f"Unsupported ZIP compression method {method}")
    if len(data) != int(member["uncompressed_size"]):
        raise RuntimeError(f"Uncompressed size mismatch for {member['name']}")
    if binascii.crc32(data) & 0xFFFFFFFF != int(member["crc32"]):
        raise RuntimeError(f"CRC mismatch for {member['name']}")
    return data


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    archive_size = content_length(session, args.url)
    directory, entry_count, directory_offset = central_directory(
        session, args.url, archive_size
    )
    members = parse_members(directory, entry_count)
    selected: list[dict[str, object]] = []
    for suffix in args.suffix:
        matches = [member for member in members if str(member["name"]).endswith(suffix)]
        if len(matches) != 1:
            raise RuntimeError(f"Suffix {suffix!r} matched {len(matches)} members")
        member = matches[0]
        data = fetch_member(session, args.url, member)
        destination = args.output_root / Path(str(member["name"])).name
        destination.write_bytes(data)
        selected.append(
            {
                "archive_name": member["name"],
                "destination": str(destination),
                "size": len(data),
                "crc32": f"{int(member['crc32']):08x}",
            }
        )
    summary = {
        "url": args.url,
        "archive_size": archive_size,
        "central_directory_offset": directory_offset,
        "archive_member_count": entry_count,
        "selected": selected,
    }
    (args.output_root / "remote_zip_members_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
