"""
ecg/holter/file_format.py
==========================
Binary .ecgh file format for 24-48 hour comphrensive recordings.

Format:
  Header: 256 bytes (magic, version, leads, fs, timestamps, patient info)
  Frames: 26 bytes each (12 x int16 + 2-byte frame index)

At 500 Hz: 12 KB/second -> ~1 GB per 24 hours (uncompressed).
"""

import struct
import time
import os
import json
import numpy as np
from typing import Optional, Dict, List, Tuple


#    Constants                                                                  
MAGIC = b'ECGH'
FORMAT_VERSION = 1
HEADER_SIZE = 256
FRAME_SIZE = 26          # 12 leads x 2 bytes + 2 byte index
LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
FLUSH_EVERY_N_FRAMES = 2500   # flush every 5 seconds at 500 Hz


class ECGHFileWriter:
    """
    Writes a .ecgh binary file in real-time from streaming ECG packets.
    Thread-safe for push() calls from the Qt main thread.
    """

    def __init__(self, path: str, patient_info: dict,
                 fs: int = 500, n_leads: int = 12):
        self.path = path
        self.patient_info = patient_info
        self.fs = fs
        self.n_leads = n_leads
        self.start_time = time.time()
        self._frame_count = 0
        self._buf: List[bytes] = []
        self._index_entries: List[dict] = []   # [{byte_offset, timestamp, frame_idx}]
        self._last_index_time = 0.0
        self._closed = False

        # Open file and write header
        self._f = open(path, 'wb')
        self._write_header()

    #    Header                                                                 

    def _write_header(self):
        hdr = bytearray(HEADER_SIZE)

        # Magic + version + n_leads + fs + start_time
        struct.pack_into('>4sHHI', hdr, 0, MAGIC, FORMAT_VERSION, self.n_leads, self.fs)
        struct.pack_into('>d', hdr, 12, self.start_time)

        # Lead names: 12 x 5 bytes
        for idx, name in enumerate(LEAD_NAMES[:self.n_leads]):
            encoded = name.encode('ascii')[:5].ljust(5, b'\x00')
            hdr[20 + idx * 5: 20 + idx * 5 + 5] = encoded

        # Patient info (JSON, 152 bytes max, starting at byte 84)
        info_str = json.dumps({
            'name': self.patient_info.get('name', ''),
            'dob':  self.patient_info.get('dob', ''),
            'gender': self.patient_info.get('gender', ''),
            'doctor': self.patient_info.get('doctor', ''),
        }).encode('utf-8')[:152]
        hdr[84: 84 + len(info_str)] = info_str

        self._f.write(bytes(hdr))
        self._f.flush()

    #    Write packets                                                          

    def write_packet(self, packet: dict):
        """Called 500x per second from HolterStreamWriter."""
        if self._closed:
            return

        frame = bytearray(FRAME_SIZE)
        for i, lead in enumerate(LEAD_NAMES[:self.n_leads]):
            val = int(packet.get(lead, 2048))
            val = max(-32768, min(32767, val))
            struct.pack_into('>h', frame, i * 2, val)

        frame_idx = self._frame_count % 65536
        struct.pack_into('>H', frame, 24, frame_idx)
        self._buf.append(bytes(frame))
        self._frame_count += 1

        # 1-second index entry for fast seeking
        current_time = self.start_time + self._frame_count / self.fs
        if current_time - self._last_index_time >= 1.0:
            byte_offset = HEADER_SIZE + (self._frame_count - 1) * FRAME_SIZE
            self._index_entries.append({
                'time': current_time,
                'frame': self._frame_count - 1,
                'offset': byte_offset
            })
            self._last_index_time = current_time

        # Flush every 5 seconds
        if len(self._buf) >= FLUSH_EVERY_N_FRAMES:
            self._flush()

    def _flush(self):
        if self._buf:
            self._f.write(b''.join(self._buf))
            self._buf.clear()
            self._f.flush()

    def finalize(self) -> dict:
        """Flush, close file, write index. Returns recording summary."""
        if self._closed:
            return {}
        self._flush()
        self._f.close()
        self._closed = True

        duration = self._frame_count / max(1, self.fs)
        index_path = self.path.replace('.ecgh', '_index.json')
        with open(index_path, 'w') as f:
            json.dump({
                'path': self.path,
                'start_time': self.start_time,
                'duration_sec': duration,
                'frames': self._frame_count,
                'fs': self.fs,
                'n_leads': self.n_leads,
                'index': self._index_entries,
            }, f)

        return {
            'path': self.path,
            'duration_sec': duration,
            'frames': self._frame_count,
            'index_path': index_path,
        }

    @property
    def elapsed_seconds(self) -> float:
        return self._frame_count / max(1, self.fs)


class ECGHFileReader:
    """
    Reads a .ecgh binary file for playback and analysis.
    """

    def __init__(self, path: str):
        self.path = path
        self._f = None
        self._ensure_open()
        self._parse_header()
        self._load_index()

    def _ensure_open(self):
        if self._f is None or getattr(self._f, 'closed', False):
            self._f = open(self.path, 'rb')

    def _parse_header(self):
        self._ensure_open()
        hdr = self._f.read(HEADER_SIZE)
        magic, version, n_leads, fs = struct.unpack_from('>4sHHI', hdr, 0)
        if magic != MAGIC:
            raise ValueError(f"Not a valid .ecgh file: {self.path}")

        self.version = version
        self.n_leads = n_leads
        self.fs = fs
        self.start_time = struct.unpack_from('>d', hdr, 12)[0]

        self.lead_names = []
        for i in range(n_leads):
            name = hdr[20 + i * 5: 25 + i * 5].rstrip(b'\x00').decode('ascii')
            self.lead_names.append(name)

        try:
            info_bytes = hdr[84:236].rstrip(b'\x00')
            self.patient_info = json.loads(info_bytes)
        except Exception:
            self.patient_info = {}

        # Calculate total frames from file size
        file_size = os.path.getsize(self.path)
        self.total_frames = (file_size - HEADER_SIZE) // FRAME_SIZE
        self.duration_sec = self.total_frames / max(1, self.fs)

    def _load_index(self):
        index_path = self.path.replace('.ecgh', '_index.json')
        self._index = []
        if os.path.exists(index_path):
            with open(index_path) as f:
                data = json.load(f)
                self._index = data.get('index', [])

    def _seek_to_second(self, target_sec: float):
        """Fast seek using index, then fine-tune by frame."""
        self._ensure_open()
        target_frame = int(target_sec * self.fs)
        target_frame = max(0, min(target_frame, self.total_frames - 1))

        # Find nearest index entry
        best_frame = 0
        for entry in self._index:
            if entry['frame'] <= target_frame:
                best_frame = entry['frame']
            else:
                break

        byte_offset = HEADER_SIZE + best_frame * FRAME_SIZE
        self._f.seek(byte_offset)
        # Skip remaining frames to reach target
        skip = target_frame - best_frame
        if skip > 0:
            self._f.seek(skip * FRAME_SIZE, 1)

    def read_range(self, start_sec: float, end_sec: float) -> np.ndarray:
        """
        Returns ndarray of shape (n_leads, N_samples).
        """
        self._ensure_open()
        start_sec = max(0.0, start_sec)
        end_sec = min(self.duration_sec, end_sec)
        n_frames = int((end_sec - start_sec) * self.fs)
        if n_frames <= 0:
            return np.zeros((self.n_leads, 0), dtype=np.float32)

        self._seek_to_second(start_sec)
        raw = self._f.read(n_frames * FRAME_SIZE)
        actual = len(raw) // FRAME_SIZE
        if actual <= 0:
            return np.zeros((self.n_leads, 0), dtype=np.float32)

        # Each frame is 13 big-endian int16 values (12 lead samples + the
        # frame index); decode the whole block at once instead of one
        # struct.unpack_from call per sample.
        frames = np.frombuffer(raw, dtype='>i2', count=actual * (FRAME_SIZE // 2))
        frames = frames.reshape(actual, FRAME_SIZE // 2)
        return np.ascontiguousarray(frames[:, :self.n_leads].T, dtype=np.float32)

    def iter_chunks(self, chunk_sec: float = 30.0):
        """Generator: yields (start_sec, data_array) for each chunk."""
        start = 0.0
        while start < self.duration_sec:
            end = min(start + chunk_sec, self.duration_sec)
            data = self.read_range(start, end)
            yield start, data
            start = end

    def get_duration_seconds(self) -> float:
        return self.duration_sec

    def close(self):
        if hasattr(self, '_f') and self._f and not getattr(self._f, 'closed', False):
            try:
                self._f.close()
            except Exception:
                pass

    def __enter__(self):
        self._ensure_open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

