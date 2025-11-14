import os
import struct
import hashlib
import math

from PIL import Image
import numpy as np


MAGIC = b"WAVIMGv1"  # 8 bytes
# header = MAGIC (8) + length (8) + sha256 (32) = 48 bytes total
HEADER_FMT = ">Q32s"  # uint64 length, 32-byte sha256


def _build_header(payload: bytes) -> bytes:
    length = len(payload)
    sha = hashlib.sha256(payload).digest()
    header = MAGIC + struct.pack(HEADER_FMT, length, sha)
    return header


def _parse_header(raw: bytes):
    """Return (payload_length, sha256_digest, payload_start_offset)."""
    magic_len = len(MAGIC)
    if len(raw) < magic_len + struct.calcsize(HEADER_FMT):
        raise ValueError("Data too short to contain header.")
    magic = raw[:magic_len]
    if magic != MAGIC:
        raise ValueError("Magic header mismatch. This is not a WAVIMGv1 image.")
    header_rest = raw[magic_len:magic_len + struct.calcsize(HEADER_FMT)]
    length, sha = struct.unpack(HEADER_FMT, header_rest)
    payload_offset = magic_len + struct.calcsize(HEADER_FMT)
    return length, sha, payload_offset


def bytes_to_png(data: bytes, out_path: str, width: int = 512):
    """
    Encode arbitrary bytes into a PNG image.
    data: bytes to store
    out_path: output PNG filename
    width: image width in pixels
    """
    header = _build_header(data)
    full = header + data

    bytes_per_pixel = 3  # RGB
    pixels_needed = math.ceil(len(full) / bytes_per_pixel)
    height = math.ceil(pixels_needed / width)

    total_bytes = width * height * bytes_per_pixel
    pad_len = total_bytes - len(full)
    full_padded = full + b"\x00" * pad_len

    arr = np.frombuffer(full_padded, dtype=np.uint8)
    arr = arr.reshape((height, width, 3))

    img = Image.fromarray(arr, mode="RGB")
    img.save(out_path, format="PNG")


def png_to_bytes(image_path: str) -> bytes:
    """
    Decode bytes from a PNG created with bytes_to_png.
    Returns the original payload bytes (after verifying checksum).
    """
    img = Image.open(image_path)
    arr = np.array(img, dtype=np.uint8)
    raw = arr.tobytes()

    length, sha_expected, offset = _parse_header(raw)
    payload = raw[offset:offset + length]
    if len(payload) != length:
        raise ValueError("Image does not contain the full payload.")

    sha_actual = hashlib.sha256(payload).digest()
    if sha_actual != sha_expected:
        raise ValueError("Checksum mismatch. Image was modified or corrupted.")

    return payload


# -------- WAV-specific wrappers -------- #

def encode_wav_to_png(wav_path: str, out_png_path: str, width: int = 512):
    with open(wav_path, "rb") as f:
        wav_data = f.read()
    bytes_to_png(wav_data, out_png_path, width=width)


def decode_png_to_wav(png_path: str, out_wav_path: str):
    wav_data = png_to_bytes(png_path)
    with open(out_wav_path, "wb") as f:
        f.write(wav_data)


# -------- Probe/test workflow -------- #

def create_probe_png(out_png_path: str, payload_size: int = 1_000_000, width: int = 512):
    """
    Create a 'probe' image with random bytes to test the image channel.
    Upload this to the server, download it again, then use verify_probe_png_*.
    """
    payload = os.urandom(payload_size)
    bytes_to_png(payload, out_png_path, width=width)
    # Save original payload to compare after retrieval
    with open(out_png_path + ".origpayload", "wb") as f:
        f.write(payload)


def verify_probe_png_roundtrip(original_payload_path: str, returned_png_path: str):
    """
    Compare the original probe payload with data recovered from the returned PNG.
    This quantifies whether the image channel is bit-perfect or corrupted.
    """
    with open(original_payload_path, "rb") as f:
        original_payload = f.read()

    try:
        recovered_payload = png_to_bytes(returned_png_path)
    except Exception as e:
        print("Decoding failed:", e)
        return

    if len(original_payload) != len(recovered_payload):
        print("Length mismatch:",
              "original =", len(original_payload),
              "recovered =", len(recovered_payload))

    diff_count = sum(a != b for a, b in zip(original_payload, recovered_payload))
    print("Total bytes:", len(original_payload))
    print("Differing bytes:", diff_count)
    if diff_count == 0:
        print("Channel appears lossless for this probe.")
    else:
        print("Channel modified data. Lossless storage not guaranteed.")

def generate_random_wav_file(filename: str, duration_seconds: int = 5, sample_rate: int = 500000):
    """
    Generate a random WAV file for testing purposes.
    """
    import wave
    import numpy as np

    num_samples = duration_seconds * sample_rate
    audio_data = np.random.randint(-32768, 32767, num_samples, dtype=np.int16)

    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)  # mono
        wf.setsampwidth(2)  # 2 bytes per sample
        wf.setframerate(sample_rate)
        wf.writeframes(audio_data.tobytes())

if __name__ == "__main__":
    # Example usage:
    # Generate a random WAV file for testing
    #generate_random_wav_file("example.wav", duration_seconds=5, sample_rate=500000)
    # Encode the generated WAV file to PNG
    encode_wav_to_png("/Users/EstNsen/Downloads/M00043 2.WAV", "/Users/EstNsen/encoded_image.png", width=512)

    # Decode the PNG back to WAV
    decode_png_to_wav("/Users/EstNsen/encoded_image.png", "/Users/EstNsen/recovered_example.wav")

    # Create a probe PNG
    create_probe_png("/Users/EstNsen/probe_image.png", payload_size=1_000_000, width=512)

    # After uploading/downloading the probe image, verify it
    verify_probe_png_roundtrip("/Users/EstNsen/probe_image.png.origpayload", "/Users/EstNsen/returned_probe_image.png")