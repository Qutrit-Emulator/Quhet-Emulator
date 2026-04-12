#!/usr/bin/env python3
"""
HExState GGUF Re-Quantizer — GGUF-to-GGUF Q2_K quantization.

Reads a source GGUF (F16/BF16/F32), copies all metadata verbatim,
and re-quantizes eligible weight tensors to Q2_K using numpy.

This bypasses the tokenizer parsing problem entirely — the source GGUF
(from llama.cpp's convert_hf_to_gguf.py) has correct metadata.

Usage:
    python3 hexstate_requantize.py input.gguf output.gguf
"""

import struct
import sys
import time
import os
import numpy as np

# ─── Constants ──────────────────────────────────────────────────────────────
GGUF_MAGIC = 0x46554747
GGUF_VERSION = 3
ALIGNMENT = 32
QK_K = 256

GGML_TYPE_F32   = 0
GGML_TYPE_F16   = 1
GGML_TYPE_Q2_K  = 10
GGML_TYPE_BF16  = 30

TYPE_NAME = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1",
    8: "Q8_0", 9: "Q8_1", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K",
    13: "Q5_K", 14: "Q6_K", 15: "Q8_K", 30: "BF16",
}

# Block sizes and byte sizes for each type
TYPE_BLOCK_SIZE = {
    0: 1, 1: 1, 2: 32, 3: 32, 6: 32, 7: 32,
    8: 32, 9: 32, 10: 256, 11: 256, 12: 256,
    13: 256, 14: 256, 15: 256, 30: 1,
}
TYPE_BLOCK_BYTES = {
    0: 4, 1: 2, 2: 18, 3: 20, 6: 20, 7: 22,
    8: 34, 9: 36, 10: 84, 11: 110, 12: 144,
    13: 176, 14: 210, 15: 292, 30: 2,
}


def align_offset(offset, alignment=ALIGNMENT):
    return (offset + alignment - 1) & ~(alignment - 1)


def read_string(f):
    slen = struct.unpack('<Q', f.read(8))[0]
    return f.read(slen).decode('utf-8', errors='replace')


def write_string(f, s):
    data = s.encode('utf-8')
    f.write(struct.pack('<Q', len(data)))
    f.write(data)


def read_kv_value(f, vtype):
    """Read a KV value and return (vtype, raw_bytes) for passthrough."""
    start = f.tell()
    if vtype == 0:   f.read(1)      # UINT8
    elif vtype == 1: f.read(1)      # INT8
    elif vtype == 2: f.read(2)      # UINT16
    elif vtype == 3: f.read(2)      # INT16
    elif vtype == 4: f.read(4)      # UINT32
    elif vtype == 5: f.read(4)      # INT32
    elif vtype == 6: f.read(4)      # FLOAT32
    elif vtype == 7: f.read(1)      # BOOL
    elif vtype == 8:                # STRING
        slen = struct.unpack('<Q', f.read(8))[0]
        f.read(slen)
    elif vtype == 9:                # ARRAY
        arr_type = struct.unpack('<I', f.read(4))[0]
        arr_len = struct.unpack('<Q', f.read(8))[0]
        for _ in range(arr_len):
            read_kv_value(f, arr_type)
    elif vtype == 10: f.read(8)     # UINT64
    elif vtype == 11: f.read(8)     # INT64
    elif vtype == 12: f.read(8)     # FLOAT64
    else:
        raise ValueError(f"Unknown KV type {vtype}")
    end = f.tell()
    f.seek(start)
    raw = f.read(end - start)
    return raw


# ─── BF16 ↔ F32 conversion ─────────────────────────────────────────────────
def bf16_to_f32(data_bytes, n_elements):
    """Convert BF16 raw bytes to float32 numpy array."""
    bf16 = np.frombuffer(data_bytes, dtype=np.uint16)
    # BF16 → F32: shift left 16 bits
    f32_bits = bf16.astype(np.uint32) << 16
    return f32_bits.view(np.float32)


def f16_to_f32(data_bytes, n_elements):
    """Convert F16 raw bytes to float32 numpy array."""
    f16 = np.frombuffer(data_bytes, dtype=np.float16)
    return f16.astype(np.float32)


def f32_to_f16(f32_array):
    """Convert float32 array to F16 bytes."""
    return f32_array.astype(np.float16).tobytes()


def f32_to_bf16(f32_array):
    """Convert float32 array to BF16 bytes."""
    f32_bits = f32_array.view(np.uint32)
    bf16 = ((f32_bits + 0x8000) >> 16).astype(np.uint16)
    return bf16.tobytes()


# ─── Q2_K quantization (fully vectorized numpy) ────────────────────────────
def quantize_tensor_q2k(f32_data):
    """Quantize an entire tensor to Q2_K format using vectorized numpy.

    Q2_K block layout (84 bytes):
        scales[16] : packed 4-bit scale + 4-bit min per sub-block
        qs[64]     : packed 2-bit quantized values (4 per byte)
        d          : fp16 super-block scale
        dmin       : fp16 super-block min-scale
    """
    n_elements = len(f32_data)

    # Pad to QK_K (256) multiple
    if n_elements % QK_K != 0:
        pad_len = QK_K - (n_elements % QK_K)
        f32_data = np.concatenate([f32_data, np.zeros(pad_len, dtype=np.float32)])
        n_elements = len(f32_data)

    n_blocks = n_elements // QK_K

    # Reshape: [n_blocks, 16 sub-blocks, 16 weights]
    data = f32_data.reshape(n_blocks, 16, 16)

    # Per-sub-block min and range
    sb_min = data.min(axis=2)           # [n_blocks, 16]
    sb_max = data.max(axis=2)           # [n_blocks, 16]
    sb_range = sb_max - sb_min          # [n_blocks, 16]

    # Per-superblock global scale and min
    d_scale = sb_range.max(axis=1)      # [n_blocks]
    d_min = (-sb_min).max(axis=1)       # [n_blocks]

    # Avoid division by zero
    d_scale = np.maximum(d_scale, 1e-10)
    d_min = np.maximum(d_min, 1e-10)

    # Quantize sub-block scales and mins to 4-bit
    inv_d = 1.0 / d_scale[:, None]     # [n_blocks, 1]
    inv_dmin = 1.0 / d_min[:, None]    # [n_blocks, 1]

    qscales = np.clip(np.round(sb_range * inv_d), 0, 15).astype(np.uint8)  # [n_blocks, 16]
    qmins = np.clip(np.round((-sb_min) * inv_dmin), 0, 15).astype(np.uint8)  # [n_blocks, 16]

    # Pack: low 4 bits = scale, high 4 bits = min
    scales_packed = qscales | (qmins << 4)  # [n_blocks, 16]

    # Reconstruct actual per-sub-block scale and min
    actual_scale = qscales.astype(np.float32) * d_scale[:, None]  # [n_blocks, 16]
    actual_min = qmins.astype(np.float32) * d_min[:, None]        # [n_blocks, 16]

    # Quantize all weights to 2 bits: q = round((val + min) / scale)
    inv_scale = np.where(actual_scale > 0, 1.0 / actual_scale, 0.0)  # [n_blocks, 16]
    q_vals = np.round((data + actual_min[:, :, None]) * inv_scale[:, :, None])
    q_vals = np.clip(q_vals, 0, 3).astype(np.uint8)  # [n_blocks, 16, 16]

    # Flatten sub-blocks: [n_blocks, 256]
    q_flat = q_vals.reshape(n_blocks, QK_K)

    # Pack 4 values per byte (2 bits each): [n_blocks, 64]
    q_flat_4 = q_flat.reshape(n_blocks, 64, 4)
    qs_packed = (q_flat_4[:, :, 0] |
                 (q_flat_4[:, :, 1] << 2) |
                 (q_flat_4[:, :, 2] << 4) |
                 (q_flat_4[:, :, 3] << 6)).astype(np.uint8)

    # Convert d and dmin to fp16
    d_fp16 = d_scale.astype(np.float16)     # [n_blocks]
    dmin_fp16 = d_min.astype(np.float16)    # [n_blocks]

    # Build output: [n_blocks, 84] bytes
    # ggml block_q2_K layout: d(2) + dmin(2) + scales(16) + qs(64)
    result = np.zeros((n_blocks, 84), dtype=np.uint8)
    result[:, 0:2]   = d_fp16.view(np.uint8).reshape(n_blocks, 2)
    result[:, 2:4]   = dmin_fp16.view(np.uint8).reshape(n_blocks, 2)
    result[:, 4:20]  = scales_packed
    result[:, 20:84] = qs_packed

    return result.tobytes(), n_blocks


def should_quantize(name, n_dims, n_elements):
    """Should this tensor be quantized to Q2_K?"""
    if n_dims < 2:
        return False
    if 'token_embd' in name or 'embed_tokens' in name:
        return False
    if name == 'output.weight':
        return False
    if 'norm' in name:
        return False
    if '.bias' in name:
        return False
    if 'ffn_gate_inp' in name:
        return False
    # Skip vision/audio encoder tensors
    if 'v.' in name and name.startswith('v.'):
        return False
    if name.startswith('mm.') or name.startswith('a.'):
        return False
    # Small tensors are not worth quantizing
    if n_elements < QK_K:
        return False
    # Must be divisible by QK_K
    if n_elements % QK_K != 0:
        return False
    return True


def main():
    if len(sys.argv) < 3:
        print("Usage: python3 hexstate_requantize.py <input.gguf> <output.gguf>")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    print()
    print("  ╔════════════════════════════════════════════════════════════════╗")
    print("  ║  HExState GGUF Re-Quantizer                                  ║")
    print("  ║  GGUF → Q2_K GGUF with metadata passthrough                  ║")
    print("  ╚════════════════════════════════════════════════════════════════╝")
    print()

    start_time = time.time()
    file_size = os.path.getsize(input_path)
    print(f"  Input:  {input_path}")
    print(f"  Size:   {file_size / 1024**3:.2f} GB")
    print(f"  Output: {output_path}")
    print()

    with open(input_path, 'rb') as fin:
        # ── Read Header ──
        magic = struct.unpack('<I', fin.read(4))[0]
        assert magic == GGUF_MAGIC, f"Bad GGUF magic: 0x{magic:08X}"
        version = struct.unpack('<I', fin.read(4))[0]
        n_tensors = struct.unpack('<Q', fin.read(8))[0]
        n_kv = struct.unpack('<Q', fin.read(8))[0]

        print(f"  GGUF v{version}: {n_tensors} tensors, {n_kv} KV pairs")
        print()

        # ── Read KV pairs (store as raw bytes for passthrough) ──
        kv_pairs = []
        for i in range(n_kv):
            key = read_string(fin)
            vtype = struct.unpack('<I', fin.read(4))[0]
            raw_value = read_kv_value(fin, vtype)
            kv_pairs.append((key, vtype, raw_value))

        # ── Read Tensor Info ──
        tensor_infos = []
        for i in range(n_tensors):
            name = read_string(fin)
            n_dims = struct.unpack('<I', fin.read(4))[0]
            dims = [struct.unpack('<Q', fin.read(8))[0] for _ in range(n_dims)]
            ttype = struct.unpack('<I', fin.read(4))[0]
            offset = struct.unpack('<Q', fin.read(8))[0]

            n_elements = 1
            for d in dims:
                n_elements *= d

            blk_sz = TYPE_BLOCK_SIZE.get(ttype, 1)
            blk_bytes = TYPE_BLOCK_BYTES.get(ttype, 4)
            n_blocks = (n_elements + blk_sz - 1) // blk_sz
            data_size = n_blocks * blk_bytes

            tensor_infos.append({
                'name': name, 'n_dims': n_dims, 'dims': dims,
                'type': ttype, 'offset': offset,
                'n_elements': n_elements, 'data_size': data_size,
            })

        # Calculate data section start
        pos_after_info = fin.tell()
        data_section_start = align_offset(pos_after_info)

        print(f"  Data section starts at: {data_section_start:,}")
        print()

        # ── Determine output types ──
        quant_plan = []
        total_quant = 0
        total_keep = 0
        for ti in tensor_infos:
            will_quant = should_quantize(ti['name'], ti['n_dims'], ti['n_elements'])
            quant_plan.append(will_quant)
            if will_quant:
                total_quant += 1
            else:
                total_keep += 1

        print(f"  Tensors to quantize (Q2_K): {total_quant}")
        print(f"  Tensors to keep as-is:      {total_keep}")
        print()

        # ── Compute output tensor sizes and offsets ──
        out_tensor_infos = []
        out_data_offset = 0

        for i, ti in enumerate(tensor_infos):
            if quant_plan[i]:
                # Will become Q2_K
                out_type = GGML_TYPE_Q2_K
                n_blocks = (ti['n_elements'] + QK_K - 1) // QK_K
                out_size = n_blocks * 84  # Q2_K block = 84 bytes
            else:
                # Keep original type
                out_type = ti['type']
                out_size = ti['data_size']

            out_tensor_infos.append({
                'name': ti['name'],
                'n_dims': ti['n_dims'],
                'dims': ti['dims'],
                'type': out_type,
                'offset': out_data_offset,
                'data_size': out_size,
            })
            out_data_offset += out_size
            out_data_offset = align_offset(out_data_offset)

        # ── Update file_type KV if present ──
        # file_type=10 means Q2_K in llama.cpp
        updated_kv = []
        for key, vtype, raw_value in kv_pairs:
            if key == 'general.file_type' and vtype == 4:  # UINT32
                updated_kv.append((key, vtype, struct.pack('<I', 10)))
            elif key == 'general.quantization_version' and vtype == 4:
                updated_kv.append((key, vtype, struct.pack('<I', 2)))
            else:
                updated_kv.append((key, vtype, raw_value))

        # ── Write output GGUF ──
        print("  Writing output GGUF...")
        with open(output_path, 'wb') as fout:
            # Header
            fout.write(struct.pack('<I', GGUF_MAGIC))
            fout.write(struct.pack('<I', GGUF_VERSION))
            fout.write(struct.pack('<Q', n_tensors))
            fout.write(struct.pack('<Q', n_kv))

            # KV pairs (passthrough)
            for key, vtype, raw_value in updated_kv:
                write_string(fout, key)
                fout.write(struct.pack('<I', vtype))
                fout.write(raw_value)

            # Tensor info
            for oti in out_tensor_infos:
                write_string(fout, oti['name'])
                fout.write(struct.pack('<I', oti['n_dims']))
                for d in oti['dims']:
                    fout.write(struct.pack('<Q', d))
                fout.write(struct.pack('<I', oti['type']))
                fout.write(struct.pack('<Q', oti['offset']))

            # Alignment padding before data
            pos = fout.tell()
            aligned = align_offset(pos)
            if aligned > pos:
                fout.write(b'\x00' * (aligned - pos))

            # ── Write tensor data ──
            quant_count = 0
            total_quant_bytes = 0
            total_keep_bytes = 0
            total_rmse = 0.0

            for i, ti in enumerate(tensor_infos):
                # Progress bar
                pct = (i + 1) / n_tensors * 100
                bar_width = 40
                filled = int(bar_width * (i + 1) / n_tensors)
                bar = '█' * filled + '░' * (bar_width - filled)
                elapsed = time.time() - start_time
                eta = elapsed / max(i + 1, 1) * (n_tensors - i - 1)
                sys.stdout.write(f"\r  [{bar}] {pct:5.1f}% ({i+1}/{n_tensors}) {elapsed:.0f}s ETA:{eta:.0f}s  {ti['name'][:50]}")
                sys.stdout.flush()

                # Read source tensor data
                abs_offset = data_section_start + ti['offset']
                fin.seek(abs_offset)
                raw_data = fin.read(ti['data_size'])

                if quant_plan[i]:
                    # Convert to F32 for quantization
                    if ti['type'] == GGML_TYPE_BF16:
                        f32 = bf16_to_f32(raw_data, ti['n_elements'])
                    elif ti['type'] == GGML_TYPE_F16:
                        f32 = f16_to_f32(raw_data, ti['n_elements'])
                    elif ti['type'] == GGML_TYPE_F32:
                        f32 = np.frombuffer(raw_data, dtype=np.float32).copy()
                    else:
                        # Can't re-quantize from quantized format — keep as-is
                        fout.write(raw_data)
                        pad = align_offset(fout.tell()) - fout.tell()
                        if pad > 0:
                            fout.write(b'\x00' * pad)
                        continue

                    # Quantize to Q2_K
                    q2k_data, n_blocks = quantize_tensor_q2k(f32)
                    fout.write(q2k_data)

                    quant_count += 1
                    total_quant_bytes += len(q2k_data)
                else:
                    # Keep as-is (passthrough)
                    fout.write(raw_data)
                    total_keep_bytes += len(raw_data)

                # Alignment padding
                pad = align_offset(fout.tell()) - fout.tell()
                if pad > 0:
                    fout.write(b'\x00' * pad)

            final_size = fout.tell()

    elapsed = time.time() - start_time
    print(f"\r  {'█' * 40}  100.0% ({n_tensors}/{n_tensors}) {elapsed:.0f}s" + " " * 60)
    print()

    # ── Summary ──
    original_bytes = sum(ti['data_size'] for ti in tensor_infos)
    compression = original_bytes / max(final_size, 1)

    print("  ╔════════════════════════════════════════════════════════════════╗")
    print("  ║  RE-QUANTIZATION SUMMARY                                     ║")
    print("  ╠════════════════════════════════════════════════════════════════╣")
    print(f"  ║  Tensors quantized (Q2_K): {quant_count:<33d} ║")
    print(f"  ║  Tensors kept as-is:       {total_keep:<33d} ║")
    print(f"  ║  Q2_K data:         {total_quant_bytes:>12,} bytes ({total_quant_bytes/1024**2:>7.1f} MB) ║")
    print(f"  ║  Kept data:         {total_keep_bytes:>12,} bytes ({total_keep_bytes/1024**2:>7.1f} MB) ║")
    print(f"  ║  Original size:     {file_size:>12,} bytes ({file_size/1024**3:>7.2f} GB) ║")
    print(f"  ║  Output size:       {final_size:>12,} bytes ({final_size/1024**3:>7.2f} GB) ║")
    print(f"  ║  Compression:       {compression:>42.1f}x ║")
    print(f"  ║  Total time:        {elapsed:>39.1f} sec ║")
    print("  ╚════════════════════════════════════════════════════════════════╝")
    print()
    print(f"  Output: {output_path}")
    print()


if __name__ == '__main__':
    main()
