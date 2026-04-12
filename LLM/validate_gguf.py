#!/usr/bin/env python3
"""Validate GGUF file structure against the v3 spec."""
import struct
import sys

GGUF_MAGIC = 0x46554747  # "GGUF" in LE
GGUF_VERSION = 3
ALIGNMENT = 32

GGML_TYPE_NAMES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1",
    8: "Q8_0", 9: "Q8_1", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K",
    13: "Q5_K", 14: "Q6_K", 15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS",
}

GGML_TYPE_SIZE = {
    0: (1, 4),    # F32: blk=1, bytes=4
    1: (1, 2),    # F16: blk=1, bytes=2
    2: (32, 18),  # Q4_0
    3: (32, 20),  # Q4_1
    8: (32, 34),  # Q8_0
    10: (256, 84),# Q2_K
    12: (256, 144),# Q4_K
    13: (256, 176),# Q5_K
    14: (256, 210),# Q6_K
}

GGUFValueType = {
    0: "UINT8", 1: "INT8", 2: "UINT16", 3: "INT16",
    4: "UINT32", 5: "INT32", 6: "FLOAT32", 7: "BOOL",
    8: "STRING", 9: "ARRAY", 10: "UINT64", 11: "INT64", 12: "FLOAT64",
}

def read_string(f):
    slen = struct.unpack('<Q', f.read(8))[0]
    return f.read(slen).decode('utf-8', errors='replace')

def read_kv_value(f, vtype):
    if vtype == 0:  return struct.unpack('<B', f.read(1))[0]
    elif vtype == 1: return struct.unpack('<b', f.read(1))[0]
    elif vtype == 2: return struct.unpack('<H', f.read(2))[0]
    elif vtype == 3: return struct.unpack('<h', f.read(2))[0]
    elif vtype == 4: return struct.unpack('<I', f.read(4))[0]
    elif vtype == 5: return struct.unpack('<i', f.read(4))[0]
    elif vtype == 6: return struct.unpack('<f', f.read(4))[0]
    elif vtype == 7: return bool(struct.unpack('<B', f.read(1))[0])
    elif vtype == 8: return read_string(f)
    elif vtype == 9:
        arr_type = struct.unpack('<I', f.read(4))[0]
        arr_len = struct.unpack('<Q', f.read(8))[0]
        if arr_len > 10:
            # Skip large arrays, just report size
            type_sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
            if arr_type == 8:  # string array
                for _ in range(arr_len):
                    slen = struct.unpack('<Q', f.read(8))[0]
                    f.read(slen)
            elif arr_type in type_sizes:
                f.read(arr_len * type_sizes[arr_type])
            else:
                # Unknown type — try to skip based on remaining array types
                f.read(arr_len)
            return f"[array of {arr_len} {GGUFValueType.get(arr_type, f'type_{arr_type}')}]"
        else:
            return [read_kv_value(f, arr_type) for _ in range(arr_len)]
    elif vtype == 10: return struct.unpack('<Q', f.read(8))[0]
    elif vtype == 11: return struct.unpack('<q', f.read(8))[0]
    elif vtype == 12: return struct.unpack('<d', f.read(8))[0]
    else:
        raise ValueError(f"Unknown KV type {vtype}")

def validate_gguf(path):
    errors = []
    warnings = []
    
    with open(path, 'rb') as f:
        file_size = f.seek(0, 2)
        f.seek(0)
        
        print(f"File: {path}")
        print(f"Size: {file_size:,} bytes ({file_size/1024/1024:.1f} MB)")
        print()
        
        # === Header ===
        magic = struct.unpack('<I', f.read(4))[0]
        if magic != GGUF_MAGIC:
            errors.append(f"Bad magic: 0x{magic:08X} (expected 0x{GGUF_MAGIC:08X})")
            print(f"FATAL: {errors[-1]}")
            return
        
        version = struct.unpack('<I', f.read(4))[0]
        n_tensors = struct.unpack('<Q', f.read(8))[0]
        n_kv = struct.unpack('<Q', f.read(8))[0]
        
        print(f"Header: magic=GGUF version={version} tensors={n_tensors} kv_pairs={n_kv}")
        if version != GGUF_VERSION:
            warnings.append(f"Version {version} (expected {GGUF_VERSION})")
        print()
        
        # === KV Pairs ===
        print(f"--- Metadata KV Pairs ({n_kv}) ---")
        for i in range(n_kv):
            key = read_string(f)
            vtype = struct.unpack('<I', f.read(4))[0]
            value = read_kv_value(f, vtype)
            
            if isinstance(value, str) and len(value) > 80:
                value = value[:77] + "..."
            print(f"  [{i:3d}] {key} = {value}")
        
        print()
        
        # === Tensor Info ===
        print(f"--- Tensor Info ({n_tensors}) ---")
        tensor_infos = []
        for i in range(n_tensors):
            name = read_string(f)
            n_dims = struct.unpack('<I', f.read(4))[0]
            dims = [struct.unpack('<Q', f.read(8))[0] for _ in range(n_dims)]
            ttype = struct.unpack('<I', f.read(4))[0]
            offset = struct.unpack('<Q', f.read(8))[0]
            
            # Compute expected size
            n_elements = 1
            for d in dims:
                n_elements *= d
            
            blk_size, blk_bytes = GGML_TYPE_SIZE.get(ttype, (1, 4))
            n_blocks = (n_elements + blk_size - 1) // blk_size
            expected_bytes = n_blocks * blk_bytes
            
            type_name = GGML_TYPE_NAMES.get(ttype, f"type_{ttype}")
            
            tensor_infos.append({
                'name': name, 'dims': dims, 'type': ttype,
                'type_name': type_name, 'offset': offset,
                'n_elements': n_elements, 'expected_bytes': expected_bytes
            })
            
            # Check Q2_K alignment
            if ttype == 10 and n_elements % 256 != 0:
                errors.append(f"Tensor '{name}': Q2_K but {n_elements} not divisible by 256")
            
            dims_str = "×".join(str(d) for d in dims)
            print(f"  [{i:3d}] {name:50s} {type_name:5s} [{dims_str}]  offset={offset}  {expected_bytes:,} bytes")
        
        print()
        
        # === Data Section ===
        # After tensor info, pad to alignment
        pos_after_info = f.tell()
        aligned_pos = (pos_after_info + ALIGNMENT - 1) & ~(ALIGNMENT - 1)
        data_section_start = aligned_pos
        
        print(f"--- Data Section ---")
        print(f"  Tensor info ends at: {pos_after_info}")
        print(f"  Data section starts at: {data_section_start} (aligned to {ALIGNMENT})")
        print()
        
        # Validate all tensor offsets
        print(f"--- Tensor Offset Validation ---")
        for i, ti in enumerate(tensor_infos):
            abs_offset = data_section_start + ti['offset']
            end_offset = abs_offset + ti['expected_bytes']
            
            # Check alignment
            if ti['offset'] % ALIGNMENT != 0 and i > 0:
                warnings.append(f"Tensor '{ti['name']}': offset {ti['offset']} not aligned to {ALIGNMENT}")
            
            if end_offset > file_size:
                errors.append(f"Tensor '{ti['name']}': data extends beyond file (offset {abs_offset} + {ti['expected_bytes']} = {end_offset} > {file_size})")
            
            status = "OK" if end_offset <= file_size else "OVERRUN"
            print(f"  [{i:3d}] {ti['name']:50s} abs={abs_offset:>12,}  end={end_offset:>12,}  {status}")
        
        # Check last tensor doesn't exceed file
        if tensor_infos:
            last = tensor_infos[-1]
            last_end = data_section_start + last['offset'] + last['expected_bytes']
            print(f"\n  Last tensor ends at: {last_end:,}")
            print(f"  File size:           {file_size:,}")
            if last_end > file_size:
                errors.append(f"Data section exceeds file by {last_end - file_size:,} bytes")
            else:
                print(f"  Remaining:           {file_size - last_end:,} bytes (padding)")
        
        print()
        
        # === Summary ===
        if errors:
            print(f"ERRORS ({len(errors)}):")
            for e in errors:
                print(f"  ❌ {e}")
        if warnings:
            print(f"WARNINGS ({len(warnings)}):")
            for w in warnings:
                print(f"  ⚠️  {w}")
        if not errors and not warnings:
            print("✅ GGUF structure appears valid")
        elif not errors:
            print("✅ GGUF structure valid (with warnings)")

if __name__ == '__main__':
    path = sys.argv[1] if len(sys.argv) > 1 else '/home/null/Pictures/HexState-d75875bfdbf32eebddc596e6862001b36519e698/Qwen2.5-0.5B-Q2_K.gguf'
    validate_gguf(path)
