
import sys
import json
import re

def analyze_trace_json(filename):
    try:
        with open(filename, 'r') as f:
            trace_data = json.load(f)
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    total_time = 0
    categories = {
        "INTT": 0,
        "NTT": 0,
        "VecModOps": 0,
        "BConv": 0,
        "Memory Reorder": 0,
        "Permutation": 0,
        "Type Conversion": 0,
        "Other": 0
    }
    subcategories = {}

    # Iterate over all items in the dictionary
    for key, event in trace_data.items():
        # strict check for duration
        if "dur" not in event:
            continue
            
        dur_list = event["dur"]
        if not dur_list:
            continue
            
        dur = float(dur_list[0])
        total_time += dur
        
        # Determine description string for classification
        # Prefer args['long_name'], then name
        name = event.get("name", "")
        
        # Exclude top-level jit_compiled_kernel_function container to avoid double counting
        if name.startswith("jit_compiled_kernel_function"):
            continue
            
        description = name
        if "args" in event:
            args = event["args"]
            long_name = args.get("long_name", "")
            tf_op = args.get("tf_op", "")
            # Combine all useful info for matching
            description = f"{name} {long_name} {tf_op}"
            
        # Clean description for simpler matching (optional, but consistent with previous text processing)
        # The previous script did line.split() then joined parts[2:].
        # Here we just use the string.
        
        # Classification Logic (Copied and adapted from analyze_trace.py)
        is_classified = False
        
        if "dot_general" in description:
            if "brcmq,cqlpm->brlmp" in description or "brcmq,lqrpm->blcmp" in description:
                categories["INTT"] += dur
                is_classified = True
            elif "brcmq,zqrpm->bzcmp" in description or "brcmq,cqnpm->brnmp" in description:
                categories["NTT"] += dur
                is_classified = True
            elif "qpb->...pb" in description or ",qpb->" in description: # BConv pattern
                categories["BConv"] += dur
                is_classified = True
            else:
                # Unclassified dot_general
                pass 
                
        if is_classified:
            continue

        # 1.5 Permutation (jax.take) - Prioritize over Type Conversion if take is the semantic op
        if "take" in description:
             categories["Permutation"] += dur
             sub_name = "take"
             if "gather" in description:
                 sub_name = "take/gather"
             subcategories.setdefault("Permutation", {}).setdefault(sub_name, 0)
             subcategories["Permutation"][sub_name] += dur
             continue

        # 1.6 VecModOps (prioritize over Type Conversion for bitcast_reduce etc)
        if "select_n" in description:
             categories["VecModOps"] += dur
             subcategories.setdefault("VecModOps", {}).setdefault("select_n", 0)
             subcategories["VecModOps"]["select_n"] += dur
             continue
        
        alu_keywords = ["add", "multiply", "shift", "and", "or", "xor", "not", "clamp", "reduce_sum", "reduce", "sub", "div", "rem", "neg"]
        
        found_alu = False
        for kw in alu_keywords:
            if kw in description:
                categories["VecModOps"] += dur
                subcategories.setdefault("VecModOps", {}).setdefault(kw, 0)
                subcategories["VecModOps"][kw] += dur
                found_alu = True
                break
        if found_alu:
            continue
            
        # 1.7 Type Conversion
        if "convert" in description:
             categories["Type Conversion"] += dur
             sub_name = "convert_element_type" if "convert_element_type" in description else "convert"
             subcategories.setdefault("Type Conversion", {}).setdefault(sub_name, 0)
             subcategories["Type Conversion"][sub_name] += dur
             continue

        if "bitcast" in description:
             # Check for real type conversion (u8 <-> u32)
             if "u8" in description and "u32" in description:
                 categories["Type Conversion"] += dur
                 subcategories.setdefault("Type Conversion", {}).setdefault("bitcast", 0)
                 subcategories["Type Conversion"]["bitcast"] += dur
             else:
                 # Otherwise categorize as VecModOps (e.g. pure reshapes or other bitcasts)
                 categories["VecModOps"] += dur
                 subcategories.setdefault("VecModOps", {}).setdefault("bitcast", 0)
                 subcategories["VecModOps"]["bitcast"] += dur
             continue

        # 3. Memory Reorder
        mem_keywords = ["gather", "reshape", "concatenate", "slice", "broadcast", "transpose", "pad", "copy"]
        found_mem = False
        for kw in mem_keywords:
            if kw in description:
                categories["Memory Reorder"] += dur
                subcategories.setdefault("Memory Reorder", {}).setdefault(kw, 0)
                subcategories["Memory Reorder"][kw] += dur
                found_mem = True
                break
        if found_mem:
             continue
        
        # 4. Other
        name_key = event.get("name", "unknown")
        categories["Other"] += dur
        subcategories.setdefault("Other", {}).setdefault(name_key, 0)
        subcategories["Other"][name_key] += dur

    # Recalculate total_time based on sum of categories to avoid double counting from parent
    total_time = sum(categories.values())
    
    print(f"Total Trace Duration (Sum): {total_time:.4f} us")
    print("-" * 30)
    for cat, time in categories.items():
        percent = (time / total_time * 100) if total_time > 0 else 0
        print(f"{cat}: {time:.4f} us ({percent:.2f}%)")
        if cat in subcategories:
            sorted_subs = sorted(subcategories[cat].items(), key=lambda x: x[1], reverse=True)
            for sub, sub_time in sorted_subs:
                sub_percent = (sub_time / time * 100) if time > 0 else 0
                print(f"  - {sub}: {sub_time:.4f} us ({sub_percent:.2f}%)")

if __name__ == "__main__":
    filename = "/home/jianming/work/FHE/CROSS/log/CROSS_v3/log_v6e8/herot_trace_filter.json"
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    analyze_trace_json(filename)
