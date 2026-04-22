import json

files = {
    "ORIGINAL": "/home/taotao/workspace/dev/THEMIS/gpt-4o-mini.themis_swebench_lite_seed42_n20_afterexcerpt_filled.json",
    "MODIFIED_1": "/home/taotao/workspace/dev/THEMIS/gpt-4o-mini.themis_swebench_lite_seed42_n20_gpt4o_mini.json",
    "MODIFIED_2": "/home/taotao/workspace/dev/THEMIS/gpt-4o-mini.themis_swebench_lite_seed42_n20_gpt4o_mini_v2.json"
}

results = {}
for name, path in files.items():
    with open(path) as f:
        results[name] = json.load(f)

original_resolved = results["ORIGINAL"]["resolved_ids"]
print(f"Original resolved ({len(original_resolved)}): {original_resolved}")

for rid in original_resolved:
    print(f"\nTracing {rid}:")
    for name in ["MODIFIED_1", "MODIFIED_2"]:
        data = results[name]
        status = "unknown"
        if rid in data.get("resolved_ids", []): status = "RESOLVED"
        elif rid in data.get("unresolved_ids", []): status = "UNRESOLVED"
        elif rid in data.get("empty_patch_ids", []): status = "EMPTY PATCH"
        elif rid in data.get("incomplete_ids", []): status = "INCOMPLETE"
        
        print(f"  In {name}: {status}")

