"""Verify the checked-in release files against their SHA-256 manifest."""
from _common import ROOT
import hashlib,json
manifest=json.loads((ROOT/'provenance/release_manifest.json').read_text(encoding='utf-8'));errors=[]
for rel,want in manifest.items():
    p=(ROOT/rel).resolve()
    if not p.is_relative_to(ROOT) or not p.is_file():errors.append(rel+': missing/invalid path');continue
    h=hashlib.sha256(p.read_bytes()).hexdigest()
    if h!=want['sha256']:errors.append(rel+': hash mismatch')
if errors:raise SystemExit('\n'.join(errors))
print(f'PASS: {len(manifest)} release files match the manifest.')
