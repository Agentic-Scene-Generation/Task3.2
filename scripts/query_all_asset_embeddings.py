"""Read-only all-source retrieval; preserve namespaced IDs and mesh paths."""
import argparse
import json
from pathlib import Path
import urllib.request

import numpy as np
import zvec

try:
    from .index_rendered_hssd_assets_zvec import _extract_embedding
except ImportError:
    from index_rendered_hssd_assets_zvec import _extract_embedding


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--collection-path', type=Path, required=True)
    parser.add_argument('--base-url', required=True)
    parser.add_argument('--model-id', required=True, help='Operator-verified exact backend identity')
    parser.add_argument('--text', required=True)
    parser.add_argument('--top-k', type=int, default=5)
    args = parser.parse_args()
    if args.top_k < 1:
        raise ValueError('top-k must be positive')
    signature = json.loads(args.collection_path.with_name(args.collection_path.name + '.input.json').read_text())
    if signature['model_id'] != args.model_id:
        raise ValueError('Query model/backend differs from index signature')
    request = urllib.request.Request(
        args.base_url.rstrip('/') + '/embeddings',
        data=json.dumps({'content': args.text, 'embd_normalize': 2}).encode(),
        headers={'Content-Type': 'application/json'}, method='POST')
    with urllib.request.urlopen(request, timeout=120) as response:
        vector = _extract_embedding(json.load(response))
    array = np.asarray(vector)
    if array.shape != (signature['dimension'],) or not np.isfinite(array).all() or np.linalg.norm(array) == 0:
        raise ValueError('Invalid query vector')
    collection = zvec.open(str(args.collection_path), option=zvec.CollectionOption(read_only=True))
    try:
        results = collection.query(queries=zvec.Query(field_name='embedding', vector=vector),
                                   topk=args.top_k,
                                   output_fields=['asset_id', 'name', 'asset_path', 'image_paths', 'content'])
        print(json.dumps([{'score': doc.score, **doc.fields} for doc in results], ensure_ascii=False, indent=2))
    finally:
        collection.close()


if __name__ == '__main__':
    main()
