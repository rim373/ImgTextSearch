import os
from elasticsearch import Elasticsearch, helpers
import json
from dotenv import load_dotenv

# ===================== LOAD ENV =====================
load_dotenv()
# ===================== CONFIG =====================
es = os.getenv("ES_HOST", "http://localhost:9200")
INDEX_NAME = os.getenv("INDEX_NAME")
JSON_FILE = os.getenv("JSON_FILE")
BATCH_SIZE = int(os.getenv("BATCH_SIZE"))

# ===================== BULK INGEST WITH VALIDATION =====================
actions = []
count_total = 0
count_skipped = 0

with open(JSON_FILE, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue

        try:
            record = json.loads(line)
        except json.JSONDecodeError as e:
            print(f"⚠️ Skipping invalid JSON line: {e}")
            continue

        # ✅ Skip if embedding is empty or not length 512
        if not record.get("embedding") or len(record["embedding"]) != 512:
            print(f"⚠️ Skipping {record.get('id', 'unknown')}: invalid embedding")
            count_skipped += 1
            continue

        action = {
            "_index": INDEX_NAME,
            "_id": record["id"],
            "_source": record
        }
        actions.append(action)
        count_total += 1

        # Bulk ingest every BATCH_SIZE records
        if len(actions) >= BATCH_SIZE:
            helpers.bulk(es, actions)
            print(f"📌 Ingested {count_total} documents so far...")
            actions = []

# Ingest any remaining actions
if actions:
    helpers.bulk(es, actions)
    print(f"📌 Ingested final batch, total documents: {count_total}")

print(f"\n✅ Done! Total documents ingested: {count_total}, skipped: {count_skipped}")
