#!/usr/bin/env bash
# Bulk-load WaterKG into a new Neo4j 5 database (offline import; stop the DBMS first).
# Usage: bash import_neo4j.sh <path-to-csv-dir> [database-name]
set -euo pipefail
CSV="${1:?path to the csv/ directory}"
DB="${2:-waterkg}"
HERE="$(cd "$(dirname "$0")" && pwd)"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
# neo4j-admin reads the column header from the separate header file, so drop the CSV's own header row.
for f in entities papers categories edges_entity_category edges_category_parent edges_entity_paper relations; do
  gzip -dc "$CSV/$f.csv.gz" | tail -n +2 | gzip -1 > "$TMP/$f.csv.gz"
done
neo4j-admin database import full "$DB" \
  --nodes=Entity="$HERE/entities.header.csv,$TMP/entities.csv.gz" \
  --nodes=Paper="$HERE/papers.header.csv,$TMP/papers.csv.gz" \
  --nodes=Category="$HERE/categories.header.csv,$TMP/categories.csv.gz" \
  --relationships=BELONGS_TO="$HERE/edges_entity_category.header.csv,$TMP/edges_entity_category.csv.gz" \
  --relationships=SUBCLASS_OF="$HERE/edges_category_parent.header.csv,$TMP/edges_category_parent.csv.gz" \
  --relationships=MENTIONED_IN="$HERE/edges_entity_paper.header.csv,$TMP/edges_entity_paper.csv.gz" \
  --relationships="$HERE/relations.header.csv,$TMP/relations.csv.gz"
echo "Imported into database '$DB'. Start Neo4j, then run constraints.cypher."
