# Documentation for the updates

1. Will update this document to reflect what changes have been done, and to which portion of the tool.

2. It is by no means a hardcore changelog, but enough to document the thought process on upgrading.

# Date Jul 16, 2026

- initial commit of the tool, and updated the README to show the screenshots of how the tool looks.

# Date Jul 17, 2026

- changes to the backend/main.py
- prepared the foundational schema for the future knowledge base: for citations, chunking and embedding, categorization - node tags
- added /api/db/version to check schema version and record counts, /api/db/export to download backup of the sqlite database 
- Added the `UPDATES.md` to the `README.md` to describes the purpose of this document

# Date Jul 22, 2026

- changes to the backend/main.py
- preparation for provenance tracking
- wired /api/nodes/{node_id}/sources to fetch data related to section score, relevance, title etc
- added source tracking for nodes and QA
- other fixes for CSS
