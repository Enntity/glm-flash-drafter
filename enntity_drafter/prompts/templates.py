"""Template prompt corpus (round 1's "template" source): Enntity-authored, deterministic, offline.

Every prompt is generated here from templates written for this project (code,
bug fixing, SQL, JSON, extraction, configs, tool calls, tables, prose, writing,
multilingual, math).  No model output, no third-party dataset, and nothing from
the incoai/GLM-5.3-Flash-DFlash2 checkpoint is used.  Names, orders, addresses
and records in the passages are fictional.

Round 1 also used two other prompt pools that are not published here: earlier
Enntity-authored prompts and CPython standard-library excerpts (PSF-2.0).  This
builder produces the 1,592 template rows of round 1 (same ids and text).

Rows are chat requests (OpenAI message format) ready for scripts/api_generate.py:
``id, source="template", category, messages, tools?, thinking, max_tokens,
reasoning_effort, temperature, split, license, prompt_sha``.

    python -m enntity_drafter.prompts.templates --out data/prompts/templates
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path

SEED = 20261014

# ---------------------------------------------------------------------------
# Exclusions: earlier held-out screen topics (consensus, idempotency, interval
# merging, LRU cache) and the matched benchmark prompts of scripts/atlas_check.py
# (hash-map prose, TCP congestion-control prose, LRU-with-TTL code, JSON
# tokenizer code, plus its warmup). Whole topics are excluded, not only the
# exact strings, so no near-duplicate leaks into training.
# ---------------------------------------------------------------------------
EXCLUDED_EXACT = [
    'Explain, in flowing prose with no lists, how a hash map works.',
    'Explain, in flowing prose with no lists, how TCP congestion control works.',
    'Write a complete Python implementation of an LRU cache with TTL with type hints. Code only.',
    'Write a complete Python implementation of a JSON tokenizer with type hints. Code only.',
    'Reply with a Python function that adds two integers.',
]
EXCLUDED_TOPIC = re.compile(
    r'\blru\b|least[- ]recently[- ]used|consensus|\braft\b|paxos|idempoten|'
    r'interval|overlapping (ranges|intervals)|hash ?map|hashmap|hash table|'
    r'congestion control|json tokeni[sz]er|flowing prose with no lists|'
    r'with type hints\. code only', re.I)


def excluded(text: str) -> str | None:
    if any(e in text for e in EXCLUDED_EXACT):
        return 'exact-benchmark-prompt'
    m = EXCLUDED_TOPIC.search(text)
    return f'topic:{m.group(0).lower()}' if m else None


def row(rid, category, source, license_, messages, *, tools=None, thinking=False, max_tokens=1024, **extra):
    r = {'id': rid, 'category': category, 'source': source, 'license': license_,
         'messages': messages, 'thinking': thinking, 'max_tokens': max_tokens}
    if tools:
        r['tools'] = tools
    r.update(extra)
    return r


def user(text):
    return [{'role': 'user', 'content': text}]



# ---------------------------------------------------------------------------
# Templated prompts written for this project
# ---------------------------------------------------------------------------
OWN = 'Enntity-authored template; no third-party text'

LANGS = ['Python', 'TypeScript', 'Rust', 'Go', 'Java', 'C++', 'C#', 'Kotlin', 'JavaScript', 'Swift']
CODE_TASKS = [
    'a binary min-heap priority queue with decrease-key',
    'a trie that supports insert, exact search and counting words with a given prefix',
    "Dijkstra's shortest-path algorithm over an adjacency-list graph, returning the path",
    'a fixed-capacity circular buffer with explicit overwrite policy',
    'a union-find structure with path compression and union by rank',
    'topological sorting that reports a cycle when one exists',
    'a parser for ISO-8601 calendar dates without using a date library',
    'a CSV line parser that handles quoted fields and escaped quotes',
    'an INI-file parser that returns nested sections',
    'a URL query-string parser that decodes percent-escapes and repeated keys',
    'binary search that returns the first index of a target in a sorted array with duplicates',
    'run-length encoding and decoding with round-trip tests',
    'an arithmetic expression evaluator using the shunting-yard algorithm',
    'a retry helper with exponential backoff, jitter and a maximum elapsed time',
    'an in-memory key-value store with snapshot and rollback',
    'a compact bitset with set, clear, test and population count',
    'a sparse vector type with an efficient dot product',
    'one iteration of k-means clustering on 2-D points',
    'a word-frequency counter that reads a text file line by line',
    'a bank account model with transfers that never allow a negative balance',
    'a finite state machine for a turnstile with coin and push events',
    'a minimal command-line argument parser supporting flags and positional arguments',
    'the Levenshtein edit distance with an O(min(m, n)) memory variant',
    'the longest common subsequence with reconstruction of one subsequence',
    'a Sudoku board validator',
    'an iterative flood fill on a character grid',
    'an N-queens solver that counts all solutions',
    'Rabin-Karp substring search with a rolling hash',
    'a stable merge sort over a generic comparable type',
    'quickselect for the k-th smallest element',
    'a double-ended queue backed by a growable ring buffer',
    'a skip list with insert, delete and search',
    'a Bloom filter sized from a target false-positive rate',
    'a consistent-hashing ring with virtual nodes',
    'a semantic-version parser and comparator',
    'an IPv4 address validator plus a CIDR containment check',
    'Base64 encoding and decoding without the standard library helper',
    'a UTF-8 byte-sequence validator',
    'a tiny regular-expression matcher supporting literals, dot and star',
    'an inventory model that raises reorder alerts below a threshold',
    'a spiral-order traversal of a matrix',
    'the maximum of every sliding window of size k using a deque',
    'prefix autocomplete ranked by query frequency',
    'a pagination helper that returns page metadata and handles out-of-range pages',
    'a keyed diff of two record lists reporting added, removed and changed records',
    'a duplicate-file finder that groups files by content hash',
    'an HTTP/1.1 request-line and header parser',
    'a web-server access-log analyzer that aggregates status codes per hour',
    'an undo/redo history for a text buffer',
    'an exponential moving average over a stream of samples',
    'a histogram builder with configurable bucket boundaries',
    'a task scheduler that runs the highest-priority ready task first',
    'matrix multiplication with dimension checking',
    'a thread-safe counter and a test that hammers it from several threads',
    'a memoized Fibonacci and an iterative version, with a comparison',
    'a Roman-numeral converter in both directions',
    'a simple tokenizer for a calculator language with numbers, identifiers and operators',
    'a generic LFU-free object pool that reuses released objects',
    'a breadth-first search that returns the shortest path in a grid maze',
    'a rate counter that reports events per second over the last minute',
]
CODE_FORMS = [
    'Respond with the code only.',
    'Include unit tests.',
    'Briefly explain the approach, then give the code.',
    'Add documentation comments and finish with two edge cases the code handles.',
    'Write idiomatic, production-quality code with error handling, then explain one design choice.',
    'Show the code, then state its time and space complexity.',
]
CODE_OPENERS = ['Implement {task} in {lang}.', 'Write {task} in {lang}.', 'In {lang}, implement {task}.',
                'I need {task}, written in {lang}.']

BUGGY = [
    ('Python', 'def average(xs):\n    total = 0\n    for x in xs:\n        total += x\n    return total / len(xs) - 1',
     'it should return the arithmetic mean and handle an empty list'),
    ('Python', 'def chunks(xs, n):\n    return [xs[i:i + n] for i in range(0, len(xs) - n, n)]',
     'the last partial chunk is dropped'),
    ('JavaScript', 'function unique(arr) {\n  const seen = {};\n  return arr.filter(x => seen[x] ? false : seen[x] = true);\n}',
     'it treats 1 and "1" as the same value'),
    ('Go', 'func Reverse(s string) string {\n\tb := []byte(s)\n\tfor i, j := 0, len(b)-1; i < j; i, j = i+1, j-1 {\n\t\tb[i], b[j] = b[j], b[i]\n\t}\n\treturn string(b)\n}',
     'it corrupts multi-byte characters'),
    ('Rust', 'fn max_index(v: &[i32]) -> usize {\n    let mut best = 0;\n    for i in 0..v.len() {\n        if v[i] > v[best] { best = i; }\n    }\n    best\n}',
     'it panics or misbehaves on an empty slice'),
    ('Python', 'import threading\ncount = 0\ndef work():\n    global count\n    for _ in range(100000):\n        count += 1\nthreads = [threading.Thread(target=work) for _ in range(4)]\nfor t in threads: t.start()\nprint(count)',
     'the printed count is wrong and nondeterministic'),
    ('TypeScript', 'async function fetchAll(urls: string[]) {\n  const out = [];\n  urls.forEach(async u => { out.push(await fetch(u)); });\n  return out;\n}',
     'it returns before any request finishes'),
    ('Java', 'public static boolean isPalindrome(String s) {\n    for (int i = 0; i < s.length(); i++)\n        if (s.charAt(i) != s.charAt(s.length() - i)) return false;\n    return true;\n}',
     'it throws an exception'),
    ('C++', 'std::vector<int> evens(const std::vector<int>& v) {\n    std::vector<int> out;\n    for (size_t i = 0; i <= v.size(); ++i) if (v[i] % 2 == 0) out.push_back(v[i]);\n    return out;\n}',
     'it reads past the end of the vector'),
    ('Python', 'def append_item(item, items=[]):\n    items.append(item)\n    return items',
     'separate calls share state unexpectedly'),
    ('SQL', 'SELECT c.name, SUM(o.amount)\nFROM customers c LEFT JOIN orders o ON o.customer_id = c.id\nWHERE o.created_at >= DATE \'2026-01-01\'\nGROUP BY c.name;',
     'customers without 2026 orders disappear from the result'),
    ('Bash', 'for f in $(ls *.log); do\n  gzip $f\ndone',
     'it breaks on file names containing spaces'),
]
BUG_FORMS = ['Find the bug, explain it, and give a corrected version.',
             'Explain why this happens and fix it with a minimal change.',
             'Fix the code and add a test that would have caught the problem.']

SQL_SCHEMAS = [
    ('orders(id, customer_id, created_at, amount), customers(id, name, country)',
     ['total revenue per country for the last 90 days', 'the top five customers by lifetime spend',
      'customers whose first order was in the current month', 'month-over-month revenue growth',
      'the median order amount per customer']),
    ('employees(id, name, manager_id, department, salary, hired_at)',
     ['each manager with the number of direct reports', 'the highest-paid employee in each department',
      'the full management chain above a given employee using a recursive query',
      'departments whose average salary exceeds the company average', 'employees hired in the same month as their manager']),
    ('page_views(user_id, url, viewed_at), signups(user_id, signed_up_at)',
     ['daily active users for the past 30 days', 'the conversion rate from first page view to signup',
      'the three most viewed URLs per day', 'users who viewed a page but never signed up',
      'the average number of page views before signup']),
    ('products(id, name, category, price), order_lines(order_id, product_id, quantity)',
     ['revenue by category', 'products that were never ordered', 'the best-selling product in each category',
      'pairs of products frequently bought together', 'the share of revenue from the top 10% of products']),
]
SQL_DIALECTS = ['PostgreSQL', 'SQLite', 'MySQL']

JSON_ENTITIES = [
    ('a fictional hardware store product', '{"sku": string, "name": string, "price_usd": number, "in_stock": boolean, "tags": [string]}'),
    ('a weeknight vegetarian recipe', '{"title": string, "servings": integer, "ingredients": [{"item": string, "quantity": string}], "steps": [string], "minutes": integer}'),
    ('a bug report for a mobile app crash', '{"title": string, "severity": "low"|"medium"|"high"|"critical", "steps_to_reproduce": [string], "expected": string, "actual": string, "platform": string}'),
    ('a three-day conference schedule', '{"name": string, "days": [{"date": string, "sessions": [{"time": string, "title": string, "room": string}]}]}'),
    ('a fictional employee directory entry', '{"id": integer, "full_name": string, "title": string, "team": string, "skills": [string], "start_date": string}'),
    ('a book recommendation list with four entries', '{"theme": string, "books": [{"title": string, "author": string, "year": integer, "why": string}]}'),
    ('an API error response for an expired token', '{"error": {"code": string, "message": string, "status": integer, "retry_after_seconds": integer|null}}'),
    ('a fictional weather forecast for a coastal town', '{"location": string, "days": [{"date": string, "high_c": number, "low_c": number, "conditions": string, "wind_kph": number}]}'),
    ('a feature flag configuration for a gradual rollout', '{"flag": string, "enabled": boolean, "rollout_percent": integer, "segments": [{"name": string, "rule": string}]}'),
    ('a fictional flight itinerary with one connection', '{"passenger": string, "segments": [{"flight": string, "from": string, "to": string, "departs": string, "arrives": string}], "total_minutes": integer}'),
    ('a database migration plan', '{"migration_id": string, "steps": [{"order": integer, "sql": string, "reversible": boolean}], "estimated_downtime_seconds": integer}'),
    ('a fictional music festival lineup', '{"festival": string, "stages": [{"stage": string, "acts": [{"artist": string, "start": string}]}]}'),
    ('a customer support ticket summary', '{"ticket_id": string, "customer_sentiment": "negative"|"neutral"|"positive", "issue": string, "actions_taken": [string], "next_step": string}'),
    ('a personal workout plan for one week', '{"goal": string, "days": [{"day": string, "exercises": [{"name": string, "sets": integer, "reps": integer}]}]}'),
    ('a fictional apartment listing', '{"address": string, "bedrooms": integer, "bathrooms": number, "rent_usd": integer, "amenities": [string], "pets_allowed": boolean}'),
]
JSON_FORMS = ['Return only valid JSON matching this shape: {shape}. Create {entity}.',
              'Produce {entity} as JSON. Use exactly this structure and no extra keys: {shape}',
              'Generate {entity}. Output a single JSON object of the form {shape} and nothing else.',
              'Return a JSON array with three different examples of {entity}, each shaped like {shape}. No prose.',
              'Write a TypeScript interface for this shape: {shape}. Then give one example value representing {entity} as JSON.',
              'Output {entity} as JSON Lines (one compact object per line, three lines), each shaped like {shape}.']

PASSAGES = [
    'On March 3, the Riverside Library will host a free workshop called "Intro to Bookbinding" from 2:00 to 4:30 pm in Room B. The instructor is Maya Okafor. Seats are limited to 12 people, and registration closes on February 28. Materials cost $8, payable at the door.',
    'Order #A-55102 shipped on 2026-05-14 via ground freight. It contains two items: a 27-inch monitor (qty 1, $219.99) and an HDMI cable (qty 3, $7.49 each). The customer, Jonah Reyes, requested delivery to the side entrance. Tracking number: 1Z84F00R2.',
    'The town council approved a budget of 4.2 million dollars for road repair, with 60 percent allocated to the north district. The vote was 5 to 2. Council member Ana Lindqvist opposed the measure, citing the lack of a bicycle lane plan. Work is scheduled to begin in July.',
    'Patient-facing note: Your follow-up appointment with Dr. Imani Torres is on Tuesday, 9 June, at 10:15. Please arrive 15 minutes early, bring your insurance card, and avoid eating for 8 hours beforehand. Call 555-0142 to reschedule.',
    'Release 3.4.0 of the toolkit adds a streaming parser, deprecates the legacy XML exporter, and fixes a memory leak in the image cache. Breaking change: the config key "cache_dir" is now "cache.path". Upgrading requires Node 20 or later.',
    'The research vessel Calder departed Halifax on 2 August with a crew of 18 and returned after 41 days. It collected 312 water samples from depths between 50 and 2,000 meters. The expedition was led by Dr. Rafael Moreau and funded by a coastal science grant.',
    'Job posting: Senior Data Engineer, remote within the EU. Salary range 85,000 to 105,000 euros. Requirements: five years with Python and SQL, experience with streaming systems, and fluency in English. Applications close on 30 September. Contact: hiring@example.org.',
    'Incident summary: at 03:12 UTC the payment-status service began returning 503 errors after a configuration push reduced its connection pool from 50 to 5. Error rates peaked at 38 percent. The change was rolled back at 03:41 UTC and full recovery was confirmed at 03:47 UTC.',
]
PASSAGES += [
    'Minutes, product sync, 12 May: Priya presented the new onboarding flow; drop-off fell from 41% to 29% in the beta. Tom will finish the Android build by 20 May. Open question: whether to keep the email verification step. Next meeting 19 May, 10:00.',
    'The Harbor Street bakery sells sourdough loaves for $6.50, baguettes for $3.25 and cinnamon rolls for $2.75. It opens at 7:00 on weekdays and 8:00 on weekends, closes on Mondays, and takes orders of more than 20 items with two days notice.',
    'Warranty terms: the blender is covered for 24 months from purchase against manufacturing defects. Damage from dropping, using it without liquid, or commercial use is excluded. Claims need the receipt and serial number, and shipping to the repair center is paid by the customer.',
    'Volunteer roster for the river cleanup on 7 September: 23 people signed up, 4 of them team leads (Alba, Kenji, Ruth, Omar). Meeting point is the boat ramp at 8:30. Gloves and bags are provided; volunteers should bring water and sun protection.',
]
EXTRACT_FORMS = ['Extract every concrete fact from the text below into a JSON object with sensible keys. Output JSON only.\n\n{p}',
                 'Read the passage and return a JSON object with the fields you can fill in confidently; use null for anything missing.\n\n{p}',
                 'Convert this text into structured data. Respond with YAML only.\n\n{p}',
                 'Summarize the passage below as a Markdown table with columns "field" and "value".\n\n{p}',
                 'List every named person, date, amount and location in this text as a JSON object with those four arrays.\n\n{p}',
                 'Write a JSON Schema that would validate structured records of the kind described here, then the record for this text.\n\n{p}']

CONFIG_TASKS = [
    'a docker-compose.yml for a web app with Postgres, Redis and a background worker',
    'a GitHub Actions workflow that lints, tests on two Python versions and uploads coverage',
    'a Kubernetes Deployment and Service for a stateless HTTP API with readiness probes',
    'an nginx server block that serves static files and proxies /api to a local upstream',
    'a pyproject.toml for a small library with optional dev dependencies',
    'a Cargo.toml for a CLI with clap and serde, plus a release profile tuned for size',
    'a systemd unit for a long-running service with restart limits',
    'a Prometheus alerting rule for sustained high error rate',
    'an OpenAPI 3 fragment describing a paginated GET /items endpoint',
    'a .editorconfig and a Prettier config for a TypeScript monorepo',
    'a Terraform resource block for a private object-storage bucket with versioning',
    'a JSON Schema for a user profile with required email and optional phone',
]
CONFIG_FORMS = ['Write {task}. Output only the file contents.',
                'Create {task}, then explain the two most important settings.',
                'Give me {task} with brief inline comments.',
                'Produce {task}. After the file, list anything that must be changed before production use.']

TOOLS = {
    'get_weather': {'description': 'Get the current weather for a city.',
                    'parameters': {'type': 'object', 'properties': {'city': {'type': 'string'}, 'unit': {'type': 'string', 'enum': ['celsius', 'fahrenheit']}}, 'required': ['city']}},
    'search_docs': {'description': 'Search the internal documentation.',
                    'parameters': {'type': 'object', 'properties': {'query': {'type': 'string'}, 'limit': {'type': 'integer'}}, 'required': ['query']}},
    'create_ticket': {'description': 'Create an issue in the tracker.',
                      'parameters': {'type': 'object', 'properties': {'title': {'type': 'string'}, 'body': {'type': 'string'}, 'priority': {'type': 'string', 'enum': ['low', 'medium', 'high']}}, 'required': ['title', 'body']}},
    'run_sql': {'description': 'Run a read-only SQL query against the analytics database.',
                'parameters': {'type': 'object', 'properties': {'query': {'type': 'string'}}, 'required': ['query']}},
    'read_file': {'description': 'Read a file from the repository.',
                  'parameters': {'type': 'object', 'properties': {'path': {'type': 'string'}}, 'required': ['path']}},
    'list_directory': {'description': 'List files in a repository directory.',
                       'parameters': {'type': 'object', 'properties': {'path': {'type': 'string'}}, 'required': ['path']}},
    'calculator': {'description': 'Evaluate an arithmetic expression exactly.',
                   'parameters': {'type': 'object', 'properties': {'expression': {'type': 'string'}}, 'required': ['expression']}},
    'set_reminder': {'description': 'Schedule a reminder for the user.',
                     'parameters': {'type': 'object', 'properties': {'text': {'type': 'string'}, 'when': {'type': 'string', 'description': 'ISO-8601 time'}}, 'required': ['text', 'when']}},
    'translate': {'description': 'Translate text to a target language.',
                  'parameters': {'type': 'object', 'properties': {'text': {'type': 'string'}, 'target_language': {'type': 'string'}}, 'required': ['text', 'target_language']}},
    'convert_units': {'description': 'Convert a quantity between units.',
                      'parameters': {'type': 'object', 'properties': {'value': {'type': 'number'}, 'from_unit': {'type': 'string'}, 'to_unit': {'type': 'string'}}, 'required': ['value', 'from_unit', 'to_unit']}},
}
TOOL_REQUESTS = [
    (['get_weather'], 'Is it going to be colder in Oslo or in Reykjavik right now? Use celsius.'),
    (['get_weather', 'set_reminder'], 'Check the weather in Lisbon and, if it is raining, remind me tomorrow at 08:00 to take an umbrella.'),
    (['search_docs'], 'How do I rotate the signing keys for the internal auth service? Look it up.'),
    (['search_docs', 'create_ticket'], 'Find out whether our docs mention a limit on upload size, and if they do not, open a medium-priority ticket asking for it to be documented.'),
    (['run_sql'], 'How many orders were placed last week, broken down by day? The table is orders(id, created_at, amount).'),
    (['run_sql', 'calculator'], 'Get total revenue for March and April from orders(id, created_at, amount) and tell me the percentage change.'),
    (['list_directory', 'read_file'], 'Find where the retry policy is configured in this repository and show me its current settings.'),
    (['read_file'], 'Open src/config/defaults.json and tell me which values look unsafe for production.'),
    (['calculator'], 'What is 17.5% of 2,348.20, plus a flat 12.75 fee?'),
    (['set_reminder'], 'Remind me to submit the expense report next Friday at 16:30.'),
    (['translate'], 'Translate "The meeting has been moved to Thursday afternoon" into German and Japanese.'),
    (['convert_units'], 'My recipe needs 350 grams of flour and 2 cups of milk. Convert both to ounces.'),
    (['create_ticket'], 'The checkout page shows a blank screen on Safari 17 after applying a coupon. File this as a high-priority bug with reproduction steps.'),
    (['get_weather', 'convert_units'], 'What is the temperature in Phoenix in fahrenheit, and what is that in kelvin?'),
    (['list_directory'], 'What test files exist under tests/integration?'),
    (['search_docs', 'read_file'], 'Our docs describe a feature flag file. Find which file it is and read it.'),
]
TOOL_REQUESTS += [
    (['get_weather'], 'Should I bring a jacket to Edinburgh this evening?'),
    (['run_sql'], 'Which five products had the most returns this quarter? Tables: returns(product_id, returned_at), products(id, name).'),
    (['search_docs'], 'What is our policy on deleting customer data after account closure?'),
    (['create_ticket', 'search_docs'], 'The nightly export job failed again with a timeout. Check the runbook and open a ticket with what you find.'),
    (['read_file', 'list_directory'], 'Show me the Dockerfile used for the worker image.'),
    (['calculator', 'convert_units'], 'I drove 412 miles on 11.3 gallons. What is that in liters per 100 km?'),
    (['translate', 'set_reminder'], 'Translate "Call the plumber" into Spanish and remind me with that text tomorrow at 9.'),
    (['get_weather', 'search_docs'], 'Our field team in Denver follows a heat policy. What is the temperature there and does the policy apply today?'),
    (['run_sql', 'create_ticket'], 'If any user signed up today without a verified email, open a low-priority ticket listing how many. Table: users(id, created_at, email_verified).'),
    (['list_directory'], 'Which configuration files are in the deploy/ directory?'),
    (['calculator'], 'Split a $187.40 dinner bill between 6 people with an 18% tip.'),
    (['set_reminder', 'get_weather'], 'Remind me at 6pm to water the garden unless it is supposed to rain in Austin.'),
    (['search_docs'], 'Find the steps for requesting production database access.'),
    (['read_file'], 'Read CHANGELOG.md and tell me what changed in the latest release.'),
    (['convert_units'], 'How many kilometers is a marathon, and how many feet?'),
    (['translate'], 'Give me the French and Italian for "Where is the nearest pharmacy?"'),
]
TOOL_SYSTEMS = [None, 'You are a careful assistant. Call tools when they are needed and never invent results.',
                'You are an operations assistant with access to tools. Prefer a single well-formed call when possible.']

JSON_INPUTS = [
    '{"users": [{"id": 1, "name": "Ada", "roles": ["admin", "dev"]}, {"id": 2, "name": "Lin", "roles": ["dev"]}, {"id": 3, "name": "Sam", "roles": []}]}',
    '{"order": {"id": "A-9", "lines": [{"sku": "X1", "qty": 2, "price": 3.5}, {"sku": "Y7", "qty": 1, "price": 12.0}], "coupon": null}}',
    '[{"city": "Lyon", "temp": 21}, {"city": "Porto", "temp": 24}, {"city": "Lyon", "temp": 19}, {"city": "Graz", "temp": 17}]',
    '{"config": {"server": {"port": 8080, "host": "0.0.0.0"}, "db": {"url": "postgres://db/app", "pool": 10}, "debug": true}}',
    '{"events": [{"t": "2026-03-01T10:00:00Z", "type": "login", "user": "u1"}, {"t": "2026-03-01T10:05:00Z", "type": "purchase", "user": "u1", "amount": 30}, {"t": "2026-03-01T11:00:00Z", "type": "login", "user": "u2"}]}',
    '{"repo": "toolkit", "issues": [{"n": 12, "labels": ["bug"], "open": true}, {"n": 15, "labels": ["docs"], "open": false}, {"n": 18, "labels": ["bug", "ui"], "open": true}]}',
    '{"invoice": {"number": 1042, "items": [{"desc": "Design", "hours": 6, "rate": 80}, {"desc": "Review", "hours": 2, "rate": 60}], "tax_rate": 0.2}}',
    '{"library": [{"title": "Dune", "year": 1965, "genres": ["sf"]}, {"title": "Emma", "year": 1815, "genres": ["romance", "classic"]}]}',
]
JSON_TRANSFORMS = [
    'Flatten this JSON into dotted-path keys and output the result as JSON only:',
    'Group the records in this JSON by a sensible key and output the grouped JSON only:',
    'Validate this JSON against what you infer the intended schema to be, and report problems as a JSON list:',
    'Rewrite this JSON with camelCase keys and computed totals where they make sense. Output JSON only:',
    'Convert this JSON to a CSV table with a header row, then explain any lossy choices in one sentence:',
]
TABLE_TASKS = [
    'Make a Markdown table comparing three common sorting algorithms by average case, worst case, memory and stability.',
    'Create a CSV with ten fictional employees: name, department, start year and salary band.',
    'Produce a Markdown table of the planets of the solar system with diameter, number of known moons and distance from the Sun.',
    'Write a Markdown table summarizing HTTP status code classes with an example of each.',
    'Create a weekly meal plan as a Markdown table with columns for each day and rows for breakfast, lunch and dinner.',
    'Make a CSV of twelve months with average temperature for a fictional temperate city.',
    'Build a Markdown comparison table of three note-taking approaches: paper, plain text files and a database app.',
    'Make a Markdown table of common Git commands with what they do and one caution each.',
    'Create a CSV inventory of a small coffee shop: item, unit, quantity on hand and reorder level.',
    'Produce a Markdown table of five programming languages with typing discipline, memory management and a typical use.',
    'Create a JSON array describing the steps of making tea, each with "step", "duration_seconds" and "tip".',
    'Make a Markdown table of keyboard shortcuts for a text editor, grouped by navigation, editing and search.',
]

PROSE_TOPICS = [
    'how vaccines train the immune system', 'why the sky is blue and sunsets are red', 'how a refrigerator moves heat',
    'how public-key cryptography lets strangers communicate securely', 'what inflation is and why central banks target it',
    'how a bill becomes law in a parliamentary system', 'how photosynthesis stores energy', 'why compound interest matters for savers',
    'how garbage collection works in managed languages', 'how GPS determines your position', 'what causes the seasons',
    'how a compiler turns source code into machine code', 'why bridges sway and how engineers prevent collapse',
    'how plate tectonics shapes continents', 'what a database index is and when it hurts', 'how noise-cancelling headphones work',
    'why the printing press changed Europe', 'how the human eye perceives color', 'how an electric motor turns current into motion',
    'what happens during a solar eclipse', 'how a search engine ranks pages', 'why antibiotic resistance spreads',
    'how a jet engine produces thrust', 'how DNS resolves a domain name', 'how a sourdough starter works',
    'why some materials conduct electricity and others do not', 'how supply chains respond to a sudden shortage',
    'how the Roman aqueducts carried water', 'what a neural network learns during training', 'how tides are caused by the Moon',
    'how virtual memory gives each process its own address space', 'how a vaccine trial is designed', 'why coral reefs bleach',
    'how containers differ from virtual machines', 'how the stock of groundwater is replenished', 'how a credit score is computed in principle',
    'why the Industrial Revolution began in Britain', 'how sound travels through different materials', 'how a lithium-ion battery charges and discharges',
    'how version control systems track changes', 'why sleep matters for memory', 'how a hurricane forms and strengthens',
    'how load balancers distribute traffic', 'what entropy means in thermodynamics', 'how bees communicate the location of food',
    'how a camera sensor captures an image', 'why languages change over time', 'how a nuclear reactor controls its chain reaction',
    'how a spreadsheet recalculates dependent cells', 'what makes a good password and why', 'how the immune system can attack the body itself',
]
PROSE_FORMS = [
    'Explain {topic} to a curious teenager in a few paragraphs.',
    'Write a clear, engaging explanation of {topic} for a general audience, using one concrete analogy.',
    'In continuous paragraphs rather than bullet points, describe {topic}.',
    'Give a concise but thorough explanation of {topic}, then mention one common misconception.',
    'Write a short magazine-style piece about {topic}.',
    'Teach {topic} step by step, as a patient tutor would, checking understanding along the way.',
    'Write an explanation of {topic} suitable for an encyclopedia entry, then add a brief historical note.',
]
WRITING_TASKS = [
    'Write a polite email to a landlord asking to fix a broken heater before the weekend.',
    'Write a short story about a lighthouse keeper who receives an unexpected letter.',
    'Draft a product description for a waterproof hiking backpack aimed at weekend travelers.',
    'Write a cover letter opening paragraph for a junior software developer position at a small nonprofit.',
    'Write a 300-word opinion piece arguing that cities should plant more street trees.',
    'Write a dialogue between a nervous student and a patient chemistry teacher before an exam.',
    'Write a heartfelt thank-you note to a mentor who is retiring.',
    'Summarize the pros and cons of working remotely in two balanced paragraphs.',
    'Write a short fable about a fox and a crow that teaches a lesson about patience.',
    'Write the opening scene of a mystery novel set on a night train.',
    'Draft a friendly announcement telling a team that the office will close early on Friday.',
    'Write a review of an imaginary neighborhood cafe, describing the food, the service and the atmosphere.',
    'Write a speech for a best friend at a small wedding, warm and a little funny.',
    'Write a letter from a time traveler to their younger self, three paragraphs long.',
    'Explain to a new employee how to give constructive feedback, in an encouraging tone.',
    'Write a short travel diary entry about arriving in a mountain village in winter.',
    'Write a persuasive paragraph encouraging people to learn a second language.',
    'Write a bedtime story about a small robot who is afraid of the dark.',
    'Write an apology email to a customer whose order arrived two weeks late.',
    'Describe a thunderstorm over the ocean in vivid sensory detail.',
    'Write a short biography of a fictional inventor who built a solar-powered bicycle.',
    'Write a newsletter intro for a community garden announcing the spring planting day.',
    'Compose a reflective essay on what makes a place feel like home.',
    'Write a scene in which two old friends meet again after twenty years.',
    'Write instructions, in friendly prose, for caring for a new houseplant.',
]
LANGS_NAT = ['French', 'Spanish', 'German', 'Portuguese', 'Italian', 'Japanese', 'Chinese', 'Arabic']
MULTILINGUAL = ['Write a short paragraph in {lang} describing your favorite season.',
                'Translate this sentence into {lang} and explain any idiom choices: "The early bird catches the worm, but the second mouse gets the cheese."',
                'Write a brief, polite message in {lang} asking a hotel for a late checkout.']

MATH = [
    'A train leaves at 9:40 and travels 210 km at 84 km/h. When does it arrive? Show your reasoning.',
    'If 3 workers paint a fence in 8 hours, how long do 5 workers take at the same rate? Explain.',
    'A shop discounts a jacket by 20% and then by another 15%. What is the total discount? Explain why it is not 35%.',
    'Prove that the sum of the first n odd numbers is n squared.',
    'Two dice are rolled. What is the probability that the sum is 8? Show the sample space reasoning.',
    'Solve for x: 3(x - 4) + 2 = 5x - 10. Show each step.',
    'A rectangle has a perimeter of 46 cm and an area of 120 square cm. Find its sides.',
    'How many ways can 5 books be arranged on a shelf if two specific books must be next to each other?',
    'Explain why dividing by zero is undefined, using a short algebraic argument.',
    'A population grows 4% per year. Roughly how many years until it doubles? Explain the rule of 70.',
    'Convert 0.375 to a fraction in lowest terms and explain the method.',
    'Find the greatest common divisor of 462 and 1071 with the Euclidean algorithm, showing the steps.',
    'A bag has 4 red and 6 blue marbles. Two are drawn without replacement. What is the chance both are red?',
    'Show that the square root of 2 is irrational.',
    'A car depreciates 12% per year. What is a $28,000 car worth after 3 years?',
    'What is the expected value of a game where you win $5 with probability 0.2 and lose $1 otherwise?',
    'Explain the difference between permutations and combinations with a small example of each.',
    'Estimate how many piano tuners work in a city of one million people, explaining each assumption.',
    'If a clock shows 3:15, what is the angle between the hour and minute hands?',
    'Sum the series 1 + 1/2 + 1/4 + 1/8 + ... and explain why it converges.',
]


def templated(rng: random.Random):
    rows = []
    # Code generation (~720): sample distinct (lang, task, form) triples.
    combos = [(l, t, f) for l in LANGS for t in CODE_TASKS for f in CODE_FORMS]
    rng.shuffle(combos)
    for i, (lang, task, form) in enumerate(combos[:580]):
        text = rng.choice(CODE_OPENERS).format(task=task, lang=lang) + ' ' + form
        rows.append(row(f'own-code-{i:04d}', 'code', 'template/code', OWN, user(text),
                        thinking=rng.random() < 0.12))
    # Bug fixing (~36).
    i = 0
    for lang, code, symptom in BUGGY:
        for form in BUG_FORMS:
            text = f'This {lang} code has a problem: {symptom}.\n\n```\n{code}\n```\n\n{form}'
            rows.append(row(f'own-bug-{i:03d}', 'code', 'template/bugfix', OWN, user(text),
                            thinking=rng.random() < 0.2))
            i += 1
    # SQL (~60).
    i = 0
    for schema, asks in SQL_SCHEMAS:
        for ask in asks:
            for dialect in SQL_DIALECTS:
                text = f'Tables: {schema}. Write a {dialect} query for {ask}. Explain any tricky part briefly.'
                rows.append(row(f'own-sql-{i:03d}', 'code', 'template/sql', OWN, user(text)))
                i += 1
    # JSON with a schema (~45), extraction (~32), configs (~36).
    i = 0
    for entity, shape in JSON_ENTITIES:
        for form in JSON_FORMS:
            rows.append(row(f'own-json-{i:03d}', 'structured', 'template/json', OWN,
                            user(form.format(entity=entity, shape=shape)), max_tokens=768))
            i += 1
    i = 0
    for p in PASSAGES:
        for form in EXTRACT_FORMS:
            rows.append(row(f'own-extract-{i:03d}', 'structured', 'template/extraction', OWN,
                            user(form.format(p=p)), max_tokens=512))
            i += 1
    i = 0
    for task in CONFIG_TASKS:
        for form in CONFIG_FORMS:
            rows.append(row(f'own-config-{i:03d}', 'structured', 'template/config', OWN,
                            user(form.format(task=task)), max_tokens=768))
            i += 1
    i = 0
    for doc in JSON_INPUTS:
        for form in JSON_TRANSFORMS:
            rows.append(row(f'own-jsonxf-{i:03d}', 'structured', 'template/json-transform', OWN,
                            user(form + '\n\n' + doc), max_tokens=768))
            i += 1
    for i, task in enumerate(TABLE_TASKS):
        rows.append(row(f'own-table-{i:03d}', 'structured', 'template/table', OWN, user(task), max_tokens=768))
        rows.append(row(f'own-table-b{i:03d}', 'structured', 'template/table', OWN,
                        user(task + ' Then add two sentences interpreting it.'), max_tokens=768))
    # Tool calls with real `tools` definitions (~48), each request x system prompt.
    i = 0
    for names, request in TOOL_REQUESTS:
        for system in TOOL_SYSTEMS:
            extra = sorted(set(rng.sample(sorted(TOOLS), 2)) | set(names))
            tools = [{'type': 'function', 'function': {'name': n, **TOOLS[n]}} for n in extra]
            msgs = ([{'role': 'system', 'content': system}] if system else []) + user(request)
            rows.append(row(f'own-tool-{i:03d}', 'tool-call', 'template/tool-call', OWN, msgs,
                            tools=tools, max_tokens=512))
            i += 1
    # Tool-call follow-ups: a completed tool result the model must use (~16).
    for j, (names, request) in enumerate(TOOL_REQUESTS):
        name = names[0]
        args = {'get_weather': {'city': 'Oslo'}, 'search_docs': {'query': request[:40]}, 'run_sql': {'query': 'SELECT 1'},
                'list_directory': {'path': '.'}, 'read_file': {'path': 'README.md'}, 'calculator': {'expression': '1+1'},
                'set_reminder': {'text': 'reminder', 'when': '2026-10-16T08:00:00'}, 'translate': {'text': 'hello', 'target_language': 'German'},
                'convert_units': {'value': 1, 'from_unit': 'g', 'to_unit': 'oz'}, 'create_ticket': {'title': 'bug', 'body': 'details'}}[name]
        result = {'get_weather': '{"city": "Oslo", "temp_c": 4.5, "conditions": "light rain"}',
                  'search_docs': '{"results": [{"title": "Operations handbook", "snippet": "See section 4.2 for the procedure."}]}',
                  'run_sql': '{"columns": ["day", "orders"], "rows": [["Mon", 41], ["Tue", 37], ["Wed", 52]]}',
                  'list_directory': '{"entries": ["src/", "tests/", "README.md", "pyproject.toml"]}',
                  'read_file': '{"content": "# Project\\nRun `make test` before committing."}',
                  'calculator': '{"value": "423.69"}', 'set_reminder': '{"status": "scheduled"}',
                  'translate': '{"translation": "Die Besprechung wurde auf Donnerstagnachmittag verschoben."}',
                  'convert_units': '{"value": 12.35, "unit": "oz"}', 'create_ticket': '{"id": "BUG-4821", "status": "open"}'}[name]
        tools = [{'type': 'function', 'function': {'name': n, **TOOLS[n]}} for n in names]
        msgs = user(request) + [
            {'role': 'assistant', 'content': '', 'tool_calls': [{'id': 'call_0', 'type': 'function',
             'function': {'name': name, 'arguments': json.dumps(args)}}]},
            {'role': 'tool', 'tool_call_id': 'call_0', 'content': result}]
        rows.append(row(f'own-toolresult-{j:03d}', 'tool-call', 'template/tool-result', OWN, msgs,
                        tools=tools, max_tokens=512))
    # Prose explanations (~250), writing (~100), multilingual (~24), math (~40).
    combos = [(t, f) for t in PROSE_TOPICS for f in PROSE_FORMS]
    rng.shuffle(combos)
    for i, (topic, form) in enumerate(combos[:350]):
        rows.append(row(f'own-explain-{i:03d}', 'prose', 'template/explain', OWN,
                        user(form.format(topic=topic)), thinking=rng.random() < 0.1))
    i = 0
    for task in WRITING_TASKS:
        for variant in ['', ' Keep it under 250 words.', ' Use a warm, informal tone.', ' Make it vivid and specific.']:
            rows.append(row(f'own-write-{i:03d}', 'prose', 'template/writing', OWN, user(task + variant)))
            i += 1
    i = 0
    for lang in LANGS_NAT:
        for form in MULTILINGUAL:
            rows.append(row(f'own-multi-{i:03d}', 'prose', 'template/multilingual', OWN,
                            user(form.format(lang=lang)), max_tokens=512))
            i += 1
    for i, q in enumerate(MATH):
        rows.append(row(f'own-math-{i:03d}', 'prose', 'template/math', OWN, user(q), max_tokens=768))
        rows.append(row(f'own-maththink-{i:03d}', 'prose', 'template/math', OWN, user(q), thinking=True,
                        max_tokens=1536))
    return rows

def build() -> tuple[list[dict], list[dict]]:
    kept, dropped, seen = [], [], set()
    for r in templated(random.Random(SEED)):
        text = '\n'.join(m.get('content') or '' for m in r['messages'])
        why = excluded(text)
        key = hashlib.sha256(json.dumps([r['messages'], r.get('tools'), r['thinking']]).encode()).hexdigest()
        if why:
            dropped.append({'id': r['id'], 'reason': why})
        elif key in seen:
            dropped.append({'id': r['id'], 'reason': 'duplicate'})
        else:
            seen.add(key)
            r.update(id=f"template-{r['id']}", corpus_source=r['source'], source='template', split='train',
                     reasoning_effort=None, temperature=1.0, sha256=key, prompt_sha=key[:16])
            kept.append(r)
    assert len({r['id'] for r in kept}) == len(kept)
    return kept, dropped


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', type=Path, required=True, help='output directory (train.jsonl, manifest.json)')
    a = ap.parse_args(argv)
    kept, dropped = build()
    data = ''.join(json.dumps(r, ensure_ascii=False, sort_keys=True) + '\n' for r in kept).encode()
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / 'train.jsonl').write_bytes(data)
    by = lambda k: dict(sorted(Counter(r[k] for r in kept).items()))  # noqa: E731
    manifest = {
        'schema': 'enntity.drafter.template-prompts/v1', 'seed': SEED, 'rows': len(kept), 'jsonl': 'train.jsonl',
        'jsonl_sha256': hashlib.sha256(data).hexdigest(), 'by_category': by('category'),
        'by_corpus_source': by('corpus_source'), 'license': OWN,
        'thinking_rows': sum(r['thinking'] for r in kept), 'tool_rows': sum('tools' in r for r in kept),
        'exclusions': {'matched_benchmark_prompts': EXCLUDED_EXACT, 'topic_regex': EXCLUDED_TOPIC.pattern,
                       'dropped': dropped},
    }
    (a.out / 'manifest.json').write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + '\n')
    print(json.dumps({k: manifest[k] for k in ('rows', 'by_category', 'thinking_rows', 'tool_rows')}))


if __name__ == '__main__':
    main()
