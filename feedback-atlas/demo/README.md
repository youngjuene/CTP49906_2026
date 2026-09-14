# Demo corpus

`demo_opinions.csv` supplies 500 synthetic AI opinions for five projects,
with 100 opinions per project and 125 per week (weeks 1–4).
`roster.csv` maps the five target IDs to project titles and the ten reviewer
IDs to display labels. Reviewers use the observer role so only the five
projects appear as feedback targets; the shared `demo` account remains available.

The running demo reads `.run/demo/atlas.db`, not the CSV directly. The September
10 replacement preserves CSV reviewer IDs, text, source, and week, and uses
`draft_key` as the stored opinion ID. Extra provenance columns remain in the CSV.
The generic `seed_ai_opinions.py` importer uses a single reviewer argument and
appends records, so it must not be used to repeat this replacement unchanged.

The previous 30-opinion database and roster were backed up at
`.run/demo/backup-20260910T012127Z/`. The classroom database is independent.
Restart the service after changing demo data; SIGUSR1 reloads only the classroom.

At 500 opinions the current demo capacity is full. Browsing, filtering,
neighbors, and admin export remain available; additional submissions receive
the existing demo-full message.
