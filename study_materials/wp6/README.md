# Classroom release guidance

**Content version:** `classroom-guidance/1.0.0`  
**Release status:** candidate teaching materials; **not research-ready**

This directory is the English/Korean operational companion for the classroom
notebooks. It documents a safe Teaching-mode route, the blinded audience
handoff, replay identity, privacy boundaries, accessibility checks, and recovery.
It does not supply ethics approval, instrument validation, audience recruitment
evidence, WCAG conformance, or a research release decision.

## Documents

| Purpose | English | 한국어 |
|---|---|---|
| Instructor preparation and class sequence | [Instructor runbook](instructor_runbook.en.md) | [교수자 운영 안내](instructor_runbook.ko.md) |
| Short student route | [Student quick-start](student_quick_start.en.md) | [학생 빠른 시작](student_quick_start.ko.md) |
| Moving from the legacy worksheet | [Worksheet migration](worksheet_migration.en.md) | [워크시트 전환 안내](worksheet_migration.ko.md) |
| Data fields, permissions, and egress | [Privacy/data dictionary](privacy_data_dictionary.en.md) | [개인정보·데이터 사전](privacy_data_dictionary.ko.md) |
| Common recovery paths | [Troubleshooting](troubleshooting.en.md) | [문제 해결](troubleshooting.ko.md) |

Machine-readable companions:

- [content_manifest.json](content_manifest.json) pairs English and Korean files by
  stable document ID.
- [replay_manifest.json](replay_manifest.json) records the immutable model
  revision, coded stimulus checksum, generation command, and replay file hashes.
- [visual_alternatives.json](visual_alternatives.json) pairs required visuals
  with English/Korean text alternatives and checksum-bound data downloads.
- [audience_reading_matrix_example.csv](audience_reading_matrix_example.csv) is
  a synthetic, non-student example of the post-reveal disagreement table.

## Release boundary

Teaching mode is the default. Missing human approvals keep Research mode and
research export unavailable; they do not prevent a safe teaching rehearsal.
The software and these documents are **not research-ready** until the named
human governance, accessibility/localization, audience-validity, instrument,
comparison-fidelity, and release gates are approved and version-locked.

The notebooks do not automatically send student artifacts, process records, or
audience responses to an instructor or external service. The three authorized
student-data egress classes remain separate: a user-initiated private download,
a permission-valid blinded audience packet, and an approved minimized Research
projection. Only the first two are available in the teaching route.
