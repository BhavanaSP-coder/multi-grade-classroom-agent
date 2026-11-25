"""AI Adaptive Classroom Assistant: attention-aware subject timetables for one class.

Modules
-------
models     Pydantic models (class profile, subjects, attention windows, feedback, timetable)
timing     period / break clock times and attention per period
learning   turns teacher feedback into adjusted attention + subject/time affinities
scheduler  deterministic, scored optimisation (greedy + local search, ranked candidates)
validator  hard-constraint and quality checks (the final word on validity)
explain    explanations derived from the actual placements
ai         Groq advice + wording (checked against the timetable before display)
service    orchestration: generate proposals, approve/reject, edits, feedback
storage    JSON persistence in data/adaptive/
forms      parsing + validation of the dashboard forms
pdf        printable timetable
"""
