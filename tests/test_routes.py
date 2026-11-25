"""Flask routes for the planner and the Adaptive Classroom Assistant (offline, AI disabled)."""
from __future__ import annotations

from adaptive import storage


def test_dashboard_planner_and_prefill(client):
    r = client.get("/")
    assert r.status_code == 200 and b"Welcome back" in r.data and b"Lessons generated" in r.data
    r = client.get("/planner")
    assert r.status_code == 200 and b"Worksheet Accessibility Settings" in r.data
    r = client.get("/planner?subject=Maths&duration=40&grades=4&attention_hint=Low")
    assert b'name="attention_hint" value="Low"' in r.data and b'value="40"' in r.data
    # Old links to "/?subject=..." still reach the pre-filled planner.
    r = client.get("/?subject=Maths&duration=40&grades=4")
    assert r.status_code == 302 and r.headers["Location"].startswith("/planner?")


def test_invalid_lesson_form_shows_errors(client):
    r = client.post("/generate", data={"topic": "", "subject": "Maths", "duration": "5"})
    assert r.status_code == 400 and b"Please fix the following" in r.data


def test_assistant_full_flow(client):
    assert client.get("/assistant/").status_code == 200
    r = client.post("/assistant/new", data={"grade": "3", "section": "C"})
    assert r.status_code == 302
    cid = storage.list_profiles()[0].id
    assert client.get(f"/assistant/?class={cid}&tab=setup").status_code == 200

    client.post(f"/assistant/{cid}/generate", data={"mode": "first"})
    p = storage.load_profile(cid)
    assert p.proposal and p.current is None
    page = client.get(f"/assistant/?class={cid}&tab=timetable").data
    assert b"Why this timetable?" in page and b"Approve" in page

    client.post(f"/assistant/{cid}/approve")
    p = storage.load_profile(cid)
    assert p.current and p.proposal is None
    page = client.get(f"/assistant/?class={cid}&tab=timetable").data
    assert b"Plan lesson for this period" in page and b"/planner?subject=Maths" in page

    r = client.post(f"/assistant/{cid}/edit", data={"target": "current", "action": "swap",
                                                    "cell_a": "Mon-1", "cell_b": "Tue-1"}, follow_redirects=True)
    assert b"Change saved" in r.data
    r = client.post(f"/assistant/{cid}/edit", data={"target": "current", "action": "swap",
                                                    "cell_a": "Mon-1", "cell_b": ""}, follow_redirects=True)
    assert b"Pick a period" in r.data

    client.post(f"/assistant/{cid}/generate", data={"mode": "regenerate"})
    client.post(f"/assistant/{cid}/reject")
    assert storage.load_profile(cid).proposal is None

    r = client.post(f"/assistant/{cid}/feedback", data={"day": "Mon", "period": "1", "subject": "Mathematics",
                                                        "understanding": "5", "engagement": "4", "concentration": "4"})
    assert r.status_code == 302 and len(storage.load_profile(cid).feedback) == 1
    r = client.get(f"/assistant/{cid}/pdf")
    assert r.status_code == 200 and r.mimetype == "application/pdf"

    bad = client.post(f"/assistant/{cid}/setup", data={"grade": "3", "start_time": "x"})
    assert bad.status_code == 400 and b"Your changes have not been saved yet" in bad.data
    client.post(f"/assistant/{cid}/delete")
    assert storage.load_profile(cid) is None


def test_unknown_class_404(client):
    assert client.get("/assistant/abcdefabcdef/pdf").status_code == 404
