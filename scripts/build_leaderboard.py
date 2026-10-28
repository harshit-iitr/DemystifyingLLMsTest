#!/usr/bin/env python3
"""
Leaderboard Builder — Aggregates autograder scores from PR comments.

Extracts participant details (Name, Roll No, IITR Email, Contact) from the
PR description template, scores from CI autograder comments, and renders:
  1. LEADERBOARD.md (formatted markdown table with prize indicators)
  2. LEADERBOARD.html (gorgeous dark-mode live dashboard with 15s auto-refresh)
  3. leaderboard.json (structured data for prizes and verification)
  4. Live terminal summary table (with optional --watch mode)

Features:
  - Zero external dependencies (uses standard library urllib & json).
  - Works with or without GITHUB_TOKEN (unauthenticated for test runs,
    token-authenticated for live 200+ student events).
  - Deduplicates multiple PRs per student (keeps best score).
  - Excludes test accounts automatically (--exclude-users).
  - Live projection dashboard (LEADERBOARD.html) ready for auditorium screens.
  - Continuous loop mode (--watch <seconds>).

Usage:
  # 1. Single run:
  python scripts/build_leaderboard.py --token ghp_yourTokenHere

  # 2. Continuous live display for auditorium (refreshes every 15s):
  python scripts/build_leaderboard.py --token ghp_yourTokenHere --watch 15

  # 3. Quick test run without token:
  python scripts/build_leaderboard.py
"""

from __future__ import annotations

import os
import sys
import re
import time
import json
import argparse
import urllib.request
import urllib.error
from datetime import datetime
from pathlib import Path

# Enforce UTF-8 stdout on Windows terminals
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REPO = "harshit-iitr/DemystifyingLLMsTest"
DEFAULT_CUTOFF = 100
DEFAULT_EXCLUDED = ""


def github_api_get(url: str, token: str = None) -> list | dict:
    """Perform a GET request to GitHub REST API with rate-limit handling."""
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "LeaderboardBuilder/3.0",
    }
    if token and token.strip():
        headers["Authorization"] = f"token {token.strip()}"

    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        if e.code == 403 and "rate limit" in body.lower():
            print("\n[ERROR] GitHub API rate limit reached (60 requests/hour unauthenticated).")
            print("To remove this limit (5,000 requests/hour):")
            print("  Pass token: python scripts/build_leaderboard.py --token <YOUR_TOKEN>")
            print("  or set: $env:GITHUB_TOKEN=\"<YOUR_TOKEN>\"\n")
            return []
        elif e.code == 401:
            print("\n[ERROR] HTTP 401 Unauthorized. Please verify your GITHUB_TOKEN.")
            return []
        else:
            print(f"\n[WARNING] HTTP {e.code} for {url}: {body[:200]}")
            return []
    except urllib.error.URLError as e:
        print(f"\n[ERROR] Network error: {e.reason}")
        return []


def get_all_prs(repo: str, token: str = None, state: str = "open") -> list:
    """Fetch all PRs (paginated)."""
    prs = []
    page = 1
    while True:
        url = f"https://api.github.com/repos/{repo}/pulls?state={state}&per_page=100&page={page}"
        data = github_api_get(url, token)
        if not data or not isinstance(data, list):
            break
        prs.extend(data)
        if len(data) < 100:
            break
        page += 1
    return prs


def get_latest_score(repo: str, pr_number: int, token: str = None):
    """Scan PR comments for the most recent Autograder score report."""
    url = f"https://api.github.com/repos/{repo}/issues/{pr_number}/comments?per_page=100"
    try:
        comments = github_api_get(url, token)
    except Exception:
        return 0, 0, None

    if not isinstance(comments, list):
        return 0, 0, None

    for comment in reversed(comments):
        body = comment.get("body", "")
        if "Autograder" in body or "Score:" in body:
            match = re.search(r"Score:\s*(\d+)\s*/\s*(\d+)", body)
            tests_match = re.search(r"Tests Passed:\s*(\d+)\s*/\s*(\d+)", body)
            score = int(match.group(1)) if match else 0
            tests_passed = int(tests_match.group(1)) if tests_match else 0
            created_at = comment.get("created_at")
            if match:
                return score, tests_passed, created_at
    return 0, 0, None


def extract_participant_info(pr_body: str) -> dict:
    """Extract Name, Roll No, IITR Email, Contact from PR description."""
    info = {"name": "—", "roll_no": "—", "email": "—", "contact": "—"}
    if not pr_body:
        return info

    field_map = {
        "Full Name": "name",
        "Enrollment / Roll No": "roll_no",
        "IITR Email": "email",
        "Discord / Phone": "contact",
    }

    placeholders = [
        "jane doe",
        "john doe",
        "harshit anand",
        "221150xx",
        "2311xxxx",
        "your_id@mfs.iitr.ac.in",
        "student_id@mfs.iitr.ac.in",
        "harshit#1234",
        "discord_handle",
        "+91-9876543210",
        "+91-xxxxxxxxxx",
    ]

    for line in pr_body.splitlines():
        line = line.strip()
        for label, key in field_map.items():
            pattern = rf"^-\s*\*\*{re.escape(label)}:\*\*(.*)$"
            match = re.match(pattern, line, re.IGNORECASE)
            if match:
                raw_val = match.group(1).strip()
                outside = re.sub(r"<!--[\s\S]*?-->", "", raw_val).strip()
                if outside:
                    val = outside
                else:
                    comments = re.findall(r"<!--([\s\S]*?)-->", raw_val)
                    val = " ".join(comments).strip()
                    val = re.sub(r"^e\.g\.,?\s*", "", val, flags=re.IGNORECASE).strip()

                val_lower = val.lower()
                is_placeholder = any(p in val_lower for p in placeholders)
                if val and not is_placeholder and val != "—":
                    info[key] = val

    return info


def render_html_dashboard(entries: list, prize_cutoff: int, last_updated: str) -> str:
    """Generate a sleek, dark-mode live leaderboard HTML dashboard."""
    rows = []
    for e in entries:
        rank = e["rank"]
        if rank == 1:
            rank_display = "🥇 1"
            badge = '<span class="badge gold">1st Place</span>'
        elif rank == 2:
            rank_display = "🥈 2"
            badge = '<span class="badge silver">2nd Place</span>'
        elif rank == 3:
            rank_display = "🥉 3"
            badge = '<span class="badge bronze">3rd Place</span>'
        elif rank <= prize_cutoff and e["score"] > 0:
            rank_display = str(rank)
            badge = '<span class="badge winner">🎉 Top 100</span>'
        else:
            rank_display = str(rank)
            badge = '<span class="badge standard">—</span>'

        name = e["name"] if e["name"] != "—" else f"@{e['username']}"
        roll = e["roll_no"]
        score = e["score"]
        tests = e["tests_passed"]
        ts = e["timestamp"][:19].replace("T", " ") if e["timestamp"] else "—"

        row_class = "winner-row" if rank <= prize_cutoff and e["score"] > 0 else ""
        rows.append(f"""
        <tr class="{row_class}">
            <td class="rank-col">{rank_display}</td>
            <td>{badge}</td>
            <td class="name-col"><strong>{name}</strong></td>
            <td><code>{roll}</code></td>
            <td><a href="https://github.com/{e['username']}" target="_blank">@{e['username']}</a></td>
            <td class="score-col"><strong>{score}</strong> / 130</td>
            <td><span class="test-pill">{tests} / 40</span></td>
            <td class="time-col">{ts}</td>
        </tr>""")

    empty_row = """
        <tr>
            <td colspan="8" style="text-align: center; padding: 48px; color: #8b949e; font-size: 1.1rem;">
                ⏳ <em>Awaiting first submissions... New Pull Requests will appear here live!</em>
            </td>
        </tr>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta http-equiv="refresh" content="15">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🏆 Challenge Leaderboard | OpenAI × DSG</title>
    <style>
        :root {{
            --bg: #0d1117;
            --card-bg: #161b22;
            --border: #30363d;
            --text: #c9d1d9;
            --text-white: #f0f6fc;
            --accent: #58a6ff;
            --green: #2ea043;
            --gold: #f1e05a;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background: var(--bg);
            color: var(--text);
            padding: 24px;
        }}
        .header {{
            text-align: center;
            margin-bottom: 24px;
            padding-bottom: 16px;
            border-bottom: 1px solid var(--border);
        }}
        .header h1 {{
            font-size: 2.2rem;
            color: var(--text-white);
            margin-bottom: 8px;
        }}
        .header h1 span {{ color: #10a37f; }}
        .header .subtitle {{
            color: #8b949e;
            font-size: 1rem;
        }}
        .stats-bar {{
            display: flex;
            justify-content: center;
            gap: 20px;
            margin-top: 14px;
            flex-wrap: wrap;
        }}
        .stat-item {{
            background: var(--card-bg);
            padding: 8px 18px;
            border-radius: 20px;
            border: 1px solid var(--border);
            font-size: 0.9rem;
        }}
        .live-dot {{
            display: inline-block;
            width: 10px;
            height: 10px;
            background: #238636;
            border-radius: 50%;
            margin-right: 6px;
            box-shadow: 0 0 8px #2ea043;
            animation: pulse 1.5s infinite;
        }}
        @keyframes pulse {{
            0% {{ opacity: 1; }}
            50% {{ opacity: 0.3; }}
            100% {{ opacity: 1; }}
        }}
        table {{
            width: 100%;
            max-width: 1200px;
            margin: 0 auto;
            border-collapse: collapse;
            background: var(--card-bg);
            border-radius: 12px;
            overflow: hidden;
            border: 1px solid var(--border);
            box-shadow: 0 10px 30px rgba(0,0,0,0.5);
        }}
        th {{
            background: #21262d;
            padding: 14px 16px;
            text-align: left;
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: #8b949e;
        }}
        td {{
            padding: 14px 16px;
            border-bottom: 1px solid var(--border);
            font-size: 0.95rem;
        }}
        tr:hover {{ background: rgba(255, 255, 255, 0.02); }}
        .rank-col {{ font-weight: bold; width: 70px; text-align: center; font-size: 1.1rem; }}
        .name-col strong {{ color: var(--text-white); }}
        .score-col {{ font-size: 1.15rem; color: #58a6ff; }}
        .score-col strong {{ color: #79c0ff; }}
        .test-pill {{
            background: rgba(56, 139, 253, 0.15);
            color: #58a6ff;
            padding: 4px 10px;
            border-radius: 12px;
            font-size: 0.85rem;
            font-weight: 600;
        }}
        .badge {{
            padding: 4px 10px;
            border-radius: 12px;
            font-size: 0.75rem;
            font-weight: 700;
            text-transform: uppercase;
        }}
        .badge.gold {{ background: #f1e05a33; color: #f1e05a; border: 1px solid #f1e05a66; }}
        .badge.silver {{ background: #c0c0c033; color: #c0c0c0; border: 1px solid #c0c0c066; }}
        .badge.bronze {{ background: #cd7f3233; color: #cd7f32; border: 1px solid #cd7f3266; }}
        .badge.winner {{ background: #23863633; color: #3fb950; border: 1px solid #23863666; }}
        .badge.standard {{ color: #6e7681; }}
        a {{ color: #58a6ff; text-decoration: none; }}
        a:hover {{ text-decoration: underline; }}
        code {{ background: rgba(110, 118, 129, 0.2); padding: 2px 6px; border-radius: 4px; font-family: monospace; }}
        .footer {{
            text-align: center;
            margin-top: 24px;
            color: #8b949e;
            font-size: 0.85rem;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🏆 <span>OpenAI × DSG</span> | Demystifying LLMs Challenge</h1>
        <div class="subtitle">Live Competition Leaderboard • Top {prize_cutoff} Win ChatGPT Plus Subscriptions</div>
        <div class="stats-bar">
            <div class="stat-item"><span class="live-dot"></span><strong>LIVE AUTO-REFRESH (15s)</strong></div>
            <div class="stat-item">👥 <strong>{len(entries)}</strong> Participants</div>
            <div class="stat-item">🎯 Cutoff: <strong>Top {prize_cutoff}</strong></div>
            <div class="stat-item">🕒 Last Updated: <strong>{last_updated}</strong></div>
        </div>
    </div>

    <table>
        <thead>
            <tr>
                <th style="text-align:center;">Rank</th>
                <th>Status</th>
                <th>Student Name</th>
                <th>Roll No</th>
                <th>GitHub</th>
                <th>Score</th>
                <th>Tests Passed</th>
                <th>Submitted At</th>
            </tr>
        </thead>
        <tbody>
            {"".join(rows) if rows else empty_row}
        </tbody>
    </table>

    <div class="footer">
        OpenAI × Data Science Group, IIT Roorkee • Tie-Breaker: Earlier submission timestamp ranks higher.
    </div>
</body>
</html>
"""


def run_leaderboard_build(args):
    repo = args.repo
    token = args.token
    prize_cutoff = args.prize_cutoff
    excluded_users = {u.strip().lower() for u in args.exclude_users.split(",") if u.strip()}

    now_utc = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    now_local = datetime.now().strftime("%H:%M:%S")

    print("\n" + "=" * 65)
    print(f"🏆 Demystifying LLMs Challenge — Live Leaderboard [{now_local}]")
    print("=" * 65)
    print(f"Target Repository : {repo}")
    print(f"Prize Cutoff      : Top {prize_cutoff} participants")
    print(f"Excluded Users    : {', '.join(sorted(excluded_users)) if excluded_users else 'None'}")
    if token:
        masked = token[:4] + "*" * (len(token) - 8) + token[-4:] if len(token) > 8 else "***"
        print(f"Authentication    : Authenticated ({masked}) — 5,000 req/hr")
    else:
        print("Authentication    : Unauthenticated — 60 req/hr")
    print("-" * 65)

    prs = get_all_prs(repo, token, state=args.state)

    raw_entries = []
    for pr in prs:
        username = pr.get("user", {}).get("login", "unknown")
        if username.lower() in excluded_users:
            continue

        pr_number = pr["number"]
        score, tests_passed, graded_at = get_latest_score(repo, pr_number, token)
        info = extract_participant_info(pr.get("body", ""))
        submission_time = graded_at or pr.get("created_at")

        raw_entries.append({
            "pr": pr_number,
            "username": username,
            "name": info["name"],
            "roll_no": info["roll_no"],
            "email": info["email"],
            "contact": info["contact"],
            "score": score,
            "tests_passed": tests_passed,
            "timestamp": submission_time,
            "url": pr.get("html_url", f"https://github.com/{repo}/pull/{pr_number}"),
        })

    # Deduplicate by username (keep best score, tie-break by earlier submission)
    student_map = {}
    for entry in raw_entries:
        user = entry["username"]
        if user not in student_map:
            student_map[user] = entry
        else:
            prev = student_map[user]
            if entry["score"] > prev["score"]:
                student_map[user] = entry
            elif entry["score"] == prev["score"] and (entry["timestamp"] or "") < (prev["timestamp"] or ""):
                student_map[user] = entry

    entries = list(student_map.values())
    entries.sort(key=lambda e: (-e["score"], e["timestamp"] or "9999"))
    for i, e in enumerate(entries):
        e["rank"] = i + 1

    # Print terminal table
    if entries:
        print(f"\n{'Rank':<5} | {'Student Name':<22} | {'GitHub':<18} | {'Score':<10} | {'Tests':<8} | {'Submitted'}")
        print("-" * 80)
        for e in entries:
            name_display = e["name"] if e["name"] != "—" else f"@{e['username']}"
            ts_short = e["timestamp"][:19].replace("T", " ") if e["timestamp"] else "—"
            print(f"#{e['rank']:<4} | {name_display:<22} | @{e['username']:<17} | {e['score']:>3}/130   | {e['tests_passed']:>2}/40   | {ts_short}")
    else:
        print("\n⏳ No active participant submissions yet. Awaiting PRs...")

    # Markdown generation
    md_lines = [
        "# 🏆 OpenAI x DSG Workshop — Live Challenge Leaderboard",
        "",
        f"*Last updated: `{now_utc}`*  ",
        f"*Total participants: `{len(entries)}`*  ",
        f"*Prize cutoff: Top `{prize_cutoff}` win ChatGPT Plus subscriptions!*",
        "",
        "| Rank | Status | Student Name | Roll No | GitHub | Score | Tests Passed | Submitted At |",
        "|:---:|:---:|---|---|---|:---:|:---:|---|",
    ]

    if entries:
        for e in entries:
            is_winner = e["rank"] <= prize_cutoff and e["score"] > 0
            prize_badge = "🎉 Winner" if is_winner else ("Top 100" if e["rank"] <= prize_cutoff else "—")
            ts_short = e["timestamp"][:19].replace("T", " ") if e["timestamp"] else "—"
            name_display = e["name"] if e["name"] != "—" else f"@{e['username']}"

            md_lines.append(
                f"| **{e['rank']}** | {prize_badge} | {name_display} | `{e['roll_no']}` | "
                f"[@{e['username']}](https://github.com/{e['username']}) | **{e['score']} / 130** | "
                f"`{e['tests_passed']} / 40` | {ts_short} |"
            )
    else:
        md_lines.append("| — | — | *Awaiting first submissions...* | — | — | — | — | — |")

    md_lines.extend([
        "",
        "---",
        "### ℹ️ How Scoring & Ties Work",
        "- **Easy Bugs (1-10):** 1 point each",
        "- **Medium Bugs (11-25):** 3 points each",
        "- **Hard Bugs (26-35):** 5 points each",
        "- **Integration Tests (1-5):** 5 points each",
        "- **Tie-Breaker:** When scores are equal, the earlier automated autograder run ranks higher.",
        "- *Leaderboard refreshes automatically during the challenge.*",
    ])

    with open(args.out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)

    html_content = render_html_dashboard(entries, prize_cutoff, now_utc)
    with open(args.out_html, "w", encoding="utf-8") as f:
        f.write(html_content)

    print("-" * 65)
    print(f"Updated: {args.out_md}")
    print(f"Updated: {args.out_html} (Open in browser to project on screen!)")
    print(f"Total Participants Displayed: {len(entries)}")
    print("=" * 65)


def main():
    parser = argparse.ArgumentParser(description="Build LLM Debugging Challenge Leaderboard")
    parser.add_argument("--repo", default=os.environ.get("GITHUB_REPO", DEFAULT_REPO),
                        help="Target GitHub repo (owner/repo)")
    parser.add_argument("--token", default=os.environ.get("GITHUB_TOKEN", "").strip(),
                        help="GitHub Personal Access Token (or GITHUB_TOKEN env var)")
    parser.add_argument("--state", choices=["open", "closed", "all"], default="open",
                        help="PR state to inspect (default: open)")
    parser.add_argument("--prize-cutoff", type=int, default=DEFAULT_CUTOFF,
                        help="Cutoff rank for prizes (default: 100)")
    parser.add_argument("--exclude-users", default=DEFAULT_EXCLUDED,
                        help="Comma-separated list of usernames to ignore (test accounts)")
    parser.add_argument("--out-md", default=str(REPO_ROOT / "LEADERBOARD.md"),
                        help="Output path for markdown leaderboard")
    parser.add_argument("--out-html", default=str(REPO_ROOT / "LEADERBOARD.html"),
                        help="Output path for HTML live dashboard")
    parser.add_argument("--out-json", default=str(REPO_ROOT / "leaderboard.json"),
                        help="Output path for JSON leaderboard")
    parser.add_argument("--watch", type=int, default=0,
                        help="Auto-refresh interval in seconds (e.g. --watch 15)")
    args = parser.parse_args()

    if args.watch > 0:
        print(f"Starting live watch mode (refreshing every {args.watch}s)... Press Ctrl+C to stop.")
        try:
            while True:
                run_leaderboard_build(args)
                time.sleep(args.watch)
        except KeyboardInterrupt:
            print("\nWatch mode stopped.")
    else:
        run_leaderboard_build(args)


if __name__ == "__main__":
    main()
