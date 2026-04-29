#!/usr/bin/env python3
"""
Generate and send weekly Slack report from traffic data.
Sends a rich formatted message to Slack via webhook.
"""

import json
import os
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd


def format_number(num):
    """Format number with commas for readability."""
    return f"{num:,}" if num >= 1000 else str(num)


def calculate_trend(current, previous):
    """Calculate percentage change and return emoji."""
    if previous == 0:
        return "🆕", 0  # New data
    change = ((current - previous) / previous) * 100
    if change > 5:
        return "📈", change
    elif change < -5:
        return "📉", change
    else:
        return "➡️", change


def generate_slack_report(data_dir='/tmp/traffic-data', webhook_url=None):
    """Generate and send Slack report."""
    data_dir = Path(data_dir)
    parquet_file = data_dir / 'traffic_history.parquet'
    metadata_file = data_dir / 'metadata.json'

    if not parquet_file.exists():
        print("❌ No data to report")
        return False

    print("📊 Generating weekly Slack report...")

    # Load data
    df = pd.read_parquet(parquet_file)
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)

    # Get last 7 days for weekly report
    today = df['date'].max()
    week_ago = today - timedelta(days=7)
    two_weeks_ago = today - timedelta(days=14)

    this_week = df[(df['date'] > week_ago) & (df['date'] <= today)]
    last_week = df[(df['date'] > two_weeks_ago) & (df['date'] <= week_ago)]

    if len(this_week) == 0:
        print("⚠️  No data for this week")
        return False

    # Calculate metrics
    clones_total = this_week['clones_total'].sum()
    clones_unique = this_week['clones_unique'].sum()
    views_total = this_week['views_total'].sum()
    views_unique = this_week['views_unique'].sum()

    # Week-over-week changes
    clones_trend, clones_change = "🆕", 0
    views_trend, views_change = "🆕", 0

    if len(last_week) > 0:
        clones_prev = last_week['clones_total'].sum()
        views_prev = last_week['views_total'].sum()
        clones_trend, clones_change = calculate_trend(clones_total, clones_prev)
        views_trend, views_change = calculate_trend(views_total, views_prev)

    # Best day this week
    best_day = this_week.loc[this_week['clones_total'].idxmax()] if len(this_week) > 0 else None

    # Parse top paths and referrers from most recent day with data
    latest = this_week.iloc[-1] if len(this_week) > 0 else None
    top_paths = []
    top_referrers = []

    if latest is not None:
        try:
            paths_data = json.loads(latest['top_paths']) if latest['top_paths'] else []
            top_paths = paths_data[:3] if paths_data else []
        except Exception:
            pass

        try:
            refs_data = json.loads(latest['top_referrers']) if latest['top_referrers'] else []
            top_referrers = refs_data[:3] if refs_data else []
        except Exception:
            pass

    # Get repository name from environment or use default
    repo_name = os.environ.get('GITHUB_REPOSITORY', 'Repository')

    # Build Slack message blocks
    blocks = [
        {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": "📊 Weekly GitHub Traffic Report",
                "emoji": True
            }
        },
        {
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": f"*{week_ago.strftime('%b %d')} - {today.strftime('%b %d, %Y')}* | {repo_name}"
                }
            ]
        },
        {
            "type": "divider"
        },
        {
            "type": "section",
            "fields": [
                {
                    "type": "mrkdwn",
                    "text": f"*Clones* {clones_trend}\n`{format_number(clones_total)}` total\n`{format_number(clones_unique)}` unique"
                },
                {
                    "type": "mrkdwn",
                    "text": f"*Views* {views_trend}\n`{format_number(views_total)}` total\n`{format_number(views_unique)}` unique"
                }
            ]
        }
    ]

    # Add week-over-week changes if we have previous week data
    if len(last_week) > 0 and best_day is not None:
        change_fields = []

        # Week-over-week changes
        wow_text = "*Week-over-week*\n"
        if clones_change != 0:
            wow_text += f"Clones: `{clones_change:+.1f}%`\n"
        if views_change != 0:
            wow_text += f"Views: `{views_change:+.1f}%`"
        change_fields.append({
            "type": "mrkdwn",
            "text": wow_text.strip()
        })

        # Best day
        change_fields.append({
            "type": "mrkdwn",
            "text": f"*Best day*\n{best_day['date'].strftime('%A')}\n`{best_day['clones_total']}` clones, `{best_day['views_total']}` views"
        })

        blocks.append({
            "type": "section",
            "fields": change_fields
        })

    # Add top content if available
    if top_paths or top_referrers:
        blocks.append({"type": "divider"})

        content_fields = []

        if top_paths:
            paths_text = "*Top Pages*\n"
            for i, path in enumerate(top_paths, 1):
                path_name = path.get('path', 'N/A') if isinstance(path, dict) else str(path)
                # Truncate long paths
                if len(path_name) > 40:
                    path_name = '...' + path_name[-37:]
                paths_text += f"{i}. `{path_name}`\n"
            content_fields.append({
                "type": "mrkdwn",
                "text": paths_text.strip()
            })

        if top_referrers:
            refs_text = "*Top Referrers*\n"
            for i, ref in enumerate(top_referrers, 1):
                ref_name = ref.get('referrer', 'N/A') if isinstance(ref, dict) else str(ref)
                # Clean up referrer display
                if ref_name == 'github.com':
                    ref_name = 'GitHub'
                elif ref_name.startswith('https://'):
                    ref_name = ref_name.replace('https://', '')
                refs_text += f"{i}. {ref_name}\n"
            content_fields.append({
                "type": "mrkdwn",
                "text": refs_text.strip()
            })

        if content_fields:
            blocks.append({
                "type": "section",
                "fields": content_fields
            })

    # Add daily breakdown as a compact chart
    if len(this_week) > 1:
        blocks.append({"type": "divider"})

        daily_text = "*Daily Breakdown*\n```\n"
        for _, row in this_week.iterrows():
            day = row['date'].strftime('%a')
            # Create a simple bar chart with unicode blocks
            clone_bar = '█' * min(int(row['clones_total'] / max(this_week['clones_total'].max(), 1) * 10), 10)
            daily_text += f"{day}: {clone_bar} {row['clones_total']}c/{row['views_total']}v\n"
        daily_text += "```"

        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": daily_text
            }
        })

    # Add footer with metadata
    blocks.extend([
        {
            "type": "divider"
        },
        {
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": f"📈 *{metadata['record_count']}* days collected | 💾 *{metadata['file_size_bytes'] / 1024:.1f}* KB stored | 🔄 Updated {datetime.now().strftime('%H:%M UTC')}"
                }
            ]
        }
    ])

    # Prepare payload
    payload = {
        "text": f"Weekly GitHub Traffic Report: {format_number(clones_total)} clones, {format_number(views_total)} views",
        "blocks": blocks
    }

    # Send to Slack if webhook URL is provided
    if not webhook_url:
        webhook_url = os.environ.get('SLACK_WEBHOOK_URL')

    if webhook_url:
        print("📤 Sending to Slack webhook...")

        req = urllib.request.Request(
            webhook_url,
            data=json.dumps(payload).encode('utf-8'),
            headers={'Content-Type': 'application/json'}
        )

        try:
            with urllib.request.urlopen(req) as response:
                if response.status == 200:
                    print("✅ Slack report sent successfully!")
                    return True
                else:
                    print(f"⚠️  Unexpected response: {response.status}")
                    return False
        except urllib.error.HTTPError as e:
            print(f"❌ HTTP Error {e.code}: {e.reason}")
            # Try to read error response
            try:
                error_body = e.read().decode('utf-8')
                print(f"   Response: {error_body}")
            except Exception:
                pass
            return False
        except Exception as e:
            print(f"❌ Failed to send Slack report: {e}")
            return False
    else:
        print("ℹ️  No Slack webhook URL provided (set SLACK_WEBHOOK_URL)")
        print("\n📋 Report Preview:")
        print(json.dumps(payload, indent=2))
        return True


if __name__ == '__main__':
    data_dir = sys.argv[1] if len(sys.argv) > 1 else '/tmp/traffic-data'
    webhook_url = sys.argv[2] if len(sys.argv) > 2 else None

    try:
        success = generate_slack_report(data_dir, webhook_url)
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"❌ Error generating Slack report: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
