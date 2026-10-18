from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

CATALOG_PATH = Path(__file__).with_name("catalog.json")


@lru_cache(maxsize=1)
def load_catalog() -> dict[str, Any]:
    return json.loads(CATALOG_PATH.read_text(encoding="utf-8"))


def supported_events() -> list[dict[str, Any]]:
    return [event for event in load_catalog()["events"] if event.get("api_supported") is True]


def supported_event_types() -> frozenset[str]:
    return frozenset(event["event_type"] for event in supported_events())


def event_by_type(event_type: str) -> dict[str, Any] | None:
    for event in load_catalog()["events"]:
        if event["event_type"] == event_type:
            return event
    return None


def event_type_for_observed(observed_event_type: str) -> str | None:
    """Return an unambiguous proposal; this is never evidence of recognition."""
    from .evidence import proposed_mapping
    candidates = proposed_mapping(observed_event_type)['candidate_types']
    return candidates[0] if len(candidates) == 1 else None


def source_links_for(event_type: str) -> list[dict[str, str]]:
    catalog = load_catalog()
    event = event_by_type(event_type)
    if event is None or not event.get("api_supported"):
        return []
    provisions = {item["id"]: item for item in catalog["provisions"]}
    sources = {item["id"]: item for item in catalog["sources"]}
    links = []
    for provision_id in event.get("provision_ids", []):
        provision = provisions[provision_id]
        source = sources[provision["source_id"]]
        links.append(
            {
                "event_type": event_type,
                "source": source["short_name"],
                "locator": provision["locator"],
                "url": provision["official_url"],
                "title": source["title"],
            }
        )
    return links
