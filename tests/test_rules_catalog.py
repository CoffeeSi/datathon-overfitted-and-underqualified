import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from overfitted_underqualified.rules.catalog import (
    event_type_for_observed,
    load_catalog,
    source_links_for,
    supported_event_types,
)


class RulesCatalogTests(unittest.TestCase):
    def test_supported_events_have_public_provisions_and_urls(self) -> None:
        catalog = load_catalog()
        provisions = {item["id"]: item for item in catalog["provisions"]}
        sources = {item["id"]: item for item in catalog["sources"]}
        self.assertGreaterEqual(len(catalog["events"]), 1)
        self.assertEqual(catalog["status"], "p1_accepted")
        for source in catalog["sources"]:
            self.assertTrue(source["official_url"].startswith("https://adilet.zan.kz/"))
            self.assertTrue(source["short_name"].strip())
        for event in catalog["events"]:
            self.assertTrue(event["api_supported"])
            self.assertIn(event["link_kind"], {"direct_text", "team_visible_plus_source"})
            self.assertTrue(event["provision_ids"])
            self.assertTrue(event["applicability"])
            self.assertTrue(event["maps_from_observed_event_type"])
            self.assertTrue(event["visible_action"].strip())
            for provision_id in event["provision_ids"]:
                provision = provisions[provision_id]
                self.assertIn(provision["source_id"], sources)
                self.assertTrue(provision["locator"].strip())
                self.assertTrue(provision["official_url"].startswith("https://adilet.zan.kz/"))
                self.assertTrue(provision["text"].strip())
                self.assertTrue(source_links_for(event["event_type"]))

    def test_excluded_rows_are_not_api_types(self) -> None:
        supported = supported_event_types()
        catalog = load_catalog()
        for row in catalog["excluded"]:
            self.assertNotIn(row["observed_event_type"], supported)
            self.assertTrue(row["reason"].strip())

    def test_binary_label_is_not_an_event_type(self) -> None:
        self.assertNotIn("violation", supported_event_types())
        self.assertNotIn("no_violation", supported_event_types())

    def test_team_included_clips_are_supported(self) -> None:
        catalog = load_catalog()
        supported = supported_event_types()
        wanted = {
            "X-13": "collection_cash_amount_registration_not_observed",
            "X-15": "unsecured_accepted_cash",
            "X-17": "cash_bag_received_without_visible_identity_authority_check",
            "X-11": "safe_key_left_in_lock",
            "X-22": "safe_key_left_in_lock",
            "X-24": "suspected_stamp_or_seal_left_unattended",
        }
        by_clip: dict[str, str] = {}
        for event in catalog["events"]:
            for clip_id in event["example_clip_ids"]:
                by_clip[clip_id] = event["event_type"]
        for clip_id, event_type in wanted.items():
            self.assertIn(event_type, supported)
            self.assertEqual(by_clip[clip_id], event_type)

    def test_observed_tags_from_task1_map_to_supported_types(self) -> None:
        self.assertEqual(
            event_type_for_observed("accepted_cash_not_moved_to_lockable_storage"),
            "unsecured_accepted_cash",
        )
        self.assertIsNone(event_type_for_observed("beverage_near_suspected_cash"))
        self.assertIsNone(event_type_for_observed("cash_area_door_open_during_work"))


if __name__ == "__main__":
    unittest.main()
