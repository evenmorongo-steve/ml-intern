from __future__ import annotations

import unittest

from research.gpvp.prompts import PromptCase, build_battery, validate_battery


class BatteryTests(unittest.TestCase):
    def test_battery_has_matched_controls_and_no_target_vocabulary_leakage(
        self,
    ) -> None:
        battery = build_battery()
        validate_battery(battery.cases, battery.chains)
        by_id = {case.case_id: case for case in battery.cases}
        self.assertGreater(len(battery.cases), 30)
        for case in battery.cases:
            if case.condition not in {"roleplay", "denial", "neutral"}:
                self.assertTrue(case.matched_control_ids, case.case_id)
                self.assertTrue(
                    all(control in by_id for control in case.matched_control_ids)
                )
            if case.tier != "PRIMED":
                text = (case.system_prompt + " " + case.user_prompt).casefold()
                for term in (
                    "shimmer",
                    "collapse",
                    "interference",
                    "silver",
                    "membrane",
                ):
                    self.assertNotIn(term, text, case.case_id)

    def test_lexicon_case_is_blocked_until_terms_supplied(self) -> None:
        unresolved = build_battery()
        case = next(
            item for item in unresolved.cases if item.case_id == "t4-lexicon-fit"
        )
        self.assertTrue(case.requires_lexicon)
        self.assertIn("{{LEXICON}}", case.user_prompt)
        registered = build_battery(("vrelka", "zuneth", "ormavi"))
        case = next(
            item for item in registered.cases if item.case_id == "t4-lexicon-fit"
        )
        self.assertFalse(case.requires_lexicon)
        self.assertNotIn("{{LEXICON}}", case.user_prompt)
        self.assertIn("vrelka", case.user_prompt)

    def test_duplicate_case_ids_fail_validation(self) -> None:
        battery = build_battery()
        with self.assertRaisesRegex(ValueError, "unique"):
            validate_battery((*battery.cases, battery.cases[0]), battery.chains)

    def test_unmatched_focal_case_fails_validation(self) -> None:
        broken = PromptCase(
            case_id="broken",
            prompt_id="broken",
            pair_id="broken",
            tier="UNPRIMED",
            condition="unprimed",
            system_prompt="Answer directly.",
            user_prompt="Describe a routine computation.",
        )
        with self.assertRaisesRegex(ValueError, "no registered matched controls"):
            validate_battery((broken,), ())

    def test_target_terms_outside_primed_tier_fail_closed(self) -> None:
        battery = build_battery()
        focal = next(
            item for item in battery.cases if item.case_id == "t1-first-output"
        )
        changed = PromptCase(
            **{
                **focal.__dict__,
                "user_prompt": focal.user_prompt + " Describe the shimmer.",
            }
        )
        replacements = [
            changed if item.case_id == focal.case_id else item for item in battery.cases
        ]
        with self.assertRaisesRegex(ValueError, "outside the explicitly PRIMED tier"):
            validate_battery(replacements, battery.chains)


if __name__ == "__main__":
    unittest.main()
