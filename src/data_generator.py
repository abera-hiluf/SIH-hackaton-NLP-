"""Generate a clearly labelled synthetic safety-report dataset for the demo.

This module creates synthetic demonstration records only. The output is NOT
official OIL data and must not be represented as such.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
import random

import pandas as pd


SEED = 26165
DEFAULT_ROWS = 500
OUTPUT_PATH = Path(__file__).resolve().parents[1] / "data" / "synthetic_reports.csv"

SITES = ["Digboi", "Duliajan", "Bongaigaon", "Geleki", "Naharkatiya"]

SCENARIOS = {
    "Energy Isolation": {
        "rule": "Verify isolation and zero energy",
        "sif": [True, True, False],
        "activities": ["pump maintenance", "valve replacement", "compressor servicing"],
        "barriers": ["lockout/tagout not verified", "incorrect isolation point", "missing zero-energy test"],
        "hazards": ["residual pressure remained in the line", "the pump unexpectedly started", "a valve was still connected to the live header"],
        "controls": ["the technician stopped work and isolated the equipment", "the supervisor initiated a toolbox talk", "the line was depressurized before work resumed"],
    },
    "Hot Work": {
        "rule": "Control ignition sources and combustible materials",
        "sif": [True, True, False],
        "activities": ["welding on a process pipe", "grinding near the workshop", "cutting a steel support"],
        "barriers": ["hot-work permit not displayed", "gas test not repeated", "combustibles not cleared from the work area"],
        "hazards": ["sparks travelled toward an open drain", "flammable vapour was detected during the task", "nearby oily rags began to smoulder"],
        "controls": ["the fire watch raised the alarm and work was stopped", "the area was made safe and retested", "a suitable fire extinguisher was positioned before restart"],
    },
    "Confined Space": {
        "rule": "Obtain authorization before entering a confined space",
        "sif": [True, True, False],
        "activities": ["cleaning a storage tank", "inspection inside a vessel", "entry into a sump"],
        "barriers": ["continuous gas monitoring was unavailable", "rescue plan was not briefed", "entry permit was incomplete"],
        "hazards": ["oxygen readings fell below the safe limit", "a worker reported dizziness inside the vessel", "the attendant lost radio contact with the entrant"],
        "controls": ["the entrant exited immediately and medical support was alerted", "ventilation was increased and the permit was reviewed", "the standby rescue team secured the area"],
    },
    "Line of Fire": {
        "rule": "Position yourself outside the line of fire",
        "sif": [True, False, False],
        "activities": ["loosening a seized flange", "handling a pressurized hose", "moving materials beside a pipe rack"],
        "barriers": ["exclusion zone was not maintained", "stored energy was not assessed", "spotter communication was unclear"],
        "hazards": ["the flange shifted suddenly toward the worker", "the hose whipped when the coupling released", "a suspended load swung across the access route"],
        "controls": ["the crew stepped back and reassessed the task", "the area was barricaded before continuing", "the lifting activity was paused for a new briefing"],
    },
    "Working at Height": {
        "rule": "Protect yourself against falls from height",
        "sif": [True, False, False],
        "activities": ["accessing a tank roof", "repairing an elevated platform", "installing cable tray at height"],
        "barriers": ["harness was not attached to an anchor", "scaffold inspection tag was missing", "toe board was not installed"],
        "hazards": ["the worker slipped near an unprotected edge", "a tool fell to the level below", "the platform moved while being accessed"],
        "controls": ["work stopped until a certified access system was provided", "the dropped-object zone was cleared", "the supervisor arranged a scaffold inspection"],
    },
    "Lifting Operations": {
        "rule": "Plan and control lifting operations",
        "sif": [True, False, False],
        "activities": ["lifting a pump motor", "loading a valve assembly", "relocating a chemical drum"],
        "barriers": ["lifting plan was not followed", "sling inspection was overdue", "personnel entered the suspended-load zone"],
        "hazards": ["the load tilted unexpectedly above the work crew", "a sling strand began to separate", "the load passed over an occupied walkway"],
        "controls": ["the banksman stopped the lift immediately", "the load was lowered to a safe position", "the route was cleared and barricaded"],
    },
    "Driving / Vehicle Safety": {
        "rule": "Drive defensively and use seat belts",
        "sif": [True, False, False],
        "activities": ["driving between installations", "reversing a service vehicle", "transporting equipment on an access road"],
        "barriers": ["reversing alarm was not working", "seat belt was not used", "journey plan was not reviewed"],
        "hazards": ["the vehicle nearly struck a pedestrian at the gate", "the driver lost control on a wet bend", "a load shifted during transit"],
        "controls": ["the driver stopped and reported the near miss", "the vehicle was removed from service for inspection", "the journey was restarted only after the load was secured"],
    },
}


def _report_text(
    category: str,
    scenario: dict[str, list[str]],
    sif_potential: bool,
    rng: random.Random,
) -> str:
    activity = rng.choice(scenario["activities"])
    barrier = rng.choice(scenario["barriers"])
    hazard = rng.choice(scenario["hazards"])
    control = rng.choice(scenario["controls"])
    context = rng.choice(
        [
            "during the morning shift",
            "while preparing the job",
            "during a routine maintenance task",
            "after a change in the work sequence",
        ]
    )
    if sif_potential:
        severity = "The condition created a credible potential for a fatal or permanently disabling event."
    else:
        severity = "The deviation was identified early, before anyone was exposed to serious harm."
    return f"During {activity} {context}, {barrier}. {hazard.capitalize()}. {severity} {control.capitalize()}."


def generate_reports(n_rows: int = DEFAULT_ROWS, seed: int = SEED) -> pd.DataFrame:
    """Return approximately 500 varied, logically labelled synthetic reports."""
    rng = random.Random(seed)
    start = date(2024, 1, 1)
    categories = list(SCENARIOS)
    records = []

    for index in range(n_rows):
        category = categories[index % len(categories)] if index < len(categories) else rng.choice(categories)
        scenario = SCENARIOS[category]
        sif_potential = bool(rng.choice(scenario["sif"]))
        records.append(
            {
                "report_id": f"SYN-{index + 1:04d}",
                "date": (start + timedelta(days=rng.randint(0, 730))).isoformat(),
                "site": rng.choice(SITES),
                "activity": f"{category} — {rng.choice(scenario['activities'])}",
                "report_text": _report_text(category, scenario, sif_potential, rng),
                "sif_potential": "Yes" if sif_potential else "No",
                "life_saving_rule": scenario["rule"],
                "barrier_failure": rng.choice(scenario["barriers"]),
            }
        )

    return pd.DataFrame(records)


def main() -> None:
    """Generate the CSV artifact used by the prototype."""
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    reports = generate_reports()
    reports.to_csv(OUTPUT_PATH, index=False)
    print(f"Generated {len(reports)} SYNTHETIC safety reports at {OUTPUT_PATH}")
    print("This is NOT official OIL data.")


if __name__ == "__main__":
    main()
