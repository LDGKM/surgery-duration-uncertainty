"""Export cases as HL7 FHIR R4 resources (first step towards a reusable pipeline).

    python src/fhir.py        # writes data/fhir_bundle_sample.json (first 20 cases)

Each case becomes three linked resources inside a FHIR "collection" Bundle:
  Patient     sex and age (VitalDB gives the age, not a birth date)
  Encounter   the surgical stay, linked to the patient
  Procedure   the operation: name, department, approach, emergency flag and
              performedPeriod (start/end of the case)

VitalDB times are seconds relative to the start of the case, with no calendar
date. To produce valid FHIR dateTime values we anchor every case on the same
arbitrary reference date (REFERENCE); only durations are meaningful.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from data import ROOT, load

REFERENCE = datetime(2000, 1, 1, 8, 0, tzinfo=timezone.utc)
GENDER = {"M": "male", "F": "female"}


def iso(seconds: float) -> str:
    return (REFERENCE + timedelta(seconds=float(seconds))).isoformat()


def case_to_resources(row) -> list[dict]:
    pid, eid, prid = f"patient-{row.subjectid}", f"encounter-{row.caseid}", f"procedure-{row.caseid}"
    patient = {"resourceType": "Patient", "id": pid,
               "gender": GENDER.get(str(row.sex), "unknown"),
               "extension": [{"url": "urn:surgery-duration-uncertainty:age-at-surgery",   # age at surgery (VitalDB has no birth date)
                              "valueAge": {"value": float(row.age), "unit": "a", "system": "http://unitsofmeasure.org", "code": "a"}}]}
    encounter = {"resourceType": "Encounter", "id": eid, "status": "finished",
                 "class": {"system": "http://terminology.hl7.org/CodeSystem/v3-ActCode", "code": "IMP", "display": "inpatient encounter"},
                 "subject": {"reference": f"Patient/{pid}"},
                 "serviceType": {"text": str(row.department)}}
    procedure = {"resourceType": "Procedure", "id": prid, "status": "completed",
                 "subject": {"reference": f"Patient/{pid}"}, "encounter": {"reference": f"Encounter/{eid}"},
                 "code": {"text": str(row.opname)},
                 "category": {"text": str(row.optype)},
                 "performedPeriod": {"start": iso(row.casestart), "end": iso(row.caseend)},
                 "reasonCode": [{"text": str(row.dx)}],
                 "extension": [{"url": "urn:surgery-duration-uncertainty:approach", "valueString": str(row.approach)},
                               {"url": "urn:surgery-duration-uncertainty:emergency", "valueBoolean": bool(row.emop)}]}
    return [patient, encounter, procedure]


def check(resource: dict) -> None:
    """Minimal structural checks (a full validation would use the official FHIR validator)."""
    assert resource["resourceType"] in {"Patient", "Encounter", "Procedure"}
    if resource["resourceType"] == "Procedure":
        p = resource["performedPeriod"]
        assert datetime.fromisoformat(p["end"]) > datetime.fromisoformat(p["start"])
        assert resource["subject"]["reference"].startswith("Patient/")


if __name__ == "__main__":
    df = load().head(20)
    entries, seen = [], set()
    for row in df.itertuples():
        for r in case_to_resources(row):
            check(r)
            if r["id"] in seen:              # a patient operated twice appears once
                continue
            seen.add(r["id"])
            entries.append({"fullUrl": f"urn:uuid:{r['resourceType'].lower()}-{r['id']}", "resource": r})
    bundle = {"resourceType": "Bundle", "type": "collection", "entry": entries}
    out = ROOT / "data" / "fhir_bundle_sample.json"
    out.write_text(json.dumps(bundle, indent=1))
    print(f"{len(entries)} resources written to {out.relative_to(ROOT)}")
