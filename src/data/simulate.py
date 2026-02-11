"""
Synthetic fleet dataset generator.

Outputs:
  - data/raw/vehicle_day.csv
  - data/raw/dtc_event.csv
  - data/raw/work_order.csv

Run:
  python -m src.data.simulate --config config.yaml
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("config.yaml did not parse into a dictionary.")
    return cfg


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def clamp(x: float, lo: float, hi: float) -> float:
    return float(np.clip(x, lo, hi))


def to_date_str(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d")


def sample_route_context(
    rng: np.random.Generator,
    route_types: List[str],
    route_probs: List[float],
) -> Tuple[str, float, float]:
    """Return (route_type, hilliness_index, stop_go_index)."""
    rt = rng.choice(route_types, p=np.array(route_probs) / np.sum(route_probs))

    if rt == "highway":
        hill = rng.uniform(0.0, 0.5)
        stopgo = rng.uniform(0.0, 0.35)
    elif rt == "mixed":
        hill = rng.uniform(0.0, 0.8)
        stopgo = rng.uniform(0.2, 0.75)
    else:  # urban
        hill = rng.uniform(0.0, 0.6)
        stopgo = rng.uniform(0.6, 1.0)

    return str(rt), float(hill), float(stopgo)


def seasonal_ambient_temp_c(
    rng: np.random.Generator,
    base_mean: float,
    base_std: float,
    day_of_year: int,
) -> float:
    """
    Simple seasonal temperature model:
    mean + amplitude*sin(...) + noise.
    """
    amplitude = 12.0
    seasonal = base_mean + amplitude * np.sin(2.0 * np.pi * (day_of_year - 80) / 365.0)
    noise = rng.normal(0.0, base_std * 0.35)
    return float(seasonal + noise)


def sample_duty_cycle(rng: np.random.Generator, route_type: str) -> float:
    # Beta distributions to make patterns plausible by route type
    if route_type == "highway":
        a, b = 5.0, 2.0
    elif route_type == "mixed":
        a, b = 4.0, 3.0
    else:
        a, b = 3.0, 4.0
    return float(rng.beta(a, b))


def miles_from_context(rng: np.random.Generator, route_type: str, duty_cycle: float, precip: int) -> float:
    if route_type == "highway":
        base = rng.normal(260.0, 50.0)
    elif route_type == "mixed":
        base = rng.normal(180.0, 45.0)
    else:
        base = rng.normal(120.0, 35.0)

    # precipitation tends to reduce miles a bit
    weather_factor = 0.90 if precip == 1 else 1.0
    miles = max(0.0, base * (0.55 + 0.75 * duty_cycle) * weather_factor)
    return float(miles)


def action_for_subsystem(subsystem: str) -> str:
    mapping = {
        "cooling": "Inspect cooling system; check leaks/thermostat; pressure test",
        "aftertreatment": "Inspect aftertreatment; check DPF/NOx; perform service regen if needed",
        "drivetrain": "Inspect drivetrain/transmission; check fluid, sensors, vibration sources",
        "electrical": "Inspect electrical; test battery/alternator; check grounds/connectors",
        "brakes_wheel_end": "Inspect brakes/wheel-end; measure wear; check sensors/bearings",
    }
    return mapping.get(subsystem, "Inspect subsystem")


def notes_for_subsystem(subsystem: str) -> str:
    mapping = {
        "cooling": "Recent overheating pattern + coolant temp spikes observed.",
        "aftertreatment": "DPF delta-P rising + regen anomalies; check soot loading.",
        "drivetrain": "Elevated vibration/trans temp; check driveline and transmission health.",
        "electrical": "Low battery voltage events; possible charging system issue.",
        "brakes_wheel_end": "Wear index high; possible brake/wheel sensor or mechanical wear.",
    }
    return mapping.get(subsystem, "Service recommendation based on risk signals.")


def generate(
    cfg: Dict[str, Any],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    seed = int(cfg["project"]["random_seed"])
    rng = np.random.default_rng(seed)

    sim = cfg["simulation"]
    n_vehicles = int(sim["n_vehicles"])
    n_days = int(sim["n_days"])
    start_date = datetime.fromisoformat(sim["start_date"])

    subsystems: List[str] = list(sim["subsystems"])
    _horizons: List[int] = list(sim["horizons_days"])  # reserved for future label generation / config sanity

    rw = sim["route_weather"]
    route_types = list(rw["route_types"])
    route_probs = list(rw["route_type_probs"])
    temp_mean = float(rw["ambient_temp_mean_c"])
    temp_std = float(rw["ambient_temp_std_c"])
    precip_prob = float(rw["precipitation_prob"])

    dtc_cfg = sim["dtc"]
    max_codes = int(dtc_cfg["max_distinct_codes_per_day"])
    base_dtc_rate = float(dtc_cfg["base_event_rate_per_day"])
    dtc_catalog = dtc_cfg["catalog_by_subsystem"]

    hm = sim["health_model"]
    init_mu = float(hm["initial_health_mean"])
    init_sd = float(hm["initial_health_std"])
    degr_base = hm["degradation_base_per_day"]
    stress_w = hm["stress_weights"]
    hz = hm["hazard"]
    base_hazard = float(hz["base_hazard"])
    alpha_health = float(hz["alpha_health"])
    alpha_dtc_recent = float(hz["alpha_dtc_recent"])
    alpha_spc_alert = float(hz["alpha_spc_alert"])

    # outputs
    vehicle_rows: List[Dict[str, Any]] = []
    dtc_rows: List[Dict[str, Any]] = []
    wo_rows: List[Dict[str, Any]] = []

    # vehicle IDs
    vehicle_ids = [f"V{idx:04d}" for idx in range(1, n_vehicles + 1)]

    # per-vehicle state
    states: Dict[str, Dict[str, Any]] = {}
    for vid in vehicle_ids:
        health = {s: clamp(rng.normal(init_mu, init_sd), 0.85, 1.0) for s in subsystems}
        states[vid] = {
            "mileage_total": float(rng.uniform(10_000, 250_000)),
            "health": health,
            "brake_wear": float(rng.uniform(0.0, 0.35)),
            "last_dtc_day_index": -10_000,
            "in_shop_until": None,  # datetime or None
        }

    # breakdown event log: list of (vehicle_id, date, subsystem)
    breakdown_events: List[Tuple[str, datetime, str]] = []

    for day_idx in range(n_days):
        dt = start_date + timedelta(days=day_idx)
        day_of_year = int(dt.strftime("%j"))

        for vid in vehicle_ids:
            st = states[vid]
            in_shop_until = st["in_shop_until"]

            # If vehicle is in shop, set miles/duty low and skip failure logic
            if in_shop_until is not None and dt <= in_shop_until:
                # keep a row to preserve timeline
                row = {
                    "date": to_date_str(dt),
                    "vehicle_id": vid,
                    "route_type": "shop",
                    "hilliness_index": 0.0,
                    "stop_go_index": 0.0,
                    "ambient_temp_c": seasonal_ambient_temp_c(rng, temp_mean, temp_std, day_of_year),
                    "precipitation": 0,
                    "duty_cycle": 0.0,
                    "miles_today": 0.0,
                    "mileage_total": st["mileage_total"],
                }
                # telematics (stable, no stress)
                row.update(
                    {
                        "coolant_temp_avg_c": 80.0,
                        "coolant_temp_max_c": 88.0,
                        "egt_max_c": 320.0,
                        "battery_v_min": 12.2,
                        "fuel_rate_avg_lph": 0.0,
                        "regen_count": 0,
                        "idle_pct": 0.0,
                        "oil_temp_avg_c": 75.0,
                        "oil_pressure_min_kpa": 320.0,
                        "intake_air_temp_avg_c": 25.0,
                        "boost_pressure_avg_kpa": 0.0,
                        "dpf_delta_p_avg_kpa": 4.0,
                        "trans_temp_avg_c": 60.0,
                        "vibration_rms": 0.35,
                        "brake_wear_index": st["brake_wear"],
                        "dtc_count_today": 0,
                        "spc_alert_today": 0,
                    }
                )
                vehicle_rows.append(row)
                continue

            # Sample route/weather context
            route_type, hilliness, stop_go = sample_route_context(rng, route_types, route_probs)
            ambient = seasonal_ambient_temp_c(rng, temp_mean, temp_std, day_of_year)
            precip = int(rng.random() < precip_prob)

            duty = sample_duty_cycle(rng, route_type)
            miles_today = miles_from_context(rng, route_type, duty, precip)

            st["mileage_total"] += miles_today

            # Stress factors
            ambient_extreme = 1.0 if (ambient < -10.0 or ambient > 30.0) else 0.0
            stress = (
                float(stress_w["duty_cycle"]) * duty
                + float(stress_w["hilliness"]) * hilliness
                + float(stress_w["stop_go"]) * stop_go
                + float(stress_w["ambient_temp_extreme"]) * ambient_extreme
            )

            # Update subsystem health (degrade)
            health = st["health"]
            for s in subsystems:
                base = float(degr_base[s])
                noise = rng.normal(0.0, base * 0.15)
                degrade = max(0.0, base * (1.0 + stress) + noise)
                health[s] = clamp(health[s] - degrade, 0.0, 1.0)

            # Brake wear accumulates with miles and stop-go
            st["brake_wear"] = clamp(st["brake_wear"] + miles_today / 180_000.0 + 0.0007 * stop_go, 0.0, 1.0)

            # Telematics generation (subsystem-linked signals)
            h_cool = health["cooling"]
            h_aft = health["aftertreatment"]
            h_drv = health["drivetrain"]
            h_ele = health["electrical"]
            h_brk = health["brakes_wheel_end"]

            coolant_avg = 82 + 10 * duty + 0.25 * ambient + 35 * (1 - h_cool) + rng.normal(0, 2.5)
            coolant_max = coolant_avg + 8 + 22 * (1 - h_cool) + rng.normal(0, 3.5)

            egt_max = 380 + 260 * duty + 140 * hilliness + 170 * (1 - h_aft) + rng.normal(0, 25)
            batt_v = 12.5 - 0.9 * (1 - h_ele) - 0.15 * precip - 0.03 * max(0.0, -ambient) + rng.normal(0, 0.08)

            fuel_rate = 22 + 12 * duty + 6 * hilliness + 3 * stop_go + rng.normal(0, 1.2)
            regen_lambda = 0.15 + 1.6 * (1 - h_aft) + 0.6 * stop_go
            regen_count = int(rng.poisson(max(0.0, regen_lambda)))

            idle_pct = clamp((6 + 45 * stop_go + 8 * (1 - h_aft) + rng.normal(0, 4.0)), 0.0, 95.0)

            oil_temp = 88 + 6 * duty + 12 * (1 - h_drv) + rng.normal(0, 2.0)
            oil_press = 320 - 95 * (1 - h_drv) - 18 * duty + rng.normal(0, 10.0)
            oil_press = clamp(oil_press, 80.0, 420.0)

            iat = ambient + 10 + rng.normal(0, 2.0)
            boost = 170 + 55 * duty + rng.normal(0, 10.0)
            boost = clamp(boost, 60.0, 320.0)

            dpf_dp = 5 + 18 * (1 - h_aft) + 3.5 * regen_count + rng.normal(0, 1.0)
            dpf_dp = clamp(dpf_dp, 1.0, 80.0)

            trans_temp = 78 + 22 * duty + 18 * hilliness + 32 * (1 - h_drv) + rng.normal(0, 3.0)
            vib = 0.35 + 1.8 * (1 - h_drv) + 0.7 * (1 - h_brk) + rng.normal(0, 0.12)
            vib = clamp(vib, 0.1, 6.0)

            brake_wear = st["brake_wear"]

            # SPC-like alert (simple, used as evidence and hazard modifier)
            spc_alert = int((coolant_max > 105.0) or (egt_max > 820.0) or (batt_v < 11.7) or (dpf_dp > 35.0))

            # DTC simulation
            dtc_rate = base_dtc_rate + 0.55 * (1.0 - min(health.values())) + 0.15 * spc_alert
            dtc_rate = clamp(dtc_rate, 0.0, 0.90)

            dtc_count_today = 0
            if rng.random() < dtc_rate:
                dtc_count_today = int(rng.integers(1, max_codes + 1))
                # pick subsystems proportional to (1 - health)
                weights = np.array([(1.0 - health[s]) + 1e-3 for s in subsystems], dtype=float)
                weights = weights / weights.sum()

                for _ in range(dtc_count_today):
                    s = str(rng.choice(subsystems, p=weights))
                    code = str(rng.choice(dtc_catalog[s]))
                    severity = int(rng.choice([1, 2, 3], p=[0.60, 0.30, 0.10]))
                    count = int(rng.integers(1, 6))
                    ts = dt + timedelta(minutes=int(rng.integers(0, 24 * 60)))

                    dtc_rows.append(
                        {
                            "timestamp": ts.isoformat(timespec="seconds"),
                            "vehicle_id": vid,
                            "dtc_code": code,
                            "subsystem": s,
                            "severity": severity,
                            "count": count,
                        }
                    )

                st["last_dtc_day_index"] = day_idx

            dtc_recent = 1.0 if (day_idx - int(st["last_dtc_day_index"]) <= 7) else 0.0

            # Failure hazard (hybrid)
            hazard_by_sub = {}
            for s in subsystems:
                rf = alpha_health * (1.0 - float(health[s])) + alpha_dtc_recent * dtc_recent + alpha_spc_alert * float(spc_alert)
                hazard_by_sub[s] = base_hazard * float(np.exp(rf))

            total_hazard = float(sum(hazard_by_sub.values()))
            p_fail_today = 1.0 - float(np.exp(-total_hazard))  # convert hazard to probability
            failed_today = bool(rng.random() < p_fail_today)

            if failed_today:
                # choose failing subsystem proportional to hazard
                hz_arr = np.array([hazard_by_sub[s] for s in subsystems], dtype=float)
                hz_arr = hz_arr / hz_arr.sum()
                fail_sub = str(rng.choice(subsystems, p=hz_arr))

                downtime_days = int(rng.integers(1, 8))
                parts_lead = int(rng.integers(0, 6))

                wo_rows.append(
                    {
                        "open_date": to_date_str(dt),
                        "close_date": to_date_str(dt + timedelta(days=downtime_days)),
                        "vehicle_id": vid,
                        "subsystem": fail_sub,
                        "action": action_for_subsystem(fail_sub),
                        "parts_lead_time_days": parts_lead,
                        "downtime_days": downtime_days,
                        "notes": notes_for_subsystem(fail_sub),
                    }
                )

                breakdown_events.append((vid, dt, fail_sub))

                # put vehicle in shop for downtime (starting next day; today remains with anomaly)
                st["in_shop_until"] = dt + timedelta(days=max(0, downtime_days - 1))

                # repair resets that subsystem health
                health[fail_sub] = clamp(rng.normal(0.98, 0.02), 0.90, 1.0)

            # Record vehicle_day row
            row = {
                "date": to_date_str(dt),
                "vehicle_id": vid,
                "route_type": route_type,
                "hilliness_index": hilliness,
                "stop_go_index": stop_go,
                "ambient_temp_c": ambient,
                "precipitation": precip,
                "duty_cycle": duty,
                "miles_today": miles_today,
                "mileage_total": st["mileage_total"],
                # telematics
                "coolant_temp_avg_c": float(coolant_avg),
                "coolant_temp_max_c": float(coolant_max),
                "egt_max_c": float(egt_max),
                "battery_v_min": float(batt_v),
                "fuel_rate_avg_lph": float(fuel_rate),
                "regen_count": int(regen_count),
                "idle_pct": float(idle_pct),
                "oil_temp_avg_c": float(oil_temp),
                "oil_pressure_min_kpa": float(oil_press),
                "intake_air_temp_avg_c": float(iat),
                "boost_pressure_avg_kpa": float(boost),
                "dpf_delta_p_avg_kpa": float(dpf_dp),
                "trans_temp_avg_c": float(trans_temp),
                "vibration_rms": float(vib),
                "brake_wear_index": float(brake_wear),
                # extras (useful for later features/evidence)
                "dtc_count_today": int(dtc_count_today),
                "spc_alert_today": int(spc_alert),
                # labels added later
                "breakdown_7d": 0,
                "breakdown_30d": 0,
                "subsystem_label": "",
            }
            vehicle_rows.append(row)

    # Labeling: next breakdown within horizons
    df_vehicle = pd.DataFrame(vehicle_rows)
    df_vehicle["date_dt"] = pd.to_datetime(df_vehicle["date"])

    # Build per-vehicle sorted events
    events_by_vehicle: Dict[str, List[Tuple[datetime, str]]] = {}
    for vid, dt, sub in breakdown_events:
        events_by_vehicle.setdefault(vid, []).append((dt, sub))
    for vid in events_by_vehicle:
        events_by_vehicle[vid] = sorted(events_by_vehicle[vid], key=lambda x: x[0])

    # compute labels per vehicle efficiently (small sizes; simple loop is fine)
    breakdown_7d = []
    breakdown_30d = []
    subsystem_label = []

    for vid, group in df_vehicle.sort_values(["vehicle_id", "date_dt"]).groupby("vehicle_id", sort=False):
        ev = events_by_vehicle.get(vid, [])
        j = 0
        for _, r in group.iterrows():
            d = r["date_dt"].to_pydatetime()
            # advance to next event at/after date
            while j < len(ev) and ev[j][0] < d:
                j += 1
            if j >= len(ev):
                breakdown_7d.append(0)
                breakdown_30d.append(0)
                subsystem_label.append("")
                continue

            next_dt, next_sub = ev[j]
            delta_days = (next_dt - d).days
            breakdown_7d.append(1 if (0 <= delta_days <= 7) else 0)
            breakdown_30d.append(1 if (0 <= delta_days <= 30) else 0)
            subsystem_label.append(next_sub if (0 <= delta_days <= 30) else "")

    df_vehicle = df_vehicle.sort_values(["vehicle_id", "date_dt"]).reset_index(drop=True)
    df_vehicle["breakdown_7d"] = breakdown_7d
    df_vehicle["breakdown_30d"] = breakdown_30d
    df_vehicle["subsystem_label"] = subsystem_label
    df_vehicle = df_vehicle.drop(columns=["date_dt"])

    df_dtc = pd.DataFrame(dtc_rows) if len(dtc_rows) else pd.DataFrame(
        columns=["timestamp", "vehicle_id", "dtc_code", "subsystem", "severity", "count"]
    )
    df_wo = pd.DataFrame(wo_rows) if len(wo_rows) else pd.DataFrame(
        columns=[
            "open_date",
            "close_date",
            "vehicle_id",
            "subsystem",
            "action",
            "parts_lead_time_days",
            "downtime_days",
            "notes",
        ]
    )

    return df_vehicle, df_dtc, df_wo


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)

    df_vehicle, df_dtc, df_wo = generate(cfg)

    raw_dir = Path(cfg["paths"]["data_raw_dir"])
    ensure_dir(raw_dir)

    p_vehicle = raw_dir / "vehicle_day.csv"
    p_dtc = raw_dir / "dtc_event.csv"
    p_wo = raw_dir / "work_order.csv"

    df_vehicle.to_csv(p_vehicle, index=False)
    df_dtc.to_csv(p_dtc, index=False)
    df_wo.to_csv(p_wo, index=False)

    # Console summary
    n_breakdowns = int(df_wo.shape[0])
    n_dtc = int(df_dtc.shape[0])
    n_rows = int(df_vehicle.shape[0])
    vehicles = df_vehicle["vehicle_id"].nunique()
    days = df_vehicle["date"].nunique()

    print("=== Synthetic data generated ===")
    print(f"vehicle_day rows: {n_rows}  (vehicles={vehicles}, days={days})")
    print(f"dtc_event rows:   {n_dtc}")
    print(f"work_order rows:  {n_breakdowns}")
    if n_rows > 0:
        rate7 = df_vehicle["breakdown_7d"].mean()
        rate30 = df_vehicle["breakdown_30d"].mean()
        print(f"label prevalence breakdown_7d:  {rate7:.4f}")
        print(f"label prevalence breakdown_30d: {rate30:.4f}")
    print(f"Wrote: {p_vehicle}")
    print(f"Wrote: {p_dtc}")
    print(f"Wrote: {p_wo}")


if __name__ == "__main__":
    main()