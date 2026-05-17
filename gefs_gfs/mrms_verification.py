from datetime import datetime, timedelta
from pathlib import Path

import numpy as np


ARCHIVE_DIRNAME = "verification_archive"


def _utc_tag(dt_value):
    return dt_value.strftime("%Y%m%d_%H")


def _utc_iso(dt_value):
    return dt_value.strftime("%Y-%m-%dT%H:%M:%SZ")


def _run_key(run_date, run_hour):
    return f"{run_date}_{run_hour}"


def _forecast_valid_end(run_date, run_hour, forecast_hour):
    run_datetime = datetime.strptime(f"{run_date} {run_hour}", "%Y%m%d %H")
    return run_datetime + timedelta(hours=int(forecast_hour))


def archive_forecast_window(
    base_dir,
    model_key,
    run_date,
    run_hour,
    forecast_hour,
    step_str,
    avg_data,
    lats,
    lons,
    accumulation_hours=6,
):
    """Save forecast fields so they can be verified later when MRMS becomes available."""
    base_path = Path(base_dir)
    archive_root = base_path / ARCHIVE_DIRNAME
    run_key = _run_key(run_date, run_hour)
    valid_end = _forecast_valid_end(run_date, run_hour, forecast_hour)
    valid_start = valid_end - timedelta(hours=accumulation_hours)
    valid_tag = _utc_tag(valid_end)

    six_hour_dir = archive_root / "six_hour" / model_key / run_key
    six_hour_dir.mkdir(parents=True, exist_ok=True)

    six_hour_path = six_hour_dir / f"valid_{valid_tag}.npz"
    accum_mm = np.asarray(avg_data, dtype=np.float32) * float(accumulation_hours)
    np.savez_compressed(
        six_hour_path,
        avg_prate_mmhr=np.asarray(avg_data, dtype=np.float32),
        accum_mm=accum_mm,
        latitude=np.asarray(lats),
        longitude=np.asarray(lons),
        model_key=model_key,
        run_date=run_date,
        run_hour=run_hour,
        run_key=run_key,
        forecast_hour=int(forecast_hour),
        step_str=step_str,
        accumulation_hours=int(accumulation_hours),
        valid_start_utc=_utc_iso(valid_start),
        valid_end_utc=_utc_iso(valid_end),
    )

    twenty_four_hour_path = _build_twenty_four_hour_archive(
        archive_root=archive_root,
        model_key=model_key,
        run_key=run_key,
        valid_end=valid_end,
    )

    return six_hour_path, twenty_four_hour_path


def _build_twenty_four_hour_archive(archive_root, model_key, run_key, valid_end):
    six_hour_dir = archive_root / "six_hour" / model_key / run_key
    twenty_four_dir = archive_root / "twenty_four_hour" / model_key / run_key
    twenty_four_dir.mkdir(parents=True, exist_ok=True)

    required_ends = [valid_end - timedelta(hours=offset) for offset in (18, 12, 6, 0)]
    required_paths = [six_hour_dir / f"valid_{_utc_tag(dt_value)}.npz" for dt_value in required_ends]
    if not all(path.exists() for path in required_paths):
        return None

    accum_parts = []
    latitude = None
    longitude = None
    forecast_hours = []
    for path in required_paths:
        with np.load(path, allow_pickle=False) as saved:
            accum_parts.append(saved["accum_mm"].astype(np.float32))
            if latitude is None:
                latitude = saved["latitude"]
                longitude = saved["longitude"]
            forecast_hours.append(int(saved["forecast_hour"]))

    accum_24h = np.sum(accum_parts, axis=0, dtype=np.float32)
    valid_start = valid_end - timedelta(hours=24)
    output_path = twenty_four_dir / f"valid_{_utc_tag(valid_end)}.npz"
    np.savez_compressed(
        output_path,
        accum_24h_mm=accum_24h,
        latitude=np.asarray(latitude),
        longitude=np.asarray(longitude),
        model_key=model_key,
        run_key=run_key,
        window_hours=24,
        source_interval_hours=6,
        forecast_hours=np.asarray(forecast_hours, dtype=np.int32),
        valid_start_utc=_utc_iso(valid_start),
        valid_end_utc=_utc_iso(valid_end),
    )
    return output_path