"""
Combines preprocessing.py and modeling.py into full end-to-end runs, for
one location or many, across three tasks: encroachment (building-only),
building-vs-not-building detection, and four-class land-cover classification.
"""

from preprocessing import (
    run_preprocessing,
    run_building_detection_preprocessing,
    run_land_cover_preprocessing,
)
from modeling import run_modeling


def run_full_pipeline(place_name: str, gee_project: str,
                       n_negative_samples: int | None = None,
                       land_cover_grid_spacing_m: float = 200,
                       **kwargs) -> dict:
    """Runs acquisition once, then all three tasks against it. kwargs are
    passed through to run_preprocessing (buffer_m, gee_buildings_asset,
    output_dir, etc.)."""
    prep = run_preprocessing(place_name, gee_project, **kwargs)

    encroachment_results = run_modeling(
        prep["training_table"], prep["feature_cols"], prep["output_dir"],
        label_col="encroachment", model_name="encroachment_rf", task_type="binary",
    )

    detection_prep = run_building_detection_preprocessing(prep, n_negative_samples=n_negative_samples)
    detection_results = run_modeling(
        detection_prep["training_table"], detection_prep["feature_cols"], detection_prep["output_dir"],
        label_col="is_building", model_name="building_detector_rf", task_type="binary",
    )

    land_cover_prep = run_land_cover_preprocessing(prep, grid_spacing_m=land_cover_grid_spacing_m)
    land_cover_results = run_modeling(
        land_cover_prep["training_table"], land_cover_prep["feature_cols"], land_cover_prep["output_dir"],
        label_col="land_cover", model_name="land_cover_rf", task_type="multiclass",
    )

    return {
        "preprocessing": prep,
        "encroachment": encroachment_results,
        "building_detection": detection_results,
        "land_cover": land_cover_results,
    }


def run_multiple_locations(locations: list[dict], gee_project: str,
                            stop_on_error: bool = False) -> dict:
    """Runs run_full_pipeline once per location. Each dict in locations
    supplies that location's arguments; gee_project is shared across all.
    A failure in one location is recorded and skipped rather than stopping
    the batch, unless stop_on_error=True."""
    import ee
    ee.Initialize(project=gee_project)

    summary = {}
    for cfg in locations:
        place_name = cfg["place_name"]
        print(f"\n=== {place_name} ===")
        try:
            results = run_full_pipeline(gee_project=gee_project, **cfg)
            summary[place_name] = {"status": "ok", "results": results}
        except Exception as exc:
            print(f"FAILED: {place_name} — {exc}")
            summary[place_name] = {"status": "failed", "error": str(exc)}
            if stop_on_error:
                raise

    ok = sum(1 for v in summary.values() if v["status"] == "ok")
    print(f"\n{ok}/{len(locations)} locations completed successfully")
    return summary


if __name__ == "__main__":
    # Nairobi as one of potentially several locations. Each dict supplies
    # that location's own arguments; gee_project is shared.
    run_multiple_locations(
        locations=[
            dict(
                place_name="Nairobi, Kenya",
                buffer_m=30,
                gee_buildings_asset="projects/sat-io/open-datasets/MSBuildings/Kenya",
                gee_buildings_local_path="nairobi_buildings_export.geojson",
                raster_local_path="nairobi_sentinel_composite.tif",
                output_dir="./output/nairobi",
                land_cover_grid_spacing_m=200,
            ),
        ],
        gee_project="causal-bus-404912",
    )