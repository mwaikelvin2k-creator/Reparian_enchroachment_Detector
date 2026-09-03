"""
Combines preprocessing.py and modeling.py into full end-to-end runs, single
or multi-location.
"""

from preprocessing import run_preprocessing
from modeling import run_modeling


def run_full_pipeline(place_name: str, gee_project: str, **kwargs) -> dict:
    """Convenience wrapper: run_preprocessing() then run_modeling(). Prefer
    calling them separately when debugging — this is for the case where
    preprocessing is already known to produce a good training table."""
    prep = run_preprocessing(place_name, gee_project, **kwargs)
    model_results = run_modeling(
        prep["training_table"], prep["feature_cols"], prep["buffer_m"], prep["output_dir"]
    )
    return {**prep, **model_results}


def run_multiple_locations(locations: list[dict], gee_project: str,
                            stop_on_error: bool = False) -> dict:
    """Runs run_full_pipeline once per location. Each dict in `locations`
    supplies that location's arguments (place_name, gee_buildings_asset,
    output_dir, etc.) — anything accepted by run_full_pipeline except
    gee_project, which is shared across all locations.

    A failure in one location is recorded and skipped rather than stopping
    the batch, unless stop_on_error=True.

    Returns {place_name: {"status": "ok"/"failed", "results"/"error": ...}}
    """
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
    from preprocessing import run_preprocessing
    from modeling import run_modeling

    # Preprocessing and modeling separately — inspect prep["training_table"]
    # before committing to a model fit.
    prep = run_preprocessing(
        place_name="Kasarani, Nairobi, Kenya",
        gee_project="causal-bus-404912",
        buffer_m=30,
        gee_buildings_asset="projects/sat-io/open-datasets/MSBuildings/Kenya",
        output_dir="./output/kasarani",
    )
    print("Training table rows:", len(prep["training_table"]))

    if len(prep["training_table"]) > 0:
        model_results = run_modeling(
            prep["training_table"], prep["feature_cols"], prep["buffer_m"], prep["output_dir"]
        )
        print(model_results["metrics"])

    # Or both steps in one call:
    # run_full_pipeline(
    #     place_name="Kasarani, Nairobi, Kenya",
    #     gee_project="causal-bus-404912",
    #     buffer_m=30,
    #     gee_buildings_asset="projects/sat-io/open-datasets/MSBuildings/Kenya",
    #     output_dir="./output/kasarani",
    # )

    # Multiple locations in one call — each dict supplies that location's
    # own arguments; gee_project is shared.
    # run_multiple_locations(
    #     locations=[
    #         dict(place_name="Kasarani, Nairobi, Kenya", buffer_m=30,
    #              gee_buildings_asset="projects/sat-io/open-datasets/MSBuildings/Kenya",
    #              output_dir="./output/kasarani"),
    #         dict(place_name="Ibadan, Nigeria", buffer_m=30,
    #              gee_buildings_asset="projects/sat-io/open-datasets/MSBuildings/Nigeria",
    #              output_dir="./output/ibadan"),
    #     ],
    #     gee_project="causal-bus-404912",
    # )
