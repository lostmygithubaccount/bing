# imports
import mlflow
import argparse

import dask.dataframe as dd

from distributed import Client

if __name__ == "__main__":

    # argparse setup
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str)
    parser.add_argument("--output", type=str, default="cody/dask/out/train.parquet")
    parser.add_argument("--npartitions", type=int, default=256)
    args = parser.parse_args()

    # distributed setup
    print("initializing...")
    c = Client()
    print(c)

    # get data
    from azureml.core import Run

    run = Run.get_context()
    ws = run.experiment.workspace
    # ds = ws.get_default_datastore()
    ds = ws.datastores["aml1pds"]
    container_name = ds.container_name
    storage_options = {"account_name": ds.account_name, "account_key": ds.account_key}

    print(ws)
    print(ds)
    print(f"az://{container_name}/{args.input}")

    # read into dataframes
    print("creating dataframes...")
    df = dd.read_table(
        f"az://{container_name}/{args.input}",
        storage_options=storage_options,
        blocksize=5e9,
    )
    print(df)

    # data processing
    print("processing data...")

    # writing data
    print("writing data...")
    print(c)
    df.repartition(npartitions=args.npartitions).to_parquet(
        f"az://{container_name}/{args.output}",
        storage_options=storage_options,
        # overwrite=True,
        write_index=False,
    )
