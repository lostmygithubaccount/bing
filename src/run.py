# imports
import mlflow
import argparse

import pandas as pd
import lightgbm as lgbm
import dask.dataframe as dd

from distributed import Client
from dask_mpi import initialize

## TODO: remove
from dask_lightgbm import LGBMRegressor

# argparse setup
parser = argparse.ArgumentParser()
parser.add_argument("--filename", type=str)
parser.add_argument("--boosting", type=str, default="gbdt")
parser.add_argument("--num_iterations", type=int, default=100)
parser.add_argument("--learning_rate", type=float, default=0.1)
parser.add_argument("--num_leaves", type=int, default=31)
parser.add_argument("--nodes", type=int, default=10)
parser.add_argument("--cpus", type=int, default=16)
args = parser.parse_args()

# distributed setup
print("initializing...")
initialize(nthreads=args.cpus)
c = Client()
print(c.dashboard_link)
print(c)

# get data
from azureml.core import Run

run = Run.get_context()
ws = run.experiment.workspace
ds = ws.get_default_datastore()
container_name = ds.container_name
storage_options = {"account_name": ds.account_name, "account_key": ds.account_key}

# read into dataframes
print("creating dataframes...")
df = dd.read_csv(
    f"az://{container_name}/{args.filename}", storage_options=storage_options, sep="\t"
).repartition(npartitions=args.nodes * args.cpus * 2)
print(df)

# data processing
print("processing data...")
X = df.drop("label", axis=1).values.persist()
y = df["label"].values.persist()

# train lightgbm
print("training lightgbm...")
print(c)

params = {
    "objective": "regression",
    "boosting": args.boosting,
    "num_iterations": args.num_iterations,
    "learning_rate": args.learning_rate,
    "num_leaves": args.num_leaves,
}

model = LGBMRegressor(**params)
model.fit(X, y)
print(model)

# predict on test data
print("making predictions...")
print(c)
