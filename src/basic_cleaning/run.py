
#!/usr/bin/env python
"""
Performs basic cleaning on the data and save the results in Weights & Biases
"""
import argparse
import logging
import wandb
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    logger.info("Intializing Weights & Biases run")
    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    # Download input artifact. This will also log that this script is using this
    # particular version of the artifact
    # artifact_local_path = run.use_artifact(args.input_artifact).file()

    # use logger at every step
    logger.info(f"Loading data from {args.input_artifact}")
    artifact_local_path = run.use_artifact(args.input_artifact).file()

    logger.info(f"Reading data from {artifact_local_path}")
    df = pd.read_csv(artifact_local_path)
    
    logger.info("Dropping duplicates")
    df = df.drop_duplicates()

    logger.info("Dropping rows with price in the wrong range")
    df = df[(df["price"] >= args.min_price) & (df["price"] <= args.max_price)]

    logger.info("Converting last_review to datetime")
    df["last_review"] = pd.to_datetime(df["last_review"])
    df = df.dropna(subset=['last_review'])

    logger.info("Filtering data by geographical coordinates")
    idx = df['longitude'].between(-74.25, -73.50) & df['latitude'].between(40.5, 41.2)
    df = df[idx].copy()


    logger.info("Saving cleaned data to artifact")
    df.to_csv("clean_sample.csv", index=False)

    # output artifact to W&B
    artifact = wandb.Artifact(
        args.output_artifact,
        type=args.output_type,
        description=args.output_description,
    )

    artifact.add_file("clean_sample.csv")
    run.log_artifact(artifact)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="This step cleans the data")

    parser.add_argument(
        "--input_artifact",
        type=str,
        help="Input artifact name",
        required=True
    )

    parser.add_argument(
        "--output_artifact",
        type=str,
        help="Output artifact name",
        required=True
    )

    parser.add_argument(
        "--output_type",
        type=str,
        help="Output artifact type",
        required=True
    )

    parser.add_argument(
        "--output_description",
        type=str,
        help="Output artifact description",
        required=True
    )

    parser.add_argument(
        "--min_price",
        type=float,
        help="Minimum price",
        required=True
    )

    parser.add_argument(
        "--max_price",
        type=float,
        help="Maximum price",
        required=True
    )

    args = parser.parse_args()
    go(args)
