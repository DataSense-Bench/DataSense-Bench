"""Convert the official OpenThoughts-Agent-SFT-10K shards without reordering rows."""
import argparse
from pathlib import Path
import tempfile

import pyarrow as pa
import pyarrow.parquet as pq


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Downloaded dataset directory")
    parser.add_argument("--output", type=Path, required=True, help="New output directory")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; choose a new directory")
    source = args.source / "data" if (args.source / "data").is_dir() else args.source
    shards = sorted(source.glob("train-*.parquet"))
    if not shards:
        parser.error(f"No train-*.parquet shards in {source}")
    message_type = pa.list_(pa.struct([("role", pa.string()), ("content", pa.string())]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.output.parent) as temp:
        stage = Path(temp) / "prepared"
        stage.mkdir()
        count, tail = 0, []
        schema = pa.schema([("messages", message_type)])
        with pq.ParquetWriter(stage / "train_10k.parquet", schema) as writer:
            for shard in shards:
                for batch in pq.ParquetFile(shard).iter_batches(batch_size=64, columns=["conversations"]):
                    rows = []
                    for conversation in batch.column(0).to_pylist():
                        if not isinstance(conversation, list) or not conversation:
                            raise ValueError(f"Empty or invalid conversation at row {count + len(rows)}")
                        messages = []
                        for message in conversation:
                            role, content = message.get("role"), message.get("content")
                            if role not in {"system", "user", "assistant", "tool"} or not isinstance(content, str):
                                raise ValueError(f"Invalid role/content at row {count + len(rows)}")
                            messages.append({"role": role, "content": content})
                        if not any(m["role"] == "assistant" for m in messages):
                            raise ValueError(f"No assistant response at row {count + len(rows)}")
                        rows.append(messages)
                    writer.write_table(pa.table({"messages": pa.array(rows, type=message_type)}))
                    count += len(rows)
                    tail = (tail + rows)[-200:]
        if count != 10000:
            raise ValueError(f"Expected 10,000 trajectories; found {count}. Download every shard.")
        # Preserve the original recipe's loader input. Evaluation during SFT is disabled;
        # these last 200 training rows are not a held-out performance benchmark.
        pq.write_table(pa.table({"messages": pa.array(tail, type=message_type)}), stage / "validation.parquet")
        stage.rename(args.output)
    print(f"Prepared {count} trajectories in original order: {args.output}")


if __name__ == "__main__":
    main()
