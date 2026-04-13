"""
Clean the dataset to remove several kinds of labels:
  - Truncated ones (of length 141 and 161)
  - Those including numbers
  - Those including "siret" (linked to sentences that do not describe activity)
  - Those ending by " activité de services" (just removing the suffix)
"""

import os

import pyarrow.parquet as pq
import pyarrow as pa
import pyarrow.compute as pc
import s3fs


SUFFIX = " activité de services"
SUFFIX_LEN = len(SUFFIX)


def clean_batch(batch: pa.RecordBatch) -> pa.Table:

    table = pa.Table.from_batches([batch])
    label = table["label"]

    # ===================================
    # 1. Filter of len != 141 et != 161
    # ===================================
    lengths = pc.utf8_length(label)

    mask_len = pc.and_(
        pc.not_equal(lengths, 141),
        pc.not_equal(lengths, 161),
    )

    table = table.filter(mask_len)
    label = table["label"]

    # ==========================================================
    # 2. Remove numbers (more than 2 digits, not a percentage)
    # ==========================================================
    # contains 2 digits
    has_two_digits = pc.match_substring_regex(label, r"\d\d")

    # contains 2 digits followed by a percentage
    has_two_digits_percent = pc.match_substring_regex(label, r"\d\d\s*%")

    # equivalant to \d\d\s*(?!%) using re
    invalid_pattern = pc.and_(
        has_two_digits,
        pc.invert(has_two_digits_percent)
    )

    mask_regex = pc.invert(invalid_pattern)

    table = table.filter(mask_regex)
    label = table["label"]

    # ======================================
    # 3. Remove labels containing "siret"
    # ======================================
    mask_regex = pc.invert(
        pc.match_substring(label, "siret")
    )

    table = table.filter(mask_regex)
    label = table["label"]

    # ==========================================
    # 4. Remove " activité de services" at the end
    # ==========================================
    ends_with_suffix = pc.ends_with(label, SUFFIX)

    trimmed = pc.utf8_slice_codeunits(
        label,
        start=0,
        stop=-SUFFIX_LEN
    )

    new_label = pc.if_else(
        ends_with_suffix,
        trimmed,
        label
    )

    table = table.set_column(
        table.schema.get_field_index("label"),
        "label",
        new_label
    )

    return table


def main():

    OLD_PATH = 's3://mateom/graal/embeddings/NAF2025/original_train_corrected.parquet'
    NEW_PATH = 's3://mateom/graal/embeddings/NAF2025/original_train_cleaned.parquet'

    fs = s3fs.S3FileSystem(
        client_kwargs={'endpoint_url': 'https://' + 'minio.lab.sspcloud.fr'},
        key=os.environ["AWS_ACCESS_KEY_ID"],
        secret=os.environ["AWS_SECRET_ACCESS_KEY"],
        token=os.environ["AWS_SESSION_TOKEN"]
    )

    with fs.open(OLD_PATH) as f:
        reader = pq.ParquetFile(f)

        writer = None

        for batch in reader.iter_batches(
            batch_size=50000,
            columns=['code', 'label', 'embedding'],
            use_threads=True
        ):

            cleaned_table = clean_batch(batch)

            if writer is None:
                writer = pq.ParquetWriter(
                    NEW_PATH,
                    cleaned_table.schema,
                    compression='snappy'
                )

            writer.write_table(cleaned_table)
            print(f"{cleaned_table.num_rows} lines have been written...")

        if writer:
            writer.close()


if __name__ == "__main__":
    main()
