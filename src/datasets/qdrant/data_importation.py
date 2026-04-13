from qdrant_client import QdrantClient
import numpy as np

MAX_POINTS_TO_RETRIEVE = 1000000


def count_codes(
    client: QdrantClient,
    collection_name: str
) -> dict:
    """
    Return every code and their number of occurences in the database.
    Format: dict[code] = count
    """
    code_hits = client.facet(
        collection_name=collection_name,
        key="code",
        exact=True,
        limit=1000
    ).hits

    code_list = {hit.value: hit.count for hit in code_hits}

    return code_list


def select_random_points(
    client: QdrantClient,
    collection_name: str,
    size: int,
    filter: dict = None,
    random_state: int = None
) -> list:
    """
    Select random points from qdrant collection with replacement.
    Filters can be applied (e.g., to sample for a given code).
    """
    assert size >= 0, ValueError("Please provide size >= 0")

    if size == 0:
        return []

    rng = np.random.default_rng(random_state)

    # Retrieve id only
    records, _ = client.scroll(
        collection_name=collection_name,
        scroll_filter=filter,
        with_payload=False,
        with_vectors=False,
        limit=MAX_POINTS_TO_RETRIEVE
    )
    ids = [record.id for record in records]

    # Random sampling
    random_ids = rng.choice(ids, size=size, replace=True)

    # Retrieve unique vectors
    random_unique_points = client.retrieve(
        collection_name=collection_name,
        ids=random_ids,
        with_payload=True,
        with_vectors=True
    )

    id_to_points = {}

    for p in random_unique_points:
        id_to_points[p.id] = p

    random_points = [id_to_points[id] for id in random_ids]

    return random_points


def exhaustive_sampling(
    client: QdrantClient,
    collection_name: str,
    size_per_code: int,
    random_state: int = None
) -> list:
    """
    Sample the same number of vectors for each code.
    """

    code_list = count_codes(client, collection_name)

    selected_points = []

    for code in code_list.keys():
        selected_points += select_random_points(
            client=client,
            collection_name=collection_name,
            size=size_per_code,
            filter={
                "must": [
                    {"key": "code", "match": {"value": code}}
                ]
            },
            random_state=random_state
        )

    return selected_points


def fetch_original_points(
    client: QdrantClient,
    collection_name: str,
    size: int,
    min_size_per_code: int,
    random_state: int = None
) -> list:
    """
    Sample the original data while maintaining code exhaustivity.
    """

    exhaustive_points = exhaustive_sampling(
        client=client,
        collection_name=collection_name,
        size_per_code=min_size_per_code,
        random_state=random_state
    )

    remaining_size = size - len(exhaustive_points)

    remaining_points = select_random_points(
        client=client,
        collection_name=collection_name,
        size=remaining_size,
        random_state=random_state
    )

    return exhaustive_points + remaining_points


def select_synthetic_data(
    client: QdrantClient,
    collection_name: str,
    code_list: list,
    random_state: int = None
) -> list:
    """
    Select synthetic data so that it matches each number of codes in code_list.
    """
    selected_points = []

    for code, count in code_list.items():
        selected_points += select_random_points(
            client=client,
            collection_name=collection_name,
            size=count,
            filter={
                "must": [
                    {"key": "code", "match": {"value": code}}
                ]
            },
            random_state=random_state
        )

    return selected_points
