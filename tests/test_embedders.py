import os

from embeddings.basic_embedder import BasicEmbedder
from embeddings.async_embedder import AsyncEmbedder

LABELS = [
    "Je suis autoentrepreuneur dans le conseil pour la transition écologique",
    "Location de logements meublés non professionnel",
    "Bonjour je cherche à obtenir mon numéro de SIRET",
    "Je fais de la flute à Arcachon",
    "Elevage de capibaras",
    "Transport et livraison de colis de petite et moyenne taille par voie routière",
    "Je ne souhaite pas donner mon activité"
]


def test_basic():
    embedder = BasicEmbedder(
        base_url=os.environ["EMBEDDING_API_BASE_URL"],
        model=os.getenv("EMBEDDING_API_MODEL", None),
        api_key=os.getenv("EMBEDDING_API_KEY", "")
    )

    embeddings = embedder.embed(LABELS)

    assert embeddings is not None, "No embedding has been generated"


def test_async():
    embedder = AsyncEmbedder(
        base_url=os.environ["EMBEDDING_API_BASE_URL"],
        model=os.getenv("EMBEDDING_API_MODEL", None),
        api_key=os.getenv("EMBEDDING_API_KEY", "")
    )

    embeddings = embedder.embed(LABELS)

    assert embeddings is not None, "No embedding has been generated"
