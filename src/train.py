import os
import logging

from dotenv import load_dotenv
import s3fs

logger = logging.getLogger(__name__)
load_dotenv(override=True)

# ==============================
#             Init
# ==============================
logger.info("Initialisation...")
fs = s3fs.S3FileSystem(
    client_kwargs={'endpoint_url': "https://" + os.environ["AWS_S3_ENDPOINT"]},
    key=os.environ["AWS_ACCESS_KEY_ID"],
    secret=os.environ["AWS_SECRET_ACCESS_KEY"],
    token=os.environ["AWS_SESSION_TOKEN"]
)


