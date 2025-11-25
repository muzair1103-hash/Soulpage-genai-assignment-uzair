from logger import logger
import psycopg
from psycopg import AsyncConnection, Connection, Error
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

CHAT_HISTORY_DB_URI_TEMPLATE = (
    "postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_DATABASE}"
)

DB_DATABASE = "baseer_gpt_db"
DB_HOST = "localhost"
DB_PORT = "5432"
DB_USER = "postgres"
DB_PASSWORD = "12345"
DB_PASSWORD = "12345"
CHAT_HISTORY_DB_URI = CHAT_HISTORY_DB_URI_TEMPLATE.format(
    DB_USER=DB_USER,
    DB_PASSWORD=DB_PASSWORD,
    DB_HOST=DB_HOST,
    DB_PORT=DB_PORT,
    DB_DATABASE=DB_DATABASE,
)


class Memory:
    memory = None

    @classmethod
    async def initialize_memory(cls):
        if cls.memory is None:
            cls.create_database()
            cls.verify_connection()

            pool = await AsyncConnection.connect(CHAT_HISTORY_DB_URI, autocommit=True)
            cls.memory = AsyncPostgresSaver(pool)  # type: ignore

            await cls.memory.setup()

        return cls.memory

    @staticmethod
    def create_database() -> Connection:
        try:
            with psycopg.connect(
                CHAT_HISTORY_DB_URI_TEMPLATE.format(
                    DB_USER=DB_USER,
                    DB_PASSWORD=DB_PASSWORD,
                    DB_HOST=DB_HOST,
                    DB_PORT=DB_PORT,
                    DB_DATABASE=DB_USER,
                ),
                autocommit=True,
            ) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT 1 FROM pg_catalog.pg_database WHERE datname = %s",
                        (DB_DATABASE,),
                    )
                    exists = cur.fetchone()
                    if not exists:
                        cur.execute(
                            f"CREATE DATABASE {DB_DATABASE}"  # type:ignore
                        )
                        cur.execute(
                            f"GRANT CONNECT ON DATABASE {DB_DATABASE} TO postgres"  # type:ignore
                        )
                        logger.info("Database created: %s", DB_DATABASE)
                    else:
                        logger.info("Database exists: %s", DB_DATABASE)
                return conn
        except Error as e:
            logger.error("Database creation failed: %s", str(e))
            raise

    @staticmethod
    def verify_connection() -> Connection:
        try:
            conn = psycopg.connect(
                CHAT_HISTORY_DB_URI_TEMPLATE.format(
                    DB_USER=DB_USER,
                    DB_PASSWORD=DB_PASSWORD,
                    DB_HOST=DB_HOST,
                    DB_PORT=DB_PORT,
                    DB_DATABASE=DB_DATABASE,
                )
            )
            return conn
        except Error as e:
            logger.error("Connection failed: %s", str(e))
            raise
