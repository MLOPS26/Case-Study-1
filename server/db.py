import duckdb
from utils.consts import DB_NAME

def init_db():
    conn = duckdb.connect(DB_NAME)

    conn.execute("""
        CREATE SEQUENCE IF NOT EXISTS user_id_seq;
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER DEFAULT nextval('user_id_seq') PRIMARY KEY,
            username VARCHAR UNIQUE NOT NULL,
            email VARCHAR UNIQUE NOT NULL,
            password_hash VARCHAR NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    conn.close()


def get_db_connection():
    return duckdb.connect(DB_NAME)


if __name__ == "__main__":
    init_db()
    print("Database initialized.")
