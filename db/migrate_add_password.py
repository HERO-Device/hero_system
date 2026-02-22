"""
Migration: Add password column to users table
Run once: python db/migrate_add_password.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import psycopg2

DB_CONFIG = {
    'host':     'localhost',
    'port':     5432,
    'user':     'postgres',
    'password': 'pgdbadmin',
    'dbname':   'hero_db',
}

def run():
    print("\n=== Migration: Add password column to users ===\n")

    try:
        conn = psycopg2.connect(**DB_CONFIG)
        conn.autocommit = True
        cur = conn.cursor()

        # Check if column already exists
        cur.execute("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name='users' AND column_name='password';
        """)

        if cur.fetchone():
            print("✓ password column already exists, nothing to do.")
        else:
            cur.execute("ALTER TABLE users ADD COLUMN password VARCHAR;")
            print("✓ Added password column to users table.")

        cur.close()
        conn.close()
        print("\nMigration complete.\n")

    except Exception as e:
        print(f"\n✗ Migration failed: {e}\n")
        sys.exit(1)

if __name__ == '__main__':
    run()
    
