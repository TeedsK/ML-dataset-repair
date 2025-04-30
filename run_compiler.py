import psycopg2
import sys
import argparse
from config import DB_SETTINGS
from compiler.compile import FeatureCompiler
import psycopg2.extras

import logging, sys

# COMMENT OR UNCOMMENT THIS IF YOU WANT TO RECEIVE LOGS LIKE DEBUGS OR INFO
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

def main():
    parser = argparse.ArgumentParser(description="Run HoloClean Feature Compiler")
    parser.add_argument(
        '--relax-constraints',
        action='store_true',
        help='Use relaxed denial constraints (generate features) instead of hard factors.'
    )
    parser.add_argument(
        '--no-relax-constraints',
        action='store_false',
        dest='relax_constraints',
        help='Use hard denial constraints (requires factor table implementation).'
    )
    parser.set_defaults(relax_constraints=True)
    args = parser.parse_args()

    conn = None
    try:
        print(f"connecting to database for feature compilation...")
        conn = psycopg2.connect(**DB_SETTINGS)
        psycopg2.extras.register_uuid()
        print("connected!")

        compiler = FeatureCompiler(conn, relax_constraints=args.relax_constraints)
        compiler.compile_all()

        print("\nFeature Compilation Phase Complete")

    except psycopg2.Error as db_err:
        print(f"error: {db_err}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        if conn:
            conn.close()
            print("Database connection closed.")

if __name__ == "__main__":
    main()