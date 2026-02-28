"""
HERO System - Add Patient
CLI script for clinicians to register new patients before a session.

Usage:
    python add_patient.py
"""

import sys
import logging
from datetime import datetime

# Make sure hero_system root is on the path
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db.db_access import HeroDB

logging.basicConfig(level=logging.WARNING)  # Suppress SQLAlchemy noise


def prompt(label: str, required: bool = True, secret: bool = False) -> str:
    """
    Prompt the clinician for input, re-asking if a required field is left blank.
    Args:
        label:    Display label shown in the terminal prompt.
        required: If True, re-prompts until a non-empty value is entered.
        secret:   If True, uses getpass to hide input (for passwords).
    Returns:
        Stripped input string, or None if not required and left blank.
    """
    while True:
        if secret:
            import getpass
            value = getpass.getpass(f"  {label}: ").strip()
        else:
            value = input(f"  {label}: ").strip()

        if value:
            return value
        elif not required:
            return None
        else:
            print(f"  ⚠ {label} is required, please try again.")


def parse_date(date_str: str):
    """
    Parse a date of birth string into a datetime object.
    Accepts DD/MM/YYYY or YYYY-MM-DD formats.
    Args:
        date_str: Raw date string from user input.
    Returns:
        datetime object if parsing succeeds, None otherwise.
    """
    if not date_str:
        return None
    for fmt in ('%d/%m/%Y', '%Y-%m-%d'):
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    print(f"  ⚠ Could not parse date '{date_str}' — skipping date of birth.")
    return None


def main():
    """
    CLI entry point for registering a new patient.

    Prompts the clinician for patient details, checks the username
    is not already taken, confirms the summary, then writes the
    new User to the database.

    Returns:
        None.
    """
    print()
    print("=" * 50)
    print("  HERO System — Add Patient")
    print("=" * 50)
    print()

    # Connect to DB
    try:
        db = HeroDB()
    except Exception as e:
        print(f"\n✗ Could not connect to database: {e}")
        print("  Make sure PostgreSQL is running and hero_db is accessible.")
        sys.exit(1)

    try:
        print("  Enter patient details:\n")

        username  = prompt("Username (e.g. johndoe NO UNDERSCORES/NUMBERS)")
        password  = prompt("Password NO UNDERSCORES/NUMBERS", secret=True)
        full_name = prompt("Full name (e.g. John Doe)")
        email     = prompt("Email (optional)", required=False)
        dob_str   = prompt("Date of birth DD/MM/YYYY (optional)", required=False)
        dob       = parse_date(dob_str)

        # Check username not already taken
        existing = db.get_user(username)
        if existing:
            print(f"\n✗ Username '{username}' already exists.")
            print("  Please choose a different username.")
            sys.exit(1)

        # Confirm
        print()
        print("  Summary:")
        print(f"    Username  : {username}")
        print(f"    Full name : {full_name}")
        print(f"    Email     : {email or '—'}")
        print(f"    DOB       : {dob.strftime('%d/%m/%Y') if dob else '—'}")
        print()

        confirm = input("  Create this patient? (y/n): ").strip().lower()
        if confirm != 'y':
            print("\n  Cancelled.")
            sys.exit(0)

        # Create user
        user = db.create_user(
            username=username,
            password=password,
            full_name=full_name,
            email=email,
            date_of_birth=dob,
        )

        if user:
            print()
            print(f"  ✓ Patient '{username}' created successfully.")
            print(f"  ✓ User ID: {user.user_id}")
            print()
        else:
            print("\n  ✗ Failed to create patient — check logs for details.")
            sys.exit(1)

    finally:
        db.close()


if __name__ == '__main__':
    main()
