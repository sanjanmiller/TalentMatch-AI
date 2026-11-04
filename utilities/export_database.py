"""
Purpose:
Export candidate information and professional profiles from the
Intelligent Recruitment System's SQLite database into separate files.

Output files:
1. data/exports/candidate_info.csv
   Contains candidate IDs, names, contact information, and timestamps.

2. data/exports/professional_info.json
   Contains structured professional profiles, including skills,
   work experience, projects, education, certifications, and achievements.

The script reads existing database records only. It does not modify
the database or process the original resume files again.
"""

import csv
import json
import sqlite3
from pathlib import Path


# ---------------------------------------------------------
# 1. Define project paths
# ---------------------------------------------------------

# Automatically locate the project's root directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Path to the existing SQLite database.
DATABASE_PATH = (
    PROJECT_ROOT / "storage" / "database" / "recruitment.db"
)

# Folder where the exported files will be stored.
EXPORT_FOLDER = PROJECT_ROOT / "data" / "exports"


# ---------------------------------------------------------
# 2. Export candidate information to CSV
# ---------------------------------------------------------

def export_candidate_info_to_csv(
    connection: sqlite3.Connection,
) -> int:
    """
    Export candidate identity records into a CSV file.

    The CSV file can be opened in Excel or other spreadsheet
    applications. Original resume text is deliberately excluded.

    Returns:
        The number of candidate records exported.
    """

    output_path = EXPORT_FOLDER / "candidate_info.csv"

    # Retrieve candidate identity information from SQLite.
    query = """
        SELECT
            candidate_id,
            full_name,
            email,
            phone,
            created_at,
            updated_at
        FROM candidate_identity
        ORDER BY candidate_id
    """

    cursor = connection.execute(query)

    # Retrieve the database column names for the CSV header.
    column_names = [
        column[0] for column in cursor.description
    ]

    # Fetch all matching candidate records.
    rows = cursor.fetchall()

    # Write the headers and records to the CSV file.
    # UTF-8 with BOM improves compatibility when opening
    # the file directly in Microsoft Excel.
    with output_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.writer(file)

        # Write the column headings as the first row.
        writer.writerow(column_names)

        # Write one row for each candidate.
        writer.writerows(rows)

    print(f"CSV file created: {output_path}")

    return len(rows)


# ---------------------------------------------------------
# 3. Export professional information to JSON
# ---------------------------------------------------------

def export_professional_info_to_json(
    connection: sqlite3.Connection,
) -> int:
    """
    Export structured professional profiles into a JSON file.

    Each profile retains its candidate ID so that it can be linked
    to the corresponding candidate record.

    Returns:
        The number of professional profiles exported.
    """

    output_path = EXPORT_FOLDER / "professional_info.json"

    # Retrieve saved professional profiles from SQLite.
    query = """
        SELECT
            candidate_id,
            profile_json,
            updated_at
        FROM resume_profiles
        ORDER BY candidate_id
    """

    cursor = connection.execute(query)
    rows = cursor.fetchall()

    exported_profiles = []

    for candidate_id, profile_json, updated_at in rows:
        # Convert the JSON text stored in SQLite into a Python
        # dictionary so it can be written as readable JSON.
        profile_data = json.loads(profile_json)

        # Keep the candidate ID as the link between the profile
        # and the candidate's identity record.
        exported_profiles.append(
            {
                "candidate_id": candidate_id,
                "updated_at": updated_at,
                "professional_info": profile_data,
            }
        )

    # Write the professional profiles to a readable JSON file.
    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            exported_profiles,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print(f"JSON file created: {output_path}")

    return len(exported_profiles)


# ---------------------------------------------------------
# 4. Run both exports
# ---------------------------------------------------------

def main() -> None:
    """
    Export candidate information and professional profiles.

    The database must already exist and contain the expected tables.
    """

    # Check that the SQLite database exists before connecting.
    if not DATABASE_PATH.exists():
        raise FileNotFoundError(
            f"Database not found: {DATABASE_PATH}\n"
            "Please ensure the database has been initialized "
            "and candidate records have been saved."
        )

    # Create the exports directory if it does not exist.
    EXPORT_FOLDER.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Open the database and export both sets of records.
    with sqlite3.connect(DATABASE_PATH) as connection:
        candidate_count = export_candidate_info_to_csv(
            connection
        )

        profile_count = export_professional_info_to_json(
            connection
        )

    # Display a summary of the export operation.
    print("\nExport completed successfully.")
    print(f"Candidate records exported: {candidate_count}")
    print(f"Professional profiles exported: {profile_count}")
    print(f"Output folder: {EXPORT_FOLDER}")


# Execute the exports when this script is run directly.
if __name__ == "__main__":
    main()