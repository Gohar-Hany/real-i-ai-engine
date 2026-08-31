"""
Google Sheets Provider — External store for the Admin Agent task queue.

Follows the same provider pattern as stores/llm/ and stores/vectordb/.
Handles Google Sheets API authentication, worksheet management,
and task record persistence.
"""

import os
import re
import datetime
import logging

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

logger = logging.getLogger("uvicorn.error")

SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]


class GoogleSheetsProvider:
    """
    Provider for reading/writing tasks to a Google Sheets-based shared memory.
    """

    def __init__(self, spreadsheet_id: str, client_id: str, client_secret: str):
        self.spreadsheet_id = spreadsheet_id
        self.client_id = client_id
        self.client_secret = client_secret
        self.service = None

        # Token is stored in the project's assets directory
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.token_path = os.path.join(base_dir, "assets", "google_token.json")

    def connect(self, run_oauth=False):
        """Authenticate and build the Google Sheets API service."""
        if not self.spreadsheet_id or not self.client_id or not self.client_secret:
            logger.warning(
                "[Google Sheets] Credentials not configured. "
                "Admin Agent Google Sheets integration will be unavailable."
            )
            return

        creds = None
        if os.path.exists(self.token_path):
            try:
                creds = Credentials.from_authorized_user_file(self.token_path, SCOPES)
            except Exception as e:
                logger.warning(f"[Google Sheets] Failed to load cached token: {e}")

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                try:
                    logger.info("[Google Sheets] Refreshing expired credentials...")
                    creds.refresh(Request())
                except Exception as e:
                    logger.warning(f"[Google Sheets] Failed to refresh token: {e}")
                    creds = None

            # If still invalid and we aren't allowed to run interactive OAuth, return
            if (not creds or not creds.valid) and not run_oauth:
                logger.warning("[Google Sheets] No valid cached credentials found. Run OAuth on write request.")
                return

            if not creds or not creds.valid:
                logger.info("[Google Sheets] Running OAuth flow...")
                client_config = {
                    "installed": {
                        "client_id": self.client_id,
                        "client_secret": self.client_secret,
                        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                        "token_uri": "https://oauth2.googleapis.com/token",
                        "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
                        "redirect_uris": ["http://localhost"],
                    }
                }
                flow = InstalledAppFlow.from_client_config(client_config, SCOPES)
                try:
                    creds = flow.run_local_server(port=0)
                except Exception as e:
                    logger.warning(
                        f"[Google Sheets] Local server browser launch failed: {e}. "
                        "Attempting to run without opening browser automatically..."
                    )
                    creds = flow.run_local_server(port=0, open_browser=False)

                # Cache credentials
                os.makedirs(os.path.dirname(self.token_path), exist_ok=True)
                with open(self.token_path, "w") as token:
                    token.write(creds.to_json())

        self.service = build("sheets", "v4", credentials=creds)
        logger.info("[Google Sheets] Connected successfully ✓")

    def disconnect(self):
        """Cleanup (no persistent connection to close)."""
        self.service = None

    def _ensure_worksheet_exists(self, title: str = "Shared Memory"):
        """Create the worksheet if it doesn't exist."""
        spreadsheet = self.service.spreadsheets().get(
            spreadsheetId=self.spreadsheet_id
        ).execute()
        sheet_titles = [
            s.get("properties", {}).get("title")
            for s in spreadsheet.get("sheets", [])
        ]

        if title not in sheet_titles:
            logger.info(f"[Google Sheets] Creating worksheet '{title}'...")
            self.service.spreadsheets().batchUpdate(
                spreadsheetId=self.spreadsheet_id,
                body={
                    "requests": [
                        {"addSheet": {"properties": {"title": title}}}
                    ]
                },
            ).execute()
            return True
        return False

    def _format_sheet_text_black(self, title: str = "Shared Memory"):
        """Set text color of all cells to black for readability."""
        try:
            spreadsheet = self.service.spreadsheets().get(
                spreadsheetId=self.spreadsheet_id
            ).execute()
            sheet_id = None
            for s in spreadsheet.get("sheets", []):
                if s.get("properties", {}).get("title") == title:
                    sheet_id = s["properties"]["sheetId"]
                    break

            if sheet_id is not None:
                self.service.spreadsheets().batchUpdate(
                    spreadsheetId=self.spreadsheet_id,
                    body={
                        "requests": [
                            {
                                "repeatCell": {
                                    "range": {
                                        "sheetId": sheet_id,
                                        "startRowIndex": 0,
                                        "startColumnIndex": 0,
                                    },
                                    "cell": {
                                        "userEnteredFormat": {
                                            "textFormat": {
                                                "foregroundColor": {
                                                    "red": 0.0,
                                                    "green": 0.0,
                                                    "blue": 0.0,
                                                }
                                              }
                                        }
                                    },
                                    "fields": "userEnteredFormat.textFormat.foregroundColor",
                                }
                            }
                        ]
                    },
                ).execute()
        except Exception as e:
            logger.warning(f"[Google Sheets] Failed to format text color: {e}")

    def write_task(
        self,
        task_type: str,
        description: str,
        course: str = "General",
        priority: str = "High",
        assigned_agent: str = "TA",
        status: str = "Pending",
        notes: str = "",
    ) -> str:
        """
        Append a new task row to the Google Sheet task queue.

        Auto-generates an incremented task ID (e.g. T001, T002) and timestamp.
        Returns a status message with the generated task ID.
        """
        if not self.service:
            logger.info("[Google Sheets] Lazily connecting to Google Sheets API before write...")
            self.connect(run_oauth=True)

        if not self.service:
            raise RuntimeError(
                "Google Sheets provider not connected. "
                "Check GOOGLE_SPREADSHEET_ID, GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET in .env"
            )


        self._ensure_worksheet_exists("Shared Memory")

        # Read existing rows to compute next Task ID
        result = self.service.spreadsheets().values().get(
            spreadsheetId=self.spreadsheet_id,
            range="Shared Memory!A:I",
        ).execute()

        rows = result.get("values", [])

        headers = [
            "Task_ID", "Task_Type", "Description", "Course",
            "Priority", "Assigned_Agent", "Status", "Created_At", "Notes",
        ]
        new_task_id = "T001"

        if not rows:
            # Sheet is empty — write headers first
            self.service.spreadsheets().values().update(
                spreadsheetId=self.spreadsheet_id,
                range="Shared Memory!A1",
                valueInputOption="RAW",
                body={"values": [headers]},
            ).execute()
            rows = [headers]

        # Calculate next Task ID
        max_num = 0
        if len(rows) > 1:
            for row in rows[1:]:
                if row and len(row) > 0:
                    match = re.search(r"T(\d+)", str(row[0]))
                    if match:
                        num = int(match.group(1))
                        if num > max_num:
                            max_num = num
            new_task_id = f"T{max_num + 1:03d}"

        created_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        new_row = [
            new_task_id, task_type, description, course,
            priority, assigned_agent, status, created_at, notes,
        ]

        self.service.spreadsheets().values().append(
            spreadsheetId=self.spreadsheet_id,
            range="Shared Memory!A:I",
            valueInputOption="RAW",
            insertDataOption="INSERT_ROWS",
            body={"values": [new_row]},
        ).execute()

        self._format_sheet_text_black("Shared Memory")

        logger.info(f"[Google Sheets] Task {new_task_id} created successfully")
        return f"SUCCESS: Task {new_task_id} successfully created and written to Google Sheets."

    def get_last_task_id(self) -> str:
        """Read the last Task_ID from the sheet (fallback method)."""
        if not self.service:
            return "UNKNOWN"

        try:
            result = self.service.spreadsheets().values().get(
                spreadsheetId=self.spreadsheet_id,
                range="Shared Memory!A:A",
            ).execute()
            rows = result.get("values", [])
            if len(rows) > 1:
                return str(rows[-1][0])
        except Exception as e:
            logger.error(f"[Google Sheets] Failed to get last task ID: {e}")

        return "UNKNOWN"
