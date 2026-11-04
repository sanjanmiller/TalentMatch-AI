
"""
Purpose:
    Send candidate shortlisting emails through Gmail SMTP.

The service reads SMTP credentials from environment variables or the
project's .env file. It sends only the email requested by its caller;
HR approval and candidate selection are handled by the workflow.

Uses Python's built-in smtplib and EmailMessage modules.
"""

import os
import smtplib
import ssl
from email.message import EmailMessage

from dotenv import load_dotenv


class EmailConfigurationError(Exception):
    """Raised when required SMTP configuration is missing or invalid."""


class EmailDeliveryError(Exception):
    """Raised when an email cannot be sent successfully."""


def _get_smtp_settings() -> dict[str, str | int]:
    """Load and validate the SMTP settings needed to send an email."""

    load_dotenv()

    host = os.getenv("SMTP_HOST", "smtp.gmail.com").strip()
    port_value = os.getenv("SMTP_PORT", "587").strip()
    username = os.getenv("SMTP_USERNAME", "").strip()
    password = os.getenv("SMTP_APP_PASSWORD", "").strip()
    sender_name = os.getenv("SMTP_SENDER_NAME", "HR Team").strip()

    if not host or not username or not password:
        raise EmailConfigurationError(
            "SMTP_HOST, SMTP_USERNAME, and SMTP_APP_PASSWORD "
            "must be configured."
        )

    try:
        port = int(port_value)
    except ValueError as exc:
        raise EmailConfigurationError(
            "SMTP_PORT must be a valid integer."
        ) from exc

    if not 1 <= port <= 65535:
        raise EmailConfigurationError(
            "SMTP_PORT must be between 1 and 65535."
        )

    return {
        "host": host,
        "port": port,
        "username": username,
        "password": password,
        "sender_name": sender_name or "HR Team",
    }


def create_shortlisting_email(
    candidate_name: str,
    candidate_email: str,
    job_title: str,
) -> EmailMessage:
    """Create the agreed shortlisting email without sending it."""

    candidate_name = candidate_name.strip()
    candidate_email = candidate_email.strip()
    job_title = job_title.strip()

    if not candidate_name:
        raise ValueError("Candidate name cannot be empty.")

    if not candidate_email or "@" not in candidate_email:
        raise ValueError("A valid candidate email address is required.")

    if not job_title:
        raise ValueError("Job title cannot be empty.")

    message = EmailMessage()
    message["To"] = candidate_email
    message["Subject"] = "Application Update – You've Been Shortlisted"

    message.set_content(
        f"""Dear {candidate_name},

We are pleased to inform you that your application has been shortlisted
for the {job_title} position.

Thank you for your interest in this opportunity and for sharing your profile
with us. The hiring manager will contact you regarding the next steps in
the recruitment process.

We appreciate your patience and interest in joining our team.

Best regards,
HR Team
"""
    )

    return message


def send_shortlisting_email(
    candidate_name: str,
    candidate_email: str,
    job_title: str,
) -> None:
    """
    Send one shortlisting email through Gmail SMTP.

    Raises:
        ValueError: If candidate details are invalid.
        EmailConfigurationError: If SMTP settings are invalid.
        EmailDeliveryError: If SMTP connection or delivery fails.
    """

    # Validate candidate details before attempting to connect to SMTP.
    message = create_shortlisting_email(
        candidate_name=candidate_name,
        candidate_email=candidate_email,
        job_title=job_title,
    )

    settings = _get_smtp_settings()

    message["From"] = (
        f'{settings["sender_name"]} <{settings["username"]}>'
    )

    try:
        # Port 587 uses STARTTLS to encrypt the connection.
        with smtplib.SMTP(
            settings["host"],
            settings["port"],
            timeout=30,
        ) as server:
            server.ehlo()
            server.starttls(context=ssl.create_default_context())
            server.ehlo()
            server.login(settings["username"], settings["password"])

            refused_recipients = server.send_message(message)

            if refused_recipients:
                raise EmailDeliveryError(
                    "The SMTP server refused the candidate's email address."
                )

    except EmailDeliveryError:
        raise
    except (smtplib.SMTPException, OSError, TimeoutError) as exc:
        # Keep credentials and raw server details out of user-facing errors.
        raise EmailDeliveryError(
            "The shortlisting email could not be sent. "
            "Check the SMTP configuration and application logs."
        ) from exc
