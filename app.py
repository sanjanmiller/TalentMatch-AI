"""
Purpose:
    Streamlit entry point for TalentMatch AI - Intelligent Recruitment
    & Candidate Matching.

What this file does:
    - Configures the page and renders the shared branding.
    - Initialises the session state safely (existing values are never
      overwritten during Streamlit reruns).
    - Renders the sidebar navigation and dispatches to the five real
      screen modules:

        1. New Recruitment - JD input + "Find relevant candidates"
           button that runs the existing LangGraph workflow ONCE per
           explicit click (screen_recruit.py).
        2. HR Review - per-candidate approve/reject decisions, final
           confirmation and the interrupt/resume that sends shortlist
           emails through the existing backend (screen_review.py).
        3. Dashboard - summary metrics from the additive
           recruitment_runs table and email_tracking (screen_history).
        4. Past JDs - read-only historical recruitment runs with
           candidate drill-downs (screen_history.py).
        5. Candidates - read-only candidate directory built from the
           existing identity, profile and tracking records
           (screen_candidates.py).

    - This file contains NO business logic; every backend interaction
      goes through the screen modules and their services, which keeps
      UI code separate from workflow/LLM/email logic.

Interacts with:
    - src/ui/components.py (branding, page config, sidebar navigation)
    - src/ui/session.py     (initialises before any widget is created)
    - src/ui/screen_recruit.py / screen_review.py / screen_history.py

Side effects:
    - On first app run the shared SQLite checkpoint database and the
      additive run-history table are created (by their own services)
      the first time they are actually used.
"""

from src.ui.components import (
    SCREEN_CANDIDATES,
    SCREEN_DASHBOARD,
    SCREEN_HR_REVIEW,
    SCREEN_NEW_RECRUITMENT,
    SCREEN_PAST_JDS,
    apply_page_config,
    render_main_header,
    render_sidebar,
)
from src.ui.screen_recruit import render_new_recruitment_screen
from src.ui.screen_review import render_hr_review_screen
from src.ui.screen_history import render_dashboard_screen
from src.ui.screen_history import render_past_jds_screen
from src.ui.screen_candidates import render_candidates_screen
from src.ui.session import initialize_session_state


def main() -> None:
    """Entry point of the TalentMatch AI Streamlit application."""

    # Page config must run before any other Streamlit element.
    apply_page_config()

    # Safe initialisation: existing session values are never
    # overwritten during Streamlit's automatic reruns, which is what
    # keeps a run's results, thread ID and decisions alive during
    # navigation and refreshes.
    initialize_session_state()

    render_main_header()

    # The sidebar radio reads/writes the keyed "current_screen" session
    # value; screen modules can also redirect it ("Go to ..." buttons)
    # and the radio value follows along automatically.
    selected_screen = render_sidebar()

    # Screen dispatch based on the sidebar selection.
    if selected_screen == SCREEN_NEW_RECRUITMENT:
        render_new_recruitment_screen()
    elif selected_screen == SCREEN_HR_REVIEW:
        render_hr_review_screen()
    elif selected_screen == SCREEN_DASHBOARD:
        render_dashboard_screen()
    elif selected_screen == SCREEN_PAST_JDS:
        render_past_jds_screen()
    else:
        render_candidates_screen()


if __name__ == "__main__":
    main()
