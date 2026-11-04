"""
Purpose:
    Shared branding, colour theme, layout and navigation helpers for
    the TalentMatch AI Streamlit application.

What this file does:
    - Centralises ALL branding text (name, subtitle, tagline) so every
      screen renders it consistently.
    - Applies the enterprise blue colour palette (navy sidebar, royal
      blue primary actions, ice-blue active navigation, pale-blue and
      white workspace) with a small, restrained set of CSS rules.
    - Renders the sidebar (branding + the single navigation radio used
      by ALL screens) and the main header.
    - Exposes the screen-name constants and shared status-label
      helpers so every screen uses identical wording.

Navigation design notes:
    - There is exactly ONE navigation mechanism: the keyed
      "current_screen" radio in the sidebar. Screen modules redirect
      it through their "Go to ..." button callbacks; nothing else
      ever renders navigation.
    - Icons are added with format_func, which only changes the
      DISPLAYED text. The underlying option values (and therefore the
      session keys, the screen dispatch and any stored session state)
      remain the plain screen names.

Interacts with:
    - src/ui/session.py (initialises "current_screen" BEFORE the radio
      is created, which is how the selection survives reruns)

Side effects:
    - render_sidebar() and render_main_header() draw UI elements; the
      navigation radio writes its selection to
      st.session_state["current_screen"].
"""

import streamlit as st

# ---------------------------------------------------------------------------
# Branding constants (single source of truth for the whole UI).
# ---------------------------------------------------------------------------

APP_NAME = "TalentMatch AI"
APP_SUBTITLE = "Intelligent Recruitment & Candidate Matching"
APP_TAGLINE = (
    "Discover the right talent. Evaluate with intelligence. "
    "Hire with confidence."
)
WELCOME_HEADING = "Welcome to TalentMatch AI"
WELCOME_SUBTEXT = "AI-powered candidate discovery and intelligent recruitment"

# ---------------------------------------------------------------------------
# Colour palette. Used both as CSS (below) and, where Streamlit does
# not support CSS targeting, as fallback styling values in screens.
# ---------------------------------------------------------------------------

COLOR_NAVY = "#163A63"   # sidebar, branding, headings
COLOR_ROYAL = "#2563EB"  # primary buttons and actions
COLOR_ROYAL_DARK = "#1D4ED8"  # primary-button hover shade
COLOR_ICE = "#DBEAFE"    # active navigation and highlights
COLOR_PALE = "#EFF6FF"   # secondary panels and backgrounds
COLOR_WHITE = "#FFFFFF"  # main workspace and cards
COLOR_SLATE = "#475569"  # secondary text
COLOR_BORDER = "#D6DFEA"  # subtle blue-grey card borders

# The five main screens, shown in the sidebar in work order.
SCREEN_NEW_RECRUITMENT = "New Recruitment"
SCREEN_HR_REVIEW = "HR Review"
SCREEN_DASHBOARD = "Dashboard"
SCREEN_PAST_JDS = "Past JDs"
SCREEN_CANDIDATES = "Candidates"

NAVIGATION_SCREENS = [
    SCREEN_NEW_RECRUITMENT,
    SCREEN_HR_REVIEW,
    SCREEN_DASHBOARD,
    SCREEN_PAST_JDS,
    SCREEN_CANDIDATES,
]

# Icons shown NEXT TO the navigation labels. These never change the
# stored selection (format_func is display-only).
NAVIGATION_ICONS = {
    SCREEN_NEW_RECRUITMENT: "🔎",
    SCREEN_HR_REVIEW: "✅",
    SCREEN_DASHBOARD: "📊",
    SCREEN_PAST_JDS: "📄",
    SCREEN_CANDIDATES: "👥",
}


def _apply_theme_css() -> None:
    """Inject the restrained CSS that implements the blue palette.

    Rules are limited to: colour/background adjustments, the active
    navigation highlight and small spacing corrections. No gradients,
    animations or decorative effects are involved.
    """

    st.markdown(
        f"""
        <style>
            /* --- Global workspace ----------------------------------- */
            #MainMenu {{visibility: hidden;}}
            footer {{visibility: hidden;}}
            header {{visibility: hidden;}}
            .block-container {{padding-top: 1.25rem;}}

            /* The sidebar COLLAPSE arrow ("»") lives inside the
               sidebar, so it stays reachable when expanded. But the
               sidebar REOPEN arrow is rendered inside Streamlit's
               (hidden) header — if it inherited hidden there, a
               collapsed sidebar could never be reopened. This rule
               keeps the expand button visible the moment it exists
               in the DOM. */
            header [data-testid="stExpandSidebarButton"] {{
                visibility: visible;
            }}

            /* Headings carry the brand navy (important because
               Streamlit injects its own stylesheet AFTER ours, and
               same-specificity later rules would otherwise win). */
            h1, h2, h3, h4, h5, h6,
            .stMarkdown h1, .stMarkdown h2, .stMarkdown h3,
            .stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {{
                color: {COLOR_NAVY} !important;
            }}

            /* Secondary text in slate grey. */
            [data-testid="stCaptionContainer"] {{
                color: {COLOR_SLATE} !important;
            }}

            /* --- Sidebar: navy with white text ---------------------- */
            [data-testid="stSidebar"] {{
                background-color: {COLOR_NAVY};
            }}
            [data-testid="stSidebar"], [data-testid="stSidebar"] * {{
                color: {COLOR_WHITE};
            }}

            /* --- Sidebar navigation --------------------------------- */
            /* The currently selected option gets an ice-blue pill with
               navy text; unselected options stay navy/white. The
               :has() selector looks at the radio input state, which
               is the only reliable way to style "the selected item"
               in a native Streamlit radio. */
            [data-testid="stSidebar"] label {{
                border-radius: 8px;
                padding: 2px 8px;
            }}
            [data-testid="stSidebar"] label:hover {{
                background-color: rgba(219, 234, 254, 0.15);
                cursor: pointer;
            }}
            [data-testid="stSidebar"] label:has(input:checked),
            [data-testid="stSidebar"] label:has(input:checked) div,
            [data-testid="stSidebar"] label:has(input:checked) span {{
                background-color: {COLOR_ICE};
                color: {COLOR_NAVY} !important;
            }}

            /* --- Buttons: royal blue primary actions --------------- */
            [data-testid="stBaseButton-primary"]:not(:disabled) {{
                background-color: {COLOR_ROYAL} !important;
                background-image: none !important;
                color: {COLOR_WHITE} !important;
                border-color: {COLOR_ROYAL} !important;
            }}
            [data-testid="stBaseButton-primary"]:not(:disabled):hover {{
                background-color: {COLOR_ROYAL_DARK} !important;
                border-color: {COLOR_ROYAL_DARK} !important;
                color: {COLOR_WHITE} !important;
            }}
            /* A DISABLED primary action stays recognisable within the
               palette: muted royal blue with reduced contrast, so it
               is clearly not clickable yet still reads as the brand. */
            [data-testid="stBaseButton-primary"]:disabled {{
                background-color: rgba(37, 99, 235, 0.40) !important;
                color: rgba(255, 255, 255, 0.85) !important;
                border-color: transparent !important;
            }}

            /* --- Cards: white with subtle blue-grey borders -------- */
            [data-testid="stVerticalBlockBorderWrapper"] {{
                background-color: {COLOR_WHITE};
                border-color: {COLOR_BORDER};
                border-width: 1px;
                border-radius: 8px;
            }}

            /* Expander headers read as card headers. */
            [data-testid="stExpander"] summary {{
                color: {COLOR_NAVY};
            }}

            /* --- Supporting panels (info boxes) stay pale blue ----- */
            [data-testid="stInfo"] {{
                background-color: {COLOR_PALE};
                border-color: {COLOR_BORDER};
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def apply_page_config() -> None:
    """Configure the browser page and apply the shared theme.

    Must run before any other Streamlit element is rendered. The page
    title is what the browser tab shows. Layout is "wide" so candidate
    assessments stay readable, and the sidebar is expanded by default.
    """

    st.set_page_config(
        page_title=APP_NAME,
        page_icon="💼",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    _apply_theme_css()


def render_main_header() -> None:
    """Render the branding block at the top of the main area."""

    st.title(APP_NAME)
    st.caption(APP_SUBTITLE)
    st.header(WELCOME_HEADING)
    st.write(WELCOME_SUBTEXT)
    st.caption(f"*{APP_TAGLINE}*")


def render_sidebar() -> str:
    """Render branding and navigation in the sidebar.

    The "current_screen" key is shared with the session-state default
    in src/ui/session.py: when no selection exists yet the default
    ("New Recruitment") is used, and because the radio is keyed, the
    HR user's selection survives Streamlit reruns. Switching screens
    never resets the current recruitment session, because every
    screen reads its working data from the same persistent
    session-state keys.

    Returns:
        The currently selected screen name.
    """

    with st.sidebar:
        # Branding block at the top of the sidebar.
        st.markdown(f"### {APP_NAME}")
        st.caption(APP_SUBTITLE)

        # One radio keeps exactly one screen selected; the key keeps
        # the selection stable across reruns. format_func only adds
        # display icons next to the labels.
        selected_screen = st.radio(
            "Navigation",
            NAVIGATION_SCREENS,
            key="current_screen",
            format_func=lambda screen: (
                f"{NAVIGATION_ICONS.get(screen, '•')}  {screen}"
            ),
            label_visibility="collapsed",
        )

        # Current-run context (only after a run has been started).
        thread_id = st.session_state.get("thread_id")
        if thread_id:
            job_title = st.session_state.get("job_title")
            st.caption(f"Current run: {job_title or '—'}")

        # Tagline pinned at the bottom of the sidebar.
        st.caption(APP_TAGLINE)

    return selected_screen


# ---------------------------------------------------------------------------
# Status wording helpers. Centralised so the review screen, the results
# area and the dashboard always use identical, unambiguous labels.
# ---------------------------------------------------------------------------

# Maps the graph's raw email_results status ("sent" / "failed" /
# "skipped") to the HR-facing label, matching the tracked wording of
# the existing email_tracking service.
EMAIL_STATUS_LABELS: dict[str, str] = {
    "sent": "Shortlisted – Mail Sent",
    "failed": "Shortlisted – Mail Not Sent",
    "skipped": (
        "Shortlisted – Mail Not Sent (skipped - duplicate prevention)"
    ),
}

# Shown for approved candidates that have no email_results entry (for
# example when the resume could not bring the email address forward).
EMAIL_STATUS_MISSING = (
    "Shortlisted – Mail Not Sent (no email attempt recorded)"
)

# Rejected candidates never receive an email; this wording is shared
# with the existing email_tracking service's NOT_SHORTLISTED label.
NOT_SHORTLISTED_LABEL = "Not Shortlisted"


def email_status_label(raw_status: str | None) -> str:
    """Turn the graph's raw email result status into an HR label.

    Args:
        raw_status: One of "sent", "failed", "skipped" or anything
            unexpected/missing.

    Returns:
        A stable, HR-readable status label.
    """

    if not raw_status:
        return EMAIL_STATUS_MISSING

    return EMAIL_STATUS_LABELS.get(
        raw_status,
        f"Unknown email status ({raw_status})",
    )
