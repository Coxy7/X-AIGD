from __future__ import annotations

import streamlit as st

from xaigd_viewer.constants import CATEGORY_LABELS, CATEGORY_ORDER


def button_label_map() -> dict[str, str]:
    labels = {
        "ArrowLeft": "Previous (←/A)",
        "ArrowRight": "Next (→/D)",
        "a": "Previous (←/A)",
        "A": "Previous (←/A)",
        "d": "Next (→/D)",
        "D": "Next (→/D)",
        "0": "0 All",
    }
    for index, category in enumerate(CATEGORY_ORDER, start=1):
        labels[str(index)] = f"{index} {CATEGORY_LABELS[category]}"
    return labels


def inject_keyboard_shortcuts() -> None:
    labels = button_label_map()
    st.html(
        f"""
        <script>
        if (!window.__xaigdViewerShortcutsInstalled) {{
          window.__xaigdViewerShortcutsInstalled = true;
          const buttonLabels = {labels!r};

          const normalize = (value) => value.replace(/\\s+/g, " ").trim();
          const isEditableTarget = (active) => {{
            const tag = active && active.tagName ? active.tagName.toLowerCase() : "";
            const role = active && active.getAttribute ? active.getAttribute("role") : "";
            return (
              tag === "input" ||
              tag === "textarea" ||
              tag === "select" ||
              role === "combobox" ||
              (active && active.isContentEditable)
            );
          }};
          const clickButton = (label) => {{
            const buttons = Array.from(document.querySelectorAll("button"));
            const target = buttons.find((button) => !button.disabled && normalize(button.textContent || "") === label);
            if (target) {{
              target.click();
              return true;
            }}
            return false;
          }};
          const clickScopedButton = (selector) => {{
            const target = document.querySelector(selector);
            if (target && !target.disabled) {{
              target.click();
              return true;
            }}
            return false;
          }};
          const clickFullscreenButton = () => {{
            const imageBlocks = Array.from(document.querySelectorAll('[data-testid="stImage"]')).reverse();
            for (const block of imageBlocks) {{
              const button = block.querySelector(
                'button[aria-label*="fullscreen" i], button[title*="fullscreen" i]'
              );
              if (button && !button.disabled) {{
                button.click();
                return true;
              }}
            }}

            const fallbackButton = Array.from(
              document.querySelectorAll('button[aria-label*="fullscreen" i], button[title*="fullscreen" i]')
            ).find((button) => !button.disabled);
            if (fallbackButton) {{
              fallbackButton.click();
              return true;
            }}
            return false;
          }};
          const closeFullscreenDialog = () => {{
            const dialogs = Array.from(document.querySelectorAll('[role="dialog"], [data-testid="stDialog"]'));
            const dialog = dialogs.find((item) => item.getClientRects().length > 0);
            if (!dialog) {{
              return false;
            }}

            const closeButton = dialog.querySelector('button[aria-label*="close" i], button[title*="close" i]');
            if (closeButton && !closeButton.disabled) {{
              closeButton.click();
              return true;
            }}

            document.dispatchEvent(new KeyboardEvent("keydown", {{
              key: "Escape",
              code: "Escape",
              bubbles: true,
            }}));
            return true;
          }};

          window.addEventListener("keydown", (event) => {{
            const active = event.target;
            if (isEditableTarget(active) || event.ctrlKey || event.metaKey || event.altKey) {{
              return;
            }}

            if (event.key === "f" || event.key === "F") {{
              event.preventDefault();
              if (!closeFullscreenDialog()) {{
                clickFullscreenButton();
              }}
              return;
            }}

            if (event.key === "s" || event.key === "S") {{
              event.preventDefault();
              clickScopedButton(".st-key-toggle_all_labels button");
              return;
            }}

            const label = buttonLabels[event.key];
            if (!label) {{
              return;
            }}

            event.preventDefault();
            clickButton(label);
          }}, true);
        }}
        </script>
        """,
        width="content",
        unsafe_allow_javascript=True,
    )
