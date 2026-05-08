from __future__ import annotations

import unittest
from unittest.mock import patch

from xaigd_viewer.keyboard import inject_keyboard_shortcuts


class KeyboardShortcutTests(unittest.TestCase):
    def test_inject_keyboard_shortcuts_includes_fullscreen_toggle(self) -> None:
        with patch("xaigd_viewer.keyboard.st.html") as html_mock:
            inject_keyboard_shortcuts()

        self.assertTrue(html_mock.called)
        script = html_mock.call_args.args[0]
        self.assertIn('event.key === "f" || event.key === "F"', script)
        self.assertIn('event.key === "s" || event.key === "S"', script)
        self.assertIn("closeFullscreenDialog", script)
        self.assertIn("clickFullscreenButton", script)
        self.assertIn('.st-key-toggle_all_labels button', script)


if __name__ == "__main__":
    unittest.main()
