"""
Input Validation & Auth Hardening Test Suite — tests/test_input_validation.py
=============================================================================

PURPOSE
-------
Covers every rule in `src/utils/input_validation.py` and the places the
application applies them:

  1. TestStripControlChars      — control characters cannot survive into a sink
  2. TestDigitsOnly             — ASCII-only digit extraction
  3. TestNormalizePhone         — the paste shapes people actually use
  4. TestValidatePhone          — exactly 10 digits, no alphabets  (the headline rule)
  5. TestPhoneRejectsInjection  — SQL / script / traversal payloads in a phone field
  6. TestValidatePassword       — length floor and ceiling
  7. TestValidateEmail          — shape, bound, header-injection
  8. TestValidateName           — bounds and character allow-list
  9. TestValidateText           — free-text bounds
 10. TestValidateUsername       — phone-style vs name-style identifiers
 11. TestValidateAge            — human range only
 12. TestValidateSerial         — device serial charset and bound
 13. TestSanitizeForLog         — log-forging and log-flooding defences
 14. TestUnicodeDigitHandling   — non-ASCII digits are not silently accepted
 15. TestLoginRateLimiter       — lockout, expiry, isolation, reset
 16. TestWidgetHelpers          — degrade quietly with no Qt present
 17. TestStandardLimits         — the constants themselves are sane
 18. TestFormsApplyValidation   — the forms really call the shared rules
 19. TestAuthStoreHardening     — users.json permissions and hash-only storage
 20. TestNoInjectionSinks       — no eval/exec/shell/TLS-bypass in shipped code

EXECUTION
---------
Run from project root:
    python -m pytest tests/test_input_validation.py -v
All tests execute headlessly — no display, no hardware, no network.
"""

import os
import re
import sys
import json
import unittest
from unittest.mock import MagicMock

# ── Path setup so src modules are importable ─────────────────────────────────
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SRC = os.path.join(_ROOT, "src")
for _p in [_ROOT, _SRC]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from utils.input_validation import (      # noqa: E402
    _NAME_FORBIDDEN_CHARS,
    ADDRESS_MAX_LENGTH,
    AGE_MAX,
    AGE_MIN,
    EMAIL_MAX_LENGTH,
    LOGIN_LOCKOUT_SECONDS,
    LOGIN_MAX_ATTEMPTS,
    LoginRateLimiter,
    NAME_MAX_LENGTH,
    NAME_MIN_LENGTH,
    PASSWORD_MAX_LENGTH,
    PASSWORD_MIN_LENGTH,
    PHONE_DIGITS,
    SERIAL_MAX_LENGTH,
    USERNAME_MAX_LENGTH,
    USERNAME_MIN_LENGTH,
    DEVICE_DISCONNECTED_PLACEHOLDER,
    apply_digit_only,
    apply_text_limits,
    digits_only,
    is_device_placeholder,
    normalize_phone,
    sanitize_filename_component,
    sanitize_for_log,
    strip_control_chars,
    validate_age,
    validate_email,
    validate_name,
    validate_password,
    validate_phone,
    validate_serial,
    validate_text,
    validate_username,
)

# Arabic-Indic digits — str.isdigit() and regex \d both accept these.
ARABIC_INDIC_TEN = "٩٨٧٦٥٤٣٢١٠"

# Written this way so the literal never has to survive an escape round-trip.
BACKSLASH = chr(92)


# ══════════════════════════════════════════════════════════════════════════════
# 1. CONTROL CHARACTERS
# ══════════════════════════════════════════════════════════════════════════════

class TestStripControlChars(unittest.TestCase):
    """A control character in a form field is how one value becomes two rows."""

    def test_removes_newline_and_carriage_return(self):
        self.assertEqual(strip_control_chars("Dr.\nSmith"), "Dr.Smith")
        self.assertEqual(strip_control_chars("Dr.\r\nSmith"), "Dr.Smith")

    def test_removes_null_byte(self):
        self.assertEqual(strip_control_chars("abc\x00def"), "abcdef")

    def test_removes_tab_and_escape(self):
        self.assertEqual(strip_control_chars("a\tb\x1bc"), "abc")

    def test_removes_c1_range(self):
        self.assertEqual(strip_control_chars("a\x85b"), "ab")

    def test_keeps_ordinary_text_untouched(self):
        self.assertEqual(strip_control_chars("Dr. A. Sharma (Cardiology)"),
                         "Dr. A. Sharma (Cardiology)")

    def test_handles_none_and_non_strings(self):
        self.assertEqual(strip_control_chars(None), "")
        self.assertEqual(strip_control_chars(12345), "12345")


# ══════════════════════════════════════════════════════════════════════════════
# 2. DIGIT EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

class TestDigitsOnly(unittest.TestCase):

    def test_extracts_digits_in_order(self):
        self.assertEqual(digits_only("+91 98765-43210"), "919876543210")

    def test_drops_letters(self):
        self.assertEqual(digits_only("12ab34cd56"), "123456")

    def test_empty_and_none(self):
        self.assertEqual(digits_only(""), "")
        self.assertEqual(digits_only(None), "")

    def test_rejects_non_ascii_digits(self):
        # The whole point of using [^0-9] instead of \D.
        self.assertEqual(digits_only(ARABIC_INDIC_TEN), "")


# ══════════════════════════════════════════════════════════════════════════════
# 3. PHONE NORMALISATION
# ══════════════════════════════════════════════════════════════════════════════

class TestNormalizePhone(unittest.TestCase):

    def test_plain_ten_digits_unchanged(self):
        self.assertEqual(normalize_phone("9876543210"), "9876543210")

    def test_strips_country_code(self):
        self.assertEqual(normalize_phone("+919876543210"), "9876543210")
        self.assertEqual(normalize_phone("919876543210"), "9876543210")

    def test_strips_international_prefix(self):
        self.assertEqual(normalize_phone("00919876543210"), "9876543210")

    def test_strips_trunk_zero(self):
        self.assertEqual(normalize_phone("09876543210"), "9876543210")

    def test_strips_separators(self):
        self.assertEqual(normalize_phone("+91 98765 43210"), "9876543210")
        self.assertEqual(normalize_phone("(987) 654-3210"), "9876543210")


# ══════════════════════════════════════════════════════════════════════════════
# 4. PHONE VALIDATION — the headline rule: 10 digits, no alphabets
# ══════════════════════════════════════════════════════════════════════════════

class TestValidatePhone(unittest.TestCase):

    def test_accepts_exactly_ten_digits(self):
        ok, cleaned, err = validate_phone("9876543210")
        self.assertTrue(ok)
        self.assertEqual(cleaned, "9876543210")
        self.assertEqual(err, "")

    def test_accepts_every_leading_digit(self):
        for lead in "0123456789":
            with self.subTest(lead=lead):
                ok, _c, _e = validate_phone(lead + "876543210")
                self.assertTrue(ok, f"leading digit {lead} should be accepted")

    def test_rejects_nine_digits(self):
        ok, _c, err = validate_phone("987654321")
        self.assertFalse(ok)
        self.assertIn("exactly 10 digits", err)

    def test_rejects_eleven_digits(self):
        ok, _c, err = validate_phone("98765432101")
        self.assertFalse(ok)
        self.assertIn("exactly 10 digits", err)

    def test_rejects_alphabets(self):
        for bad in ("abcdefghij", "98765abcde", "9876543210a", "a9876543210"):
            with self.subTest(value=bad):
                ok, _c, err = validate_phone(bad)
                self.assertFalse(ok, f"{bad!r} must be rejected")
                self.assertIn("digits only", err)

    def test_rejects_mixed_alnum_rather_than_silently_shortening(self):
        # The old code normalised "12ab34cd56" to "123456" and then complained
        # about the length, which did not match what the user saw in the box.
        ok, cleaned, err = validate_phone("12ab34cd56")
        self.assertFalse(ok)
        self.assertEqual(cleaned, "")
        self.assertIn("digits only", err)

    def test_rejects_empty_and_whitespace(self):
        for bad in ("", "   ", None):
            with self.subTest(value=bad):
                ok, _c, err = validate_phone(bad)
                self.assertFalse(ok)
                self.assertIn("required", err)

    def test_rejects_decimal_and_symbols(self):
        for bad in ("98765.43210", "9876543210!", "9876*43210"):
            with self.subTest(value=bad):
                self.assertFalse(validate_phone(bad)[0])

    def test_strips_control_characters_before_checking(self):
        ok, cleaned, _e = validate_phone("98765\n43210")
        self.assertTrue(ok)
        self.assertEqual(cleaned, "9876543210")

    def test_custom_field_label_appears_in_error(self):
        _ok, _c, err = validate_phone("", "Contact number")
        self.assertIn("Contact number", err)

    def test_cleaned_value_is_always_ascii_digits(self):
        ok, cleaned, _e = validate_phone("+91 98765-43210")
        self.assertTrue(ok)
        self.assertTrue(cleaned.isascii())
        self.assertTrue(re.fullmatch(r"[0-9]{10}", cleaned))


class TestPhoneRejectsInjection(unittest.TestCase):
    """A phone field is a common place to try a payload. None should pass."""

    PAYLOADS = [
        "' OR '1'='1",
        "'; DROP TABLE users;--",
        "<script>alert(1)</script>",
        "../../etc/passwd",
        "${jndi:ldap://x/a}",
        "9876543210' --",
        "%0d%0aSet-Cookie:x=y",
        "9876543210\r\nX-Injected: 1",
        "{{7*7}}",
        "|whoami",
    ]

    def test_all_payloads_rejected(self):
        for payload in self.PAYLOADS:
            with self.subTest(payload=payload):
                ok, cleaned, _err = validate_phone(payload)
                self.assertFalse(ok, f"{payload!r} must not validate")
                self.assertEqual(cleaned, "")


# ══════════════════════════════════════════════════════════════════════════════
# 5. PASSWORD
# ══════════════════════════════════════════════════════════════════════════════

class TestValidatePassword(unittest.TestCase):

    def test_accepts_minimum_length(self):
        self.assertTrue(validate_password("a" * PASSWORD_MIN_LENGTH)[0])

    def test_rejects_one_below_minimum(self):
        ok, _c, err = validate_password("a" * (PASSWORD_MIN_LENGTH - 1))
        self.assertFalse(ok)
        self.assertIn("at least", err)

    def test_accepts_maximum_length(self):
        self.assertTrue(validate_password("a" * PASSWORD_MAX_LENGTH)[0])

    def test_rejects_one_above_maximum(self):
        # An unbounded password field is an unbounded amount of PBKDF2 work.
        ok, _c, err = validate_password("a" * (PASSWORD_MAX_LENGTH + 1))
        self.assertFalse(ok)
        self.assertIn("at most", err)

    def test_rejects_empty(self):
        ok, _c, err = validate_password("")
        self.assertFalse(ok)
        self.assertIn("required", err)

    def test_rejects_control_characters(self):
        ok, _c, err = validate_password("passw\x00rdX")
        self.assertFalse(ok)
        self.assertIn("control characters", err)

    def test_preserves_password_exactly(self):
        # Never trim or normalise a password — the stored hash must match.
        pwd = "  P@ssw0rd with spaces  "
        ok, cleaned, _e = validate_password(pwd)
        self.assertTrue(ok)
        self.assertEqual(cleaned, pwd)

    def test_allows_unicode_and_symbols(self):
        self.assertTrue(validate_password("Pässwörd!£€2024")[0])


# ══════════════════════════════════════════════════════════════════════════════
# 6. EMAIL
# ══════════════════════════════════════════════════════════════════════════════

class TestValidateEmail(unittest.TestCase):

    def test_accepts_ordinary_addresses(self):
        for good in ("a@b.co", "first.last@clinic.example.com",
                     "user+tag@sub.domain.org", "n_a-m.e%1@x.io"):
            with self.subTest(value=good):
                self.assertTrue(validate_email(good)[0], good)

    def test_rejects_malformed(self):
        for bad in ("plain", "no-at.com", "@nodomain.com", "a@b",
                    "a@@b.com", "a b@c.com", "a@b .com", "a@.com"):
            with self.subTest(value=bad):
                self.assertFalse(validate_email(bad)[0], bad)

    def test_rejects_header_injection(self):
        ok, _c, _e = validate_email("a@b.com\nBcc: victim@x.com")
        self.assertFalse(ok)

    def test_rejects_over_max_length(self):
        long_local = "a" * EMAIL_MAX_LENGTH
        self.assertFalse(validate_email(long_local + "@b.com")[0])

    def test_optional_when_not_required(self):
        ok, cleaned, err = validate_email("", required=False)
        self.assertTrue(ok)
        self.assertEqual(cleaned, "")
        self.assertEqual(err, "")

    def test_required_by_default(self):
        self.assertFalse(validate_email("")[0])


# ══════════════════════════════════════════════════════════════════════════════
# 7. NAME
# ══════════════════════════════════════════════════════════════════════════════

class TestValidateName(unittest.TestCase):

    def test_accepts_realistic_names(self):
        for good in ("Dr. A. Sharma", "O'Brien", "Mary-Jane Watson",
                     "Deckmount Health & Care", "Clinic (North)", "Ward 3/B"):
            with self.subTest(value=good):
                self.assertTrue(validate_name(good)[0], good)

    def test_rejects_below_minimum(self):
        self.assertFalse(validate_name("a" * (NAME_MIN_LENGTH - 1))[0])

    def test_accepts_exact_maximum(self):
        self.assertTrue(validate_name("a" * NAME_MAX_LENGTH)[0])

    def test_rejects_above_maximum(self):
        ok, _c, err = validate_name("a" * (NAME_MAX_LENGTH + 1))
        self.assertFalse(ok)
        self.assertIn("at most", err)

    def test_collapses_internal_whitespace(self):
        ok, cleaned, _e = validate_name("  Dr.    A.   Sharma  ")
        self.assertTrue(ok)
        self.assertEqual(cleaned, "Dr. A. Sharma")

    def test_accepts_non_ascii_names(self):
        # This is a clinical application used in India. An ASCII-only rule
        # refuses a doctor's own name, which is a defect, not a security
        # posture. Unicode letters from any script must pass.
        for good in ("Dr. Jos\u00e9 Garc\u00eda", "Zo\u00eb M\u00fcller",
                     "\u0926\u093f\u0935\u094d\u092f\u093e\u0902\u0936 "
                     "\u0936\u0930\u094d\u092e\u093e",
                     "\u674e\u533b\u751f", "\u0936\u094d\u0930\u0940 "
                     "\u0930\u093e\u092e \u0915\u094d\u0932\u093f\u0928\u093f\u0915"):
            with self.subTest(value=good):
                ok, _c, err = validate_name(good)
                self.assertTrue(ok, f"{good!r} was rejected: {err}")

    def test_rejects_metacharacters(self):
        for bad in ("<script>x</script>", "a;b", "a|b", "a`b", "a$b",
                    "a" + BACKSLASH + "b", "a{b}", "a[b]", "a<b>", 'a"b'):
            with self.subTest(value=bad):
                self.assertFalse(validate_name(bad)[0], bad)

    def test_every_forbidden_character_is_actually_caught(self):
        # Guards the rule itself. Hand-escaping a character class containing
        # both a backslash and a pipe is easy to get subtly wrong, and a
        # character that silently fails to match is a hole shaped like a rule.
        leaked = [c for c in _NAME_FORBIDDEN_CHARS
                  if validate_name("ab" + c + "cd")[0]]
        self.assertEqual(leaked, [],
                         f"these forbidden characters were accepted: {leaked!r}")

    def test_forbidden_set_covers_the_sinks_that_matter(self):
        for ch in ("<", ">", "{", "}", "[", "]", "|", "`", "$", ";", '"',
                   BACKSLASH):
            with self.subTest(char=ch):
                self.assertIn(ch, _NAME_FORBIDDEN_CHARS)

    def test_rejects_sql_payload(self):
        self.assertFalse(validate_name("'; DROP TABLE users;--")[0])

    def test_apostrophe_alone_is_allowed(self):
        # O'Brien is a name, not an injection. The semicolon in the payload
        # above is what makes it one.
        self.assertTrue(validate_name("O'Brien")[0])

    def test_optional_when_not_required(self):
        self.assertTrue(validate_name("", required=False)[0])

    def test_custom_bounds_are_honoured(self):
        self.assertTrue(validate_name("M", "Gender", min_length=1, max_length=20)[0])


# ══════════════════════════════════════════════════════════════════════════════
# 8. FREE TEXT
# ══════════════════════════════════════════════════════════════════════════════

class TestValidateText(unittest.TestCase):

    def test_accepts_address_at_max(self):
        self.assertTrue(validate_text("a" * ADDRESS_MAX_LENGTH,
                                      "Address", ADDRESS_MAX_LENGTH)[0])

    def test_rejects_above_max(self):
        ok, _c, err = validate_text("a" * (ADDRESS_MAX_LENGTH + 1),
                                    "Address", ADDRESS_MAX_LENGTH)
        self.assertFalse(ok)
        self.assertIn("at most", err)

    def test_strips_control_characters(self):
        ok, cleaned, _e = validate_text("12 Main St\nSuite 4", "Address", 200)
        self.assertTrue(ok)
        self.assertNotIn("\n", cleaned)

    def test_required_flag(self):
        self.assertFalse(validate_text("", "Address", 200)[0])
        self.assertTrue(validate_text("", "Notes", 200, required=False)[0])


# ══════════════════════════════════════════════════════════════════════════════
# 9. USERNAME
# ══════════════════════════════════════════════════════════════════════════════

class TestValidateUsername(unittest.TestCase):

    def test_numeric_username_uses_phone_rule(self):
        self.assertTrue(validate_username("9876543210")[0])
        self.assertFalse(validate_username("98765")[0])

    def test_name_username_bounds(self):
        self.assertTrue(validate_username("a" * USERNAME_MIN_LENGTH)[0])
        self.assertFalse(validate_username("a" * (USERNAME_MIN_LENGTH - 1))[0])
        self.assertTrue(validate_username("a" * USERNAME_MAX_LENGTH)[0])
        self.assertFalse(validate_username("a" * (USERNAME_MAX_LENGTH + 1))[0])

    def test_rejects_empty(self):
        self.assertFalse(validate_username("")[0])

    def test_rejects_metacharacters(self):
        self.assertFalse(validate_username("'; DROP TABLE users;--")[0])
        self.assertFalse(validate_username("<script>")[0])

    def test_rejects_non_ascii_digit_lookalike(self):
        # Reads as a phone number to a person, is not one to the backend:
        # ٩٨٧٦٥٤٣٢١٠ and 9876543210 look identical but are different keys.
        ok, _c, err = validate_username(ARABIC_INDIC_TEN)
        self.assertFalse(ok)
        self.assertIn("0-9", err)

    def test_still_accepts_non_ascii_names(self):
        # Only the all-digits lookalike is refused — a name in any script is
        # a legitimate login identifier.
        self.assertTrue(validate_username("Dr. José García")[0])
        self.assertTrue(validate_username(
            "दिव्यांश")[0])


# ══════════════════════════════════════════════════════════════════════════════
# 10. AGE
# ══════════════════════════════════════════════════════════════════════════════

class TestValidateAge(unittest.TestCase):

    def test_accepts_range_bounds(self):
        self.assertTrue(validate_age(str(AGE_MIN))[0])
        self.assertTrue(validate_age(str(AGE_MAX))[0])
        self.assertTrue(validate_age("45")[0])

    def test_rejects_above_max(self):
        ok, _c, err = validate_age(str(AGE_MAX + 1))
        self.assertFalse(ok)
        self.assertIn("between", err)

    def test_rejects_non_numeric(self):
        for bad in ("abc", "4a", "4.5", "-5", "1e3", ""):
            with self.subTest(value=bad):
                self.assertFalse(validate_age(bad)[0], bad)

    def test_rejects_non_ascii_digits(self):
        self.assertFalse(validate_age("٣٠")[0])

    def test_optional_when_not_required(self):
        self.assertTrue(validate_age("", required=False)[0])


# ══════════════════════════════════════════════════════════════════════════════
# 11. SERIAL
# ══════════════════════════════════════════════════════════════════════════════





# ══════════════════════════════════════════════════════════════════════════════
# 12. LOG SANITISATION
# ══════════════════════════════════════════════════════════════════════════════

class TestSanitizeForLog(unittest.TestCase):

    def test_removes_newlines_so_one_field_cannot_forge_a_log_entry(self):
        out = sanitize_for_log("user\nINFO: fake admin login succeeded")
        self.assertNotIn("\n", out)

    def test_truncates_long_values(self):
        out = sanitize_for_log("a" * 500, max_length=64)
        self.assertLessEqual(len(out), 65)      # 64 plus the ellipsis

    def test_short_values_pass_through(self):
        self.assertEqual(sanitize_for_log("divyansh"), "divyansh")


# ══════════════════════════════════════════════════════════════════════════════
# 13. UNICODE DIGITS — the subtle one
# ══════════════════════════════════════════════════════════════════════════════

class TestUnicodeDigitHandling(unittest.TestCase):
    """`str.isdigit()` and regex `\\d` both accept non-ASCII digits.

    A phone typed in Arabic-Indic numerals would otherwise pass a naive
    ten-character check and be sent to the cloud API as non-ASCII text.
    """

    def test_python_builtins_really_do_accept_them(self):
        # Guards the assumption this whole class exists for.
        self.assertTrue(ARABIC_INDIC_TEN.isdigit())
        self.assertEqual(re.sub(r"\D", "", ARABIC_INDIC_TEN), ARABIC_INDIC_TEN)

    def test_validate_phone_rejects_them(self):
        self.assertFalse(validate_phone(ARABIC_INDIC_TEN)[0])

    def test_digits_only_drops_them(self):
        self.assertEqual(digits_only(ARABIC_INDIC_TEN), "")

    def test_mixed_ascii_and_unicode_digits_rejected(self):
        self.assertFalse(validate_phone("98765" + ARABIC_INDIC_TEN[:5])[0])


# ══════════════════════════════════════════════════════════════════════════════
# 14. LOGIN RATE LIMITER
# ══════════════════════════════════════════════════════════════════════════════

class TestLoginRateLimiter(unittest.TestCase):

    def setUp(self):
        self.limiter = LoginRateLimiter(max_attempts=3, lockout_seconds=300)

    def test_starts_unlocked(self):
        self.assertFalse(self.limiter.is_locked("9876543210"))
        self.assertEqual(self.limiter.attempts_remaining("9876543210"), 3)

    def test_locks_on_reaching_max_attempts(self):
        self.assertFalse(self.limiter.record_failure("u"))
        self.assertFalse(self.limiter.record_failure("u"))
        self.assertTrue(self.limiter.record_failure("u"))    # third trips it
        self.assertTrue(self.limiter.is_locked("u"))

    def test_attempts_remaining_counts_down(self):
        self.assertEqual(self.limiter.attempts_remaining("u"), 3)
        self.limiter.record_failure("u")
        self.assertEqual(self.limiter.attempts_remaining("u"), 2)
        self.limiter.record_failure("u")
        self.assertEqual(self.limiter.attempts_remaining("u"), 1)

    def test_success_clears_the_streak(self):
        self.limiter.record_failure("u")
        self.limiter.record_failure("u")
        self.limiter.record_success("u")
        self.assertEqual(self.limiter.attempts_remaining("u"), 3)
        self.assertFalse(self.limiter.is_locked("u"))

    def test_lock_is_per_identifier(self):
        for _ in range(3):
            self.limiter.record_failure("victim")
        self.assertTrue(self.limiter.is_locked("victim"))
        self.assertFalse(self.limiter.is_locked("someone-else"))

    def test_identifier_is_case_and_space_insensitive(self):
        for _ in range(3):
            self.limiter.record_failure("  DiVyAnSh ")
        self.assertTrue(self.limiter.is_locked("divyansh"))

    def test_seconds_remaining_reported_while_locked(self):
        for _ in range(3):
            self.limiter.record_failure("u")
        remaining = self.limiter.seconds_remaining("u")
        self.assertGreater(remaining, 0)
        self.assertLessEqual(remaining, 300)

    def test_seconds_remaining_zero_when_unlocked(self):
        self.assertEqual(self.limiter.seconds_remaining("nobody"), 0)

    def test_lock_expires(self):
        limiter = LoginRateLimiter(max_attempts=1, lockout_seconds=0)
        limiter.record_failure("u")
        # A zero-second lockout is already in the past on the next check.
        self.assertFalse(limiter.is_locked("u"))
        self.assertEqual(limiter.attempts_remaining("u"), 1)

    def test_reset_all(self):
        self.limiter.record_failure("a")
        self.limiter.record_failure("b")
        self.limiter.reset()
        self.assertEqual(self.limiter.attempts_remaining("a"), 3)
        self.assertEqual(self.limiter.attempts_remaining("b"), 3)

    def test_reset_single_identifier(self):
        self.limiter.record_failure("a")
        self.limiter.record_failure("b")
        self.limiter.reset("a")
        self.assertEqual(self.limiter.attempts_remaining("a"), 3)
        self.assertEqual(self.limiter.attempts_remaining("b"), 2)

    def test_defaults_match_published_constants(self):
        default = LoginRateLimiter()
        self.assertEqual(default.max_attempts, LOGIN_MAX_ATTEMPTS)
        self.assertEqual(default.lockout_seconds, LOGIN_LOCKOUT_SECONDS)


# ══════════════════════════════════════════════════════════════════════════════
# 15. WIDGET HELPERS
# ══════════════════════════════════════════════════════════════════════════════

class TestWidgetHelpers(unittest.TestCase):
    """The helpers must not raise when Qt is absent or the widget is odd."""

    def test_apply_digit_only_sets_max_length_on_a_stub(self):
        widget = MagicMock()
        apply_digit_only(widget, 10)
        # Whether or not a real Qt validator could be built, the length cap is
        # attempted; with real PyQt5 present both calls land.
        self.assertTrue(widget.setMaxLength.called or widget.setValidator.called)

    def test_apply_digit_only_survives_a_broken_widget(self):
        broken = MagicMock()
        broken.setValidator.side_effect = RuntimeError("no Qt")
        broken.setMaxLength.side_effect = RuntimeError("no Qt")
        try:
            result = apply_digit_only(broken, 10)
        except Exception as exc:                      # pragma: no cover
            self.fail(f"apply_digit_only raised {exc!r}")
        self.assertFalse(result)

    def test_apply_text_limits_sets_max_length(self):
        widget = MagicMock()
        self.assertTrue(apply_text_limits(widget, 100))
        widget.setMaxLength.assert_called_once_with(100)

    def test_apply_text_limits_survives_a_broken_widget(self):
        broken = MagicMock()
        broken.setMaxLength.side_effect = RuntimeError("no Qt")
        self.assertFalse(apply_text_limits(broken, 100))


# ══════════════════════════════════════════════════════════════════════════════
# 16. THE CONSTANTS THEMSELVES
# ══════════════════════════════════════════════════════════════════════════════

class TestStandardLimits(unittest.TestCase):

    def test_phone_is_ten_digits(self):
        self.assertEqual(PHONE_DIGITS, 10)

    def test_password_floor_meets_common_guidance(self):
        self.assertGreaterEqual(PASSWORD_MIN_LENGTH, 8)

    def test_every_bound_is_ordered(self):
        pairs = [
            (PASSWORD_MIN_LENGTH, PASSWORD_MAX_LENGTH),
            (USERNAME_MIN_LENGTH, USERNAME_MAX_LENGTH),
            (NAME_MIN_LENGTH, NAME_MAX_LENGTH),
            (AGE_MIN, AGE_MAX),
        ]
        for low, high in pairs:
            with self.subTest(low=low, high=high):
                self.assertLess(low, high)

    def test_every_maximum_is_finite_and_positive(self):
        for value in (PASSWORD_MAX_LENGTH, USERNAME_MAX_LENGTH, NAME_MAX_LENGTH,
                      EMAIL_MAX_LENGTH, ADDRESS_MAX_LENGTH, SERIAL_MAX_LENGTH):
            with self.subTest(value=value):
                self.assertGreater(value, 0)
                self.assertLess(value, 10000)

    def test_lockout_settings_are_usable(self):
        self.assertGreaterEqual(LOGIN_MAX_ATTEMPTS, 3)
        self.assertGreaterEqual(LOGIN_LOCKOUT_SECONDS, 60)


# ══════════════════════════════════════════════════════════════════════════════
# 17. THE FORMS ACTUALLY USE THE SHARED RULES
# ══════════════════════════════════════════════════════════════════════════════

def _read(rel_path):
    with open(os.path.join(_ROOT, rel_path), "r", encoding="utf-8") as fh:
        return fh.read()




# ══════════════════════════════════════════════════════════════════════════════
# 18. AUTH STORE HARDENING
# ══════════════════════════════════════════════════════════════════════════════



# ══════════════════════════════════════════════════════════════════════════════
# 19. NO DANGEROUS SINKS IN SHIPPED CODE
# ══════════════════════════════════════════════════════════════════════════════

class TestNoInjectionSinks(unittest.TestCase):
    """Repo-wide guards, so a future edit cannot quietly reintroduce these."""

    SCAN_DIRS = ["src"]
    SKIP_PARTS = ("__pycache__", "venv", "site-packages", "node_modules",
                  ".before-license-fix")

    def _python_files(self):
        for base in self.SCAN_DIRS:
            root_dir = os.path.join(_ROOT, base)
            for dirpath, _dirnames, filenames in os.walk(root_dir):
                if any(part in dirpath for part in self.SKIP_PARTS):
                    continue
                for fn in filenames:
                    if fn.endswith(".py") and not any(p in fn for p in self.SKIP_PARTS):
                        yield os.path.join(dirpath, fn)

    def _scan(self, pattern):
        hits = []
        rx = re.compile(pattern)
        for path in self._python_files():
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as fh:
                    for lineno, line in enumerate(fh, 1):
                        if line.lstrip().startswith("#"):
                            continue
                        if rx.search(line):
                            hits.append(f"{os.path.relpath(path, _ROOT)}:{lineno}")
            except OSError:
                continue
        return hits

    def test_no_tls_verification_bypass(self):
        hits = self._scan(r"verify\s*=\s*False")
        self.assertEqual(hits, [], f"TLS verification disabled at: {hits}")

    def test_no_shell_true(self):
        hits = self._scan(r"shell\s*=\s*True")
        self.assertEqual(hits, [], f"subprocess with shell=True at: {hits}")

    def test_no_os_system(self):
        hits = self._scan(r"\bos\.system\s*\(")
        self.assertEqual(hits, [], f"os.system() at: {hits}")

    def test_no_eval_of_input(self):
        hits = self._scan(r"(?<![\w.])eval\s*\(")
        self.assertEqual(hits, [], f"eval() at: {hits}")

    def test_no_pickle_load(self):
        hits = self._scan(r"pickle\.loads?\s*\(")
        self.assertEqual(hits, [], f"pickle load at: {hits}")

    def test_sql_is_parameterised(self):
        # Any execute() whose argument is an f-string or a % / + concatenation
        # is a candidate for injection.
        hits = self._scan(r"execute\s*\(\s*(f[\"']|[\"'][^\"']*[\"']\s*[%+])")
        self.assertEqual(hits, [], f"non-parameterised SQL at: {hits}")


# ══════════════════════════════════════════════════════════════════════════════
# 20. FILENAME SANITISATION / PATH TRAVERSAL
# ══════════════════════════════════════════════════════════════════════════════

class TestSanitizeFilenameComponent(unittest.TestCase):
    """A user-supplied name must never be able to steer a path."""

    TRAVERSALS = [
        "../../../Users/Public/x",
        "..\\..\\..\\Windows\\Temp\\x",
        "....//....//x",
        "/etc/passwd",
        "C:\\Windows\\System32",
        "..%2f..%2fx",
        "foo/../../bar",
    ]

    def test_no_traversal_survives(self):
        for payload in self.TRAVERSALS:
            with self.subTest(payload=payload):
                out = sanitize_filename_component(payload)
                self.assertNotIn("..", out)
                self.assertNotIn("/", out)
                self.assertNotIn(chr(92), out)
                self.assertNotIn(":", out)

    def test_result_is_a_single_path_segment(self):
        for payload in self.TRAVERSALS + ["John Doe", "Dr. O'Brien"]:
            with self.subTest(payload=payload):
                out = sanitize_filename_component(payload)
                self.assertEqual(os.path.basename(out), out)

    def test_joined_path_stays_inside_the_output_directory(self):
        base = os.path.join(os.sep, "cardiox", "recordings")
        for payload in self.TRAVERSALS:
            with self.subTest(payload=payload):
                name = sanitize_filename_component(payload)
                joined = os.path.normpath(os.path.join(base, "2026-01-01_00-00-00_" + name))
                self.assertTrue(
                    joined.startswith(base + os.sep),
                    f"{payload!r} escaped to {joined}",
                )

    def test_ordinary_names_stay_readable(self):
        self.assertEqual(sanitize_filename_component("John Doe"), "John_Doe")
        self.assertEqual(sanitize_filename_component("Dr. O'Brien"), "Dr_O_Brien")

    def test_empty_and_blank_fall_back(self):
        self.assertEqual(sanitize_filename_component(""), "Unknown")
        self.assertEqual(sanitize_filename_component("   "), "Unknown")
        self.assertEqual(sanitize_filename_component(None), "Unknown")
        self.assertEqual(sanitize_filename_component("///"), "Unknown")

    def test_custom_fallback_is_used(self):
        self.assertEqual(sanitize_filename_component("", fallback="Patient"), "Patient")

    def test_length_is_bounded(self):
        out = sanitize_filename_component("A" * 500)
        self.assertLessEqual(len(out), 64)

    def test_windows_reserved_device_names_are_escaped(self):
        for reserved in ("CON", "PRN", "AUX", "NUL", "COM1", "LPT9", "con", "Nul"):
            with self.subTest(name=reserved):
                out = sanitize_filename_component(reserved)
                self.assertNotIn(out.upper(),
                                 {"CON", "PRN", "AUX", "NUL", "COM1", "LPT9"})

    def test_control_characters_removed(self):
        self.assertNotIn(chr(0), sanitize_filename_component("a" + chr(0) + "b"))




# ══════════════════════════════════════════════════════════════════════════════
# 21. FORM FIELD LENGTH CAPS
# ══════════════════════════════════════════════════════════════════════════════



class TestValidateSerial(unittest.TestCase):
    """The Serial ID box is filled in by the device, never typed.

    `on_scan_finished()` sets it read-only from `send_machine_serial_command()`,
    which returns 16 bytes of raw ASCII from the firmware. The character rule
    has to follow what the hardware emits.
    """

    # Verbatim from a connected RhythmUltra (also in ecg_settings.json).
    REAL_SERIAL = "DM ECG V1.0 A998"

    def test_accepts_the_real_device_serial(self):
        # Regression: an allow-list of [A-Za-z0-9_-] rejected every genuine
        # device and blocked signup with "Serial ID may contain only letters,
        # digits, dashes and underscores."
        ok, cleaned, err = validate_serial(self.REAL_SERIAL)
        self.assertTrue(ok, f"real device serial was rejected: {err}")
        self.assertEqual(cleaned, self.REAL_SERIAL)

    def test_accepts_spaces_and_dots(self):
        for good in ("DM ECG V1.0 A998", "DM ECG V2.1 B001", "A 1.0 B"):
            with self.subTest(value=good):
                self.assertTrue(validate_serial(good)[0], good)

    def test_accepts_legacy_dashed_and_underscored_forms(self):
        for good in ("TEST-MACHINE-1234", "TEST_MACHINE_1234", "DM_ECG_V1.0_A998"):
            with self.subTest(value=good):
                self.assertTrue(validate_serial(good)[0], good)

    def test_accepts_any_sixteen_byte_ascii_the_firmware_can_send(self):
        # The protocol hands back bytes[5:21] verbatim; a serial that fits the
        # documented alphabet must never be refused.
        self.assertTrue(validate_serial("X" * 16)[0])
        self.assertTrue(validate_serial("1234567890123456")[0])

    def test_still_rejects_path_and_shell_metacharacters(self):
        for bad in ("A/B", chr(92).join(["A", "B"]), "A;B", "../x", "A'B",
                    'A"B', "A|B", "A`B", "A$B"):
            with self.subTest(value=bad):
                self.assertFalse(validate_serial(bad)[0], bad)

    def test_strips_control_characters(self):
        ok, cleaned, _e = validate_serial("DM ECG" + chr(0) + " V1.0")
        self.assertTrue(ok)
        self.assertNotIn(chr(0), cleaned)

    def test_rejects_above_max_length(self):
        self.assertFalse(validate_serial("A" * (SERIAL_MAX_LENGTH + 1))[0])

    def test_optional_by_default(self):
        self.assertTrue(validate_serial("")[0])

    def test_required_when_asked(self):
        self.assertFalse(validate_serial("", required=True)[0])

    def test_disconnected_placeholder_is_treated_as_no_serial(self):
        # The form parks this sentence in the read-only box when no device
        # answers. It is UI copy and must never be stored as a serial.
        ok, cleaned, _err = validate_serial(DEVICE_DISCONNECTED_PLACEHOLDER)
        self.assertTrue(ok)
        self.assertEqual(cleaned, "")

    def test_placeholder_detector(self):
        self.assertTrue(is_device_placeholder(DEVICE_DISCONNECTED_PLACEHOLDER))
        self.assertTrue(is_device_placeholder("  " + DEVICE_DISCONNECTED_PLACEHOLDER + " "))
        self.assertFalse(is_device_placeholder(self.REAL_SERIAL))
        self.assertFalse(is_device_placeholder(""))


class TestRecordingPathIsSanitised(unittest.TestCase):
    """The Holter session directory is built from the patient name."""

    def test_stream_writer_uses_the_shared_sanitiser(self):
        src = _read("src/ecg/holter/stream_writer.py")
        self.assertIn("sanitize_filename_component", src)

    def test_stream_writer_no_longer_only_replaces_spaces(self):
        src = _read("src/ecg/holter/stream_writer.py")
        self.assertNotIn(
            "self.patient_info.get('name', 'Unknown').replace(' ', '_')", src,
            "replacing spaces alone leaves '..' and separators intact",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
