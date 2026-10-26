import ast
import json
import re
import unittest
from pathlib import Path

from app.models.llm_provider import get_llm_provider
from app.utils import utils


ROOT_DIR = Path(__file__).parent.parent.parent
WEBUI_MAIN = ROOT_DIR / "webui" / "Main.py"
I18N_DIR = ROOT_DIR / "webui" / "i18n"
LLM_PROVIDER_TIPS_PREFIX = "llm_provider_tips."
TTS_PROVIDER_TIPS_PREFIX = "tts_provider_tips."
SECONDARY_LOCALES = ("az", "ca", "de", "es", "fr", "id", "it", "ko", "pt", "ru", "tr", "vi")
PROVIDER_TIPS_PREFIXES = (
    LLM_PROVIDER_TIPS_PREFIX,
    TTS_PROVIDER_TIPS_PREFIX,
)
# Partner Provider brand names and detailed descriptions are maintained in Chinese and English only. Secondary locales
# fall back to English uniformly to avoid duplicating identical brand names across a dozen files and partial updates later.
ENGLISH_FALLBACK_KEYS = frozenset(
    {
        "Auto (AI Decided)",
        "Slow Zoom (Default)",
        "Pan Left",
        "Pan Right",
        "Tilt Up",
        "Tilt Down",
        "Cross Dissolve",
        "Jump Cut",
        "Ripple Dissolve",
        "Iris In",
        "Iris Out",
        "Iris Box",
        "Clock Wipe",
        "Wipe Left",
        "Wipe Right",
        "Cross Zoom",
        "Dip to Black",
        "Dip to White",
        "Logs",
        "Transitions & Effects",
        "AI-Selected Scene Transitions & Camera Motions",
        "Scene",
        "Source Material",
        "Camera Motion",
        "Scene Transition",
        "Transitions & effects will be displayed here as clips are composed.",
        "Diagnostic Logging",
        "Diagnostic Logging Help",
        "Log Level",
        "Trace API Requests",
        "Trace API Requests Help",
        "Apply Logging Settings",
        "Logging Settings Applied",
        "AI Video Quote Required",
        "AI Video Quote Retained For Retry",
        "AI Video Quote Estimate Incomplete",
        "AI Video Quote Summary",
        "AI Video Quote Summary Singular",
        "AI Video Model",
        "AI Video Model Reference Price",
        "AI Video Model List Load Failed",
        "AI Video Duration Basis Actual",
        "AI Video Duration Basis Estimated",
        "AI Video Material Coverage",
        "AI Video Scene Count",
        "Confirm AI Video Charge",
        "Confirm AI Video Charge Help",
        "Confirm AI Video Charge Required",
        "Custom API Endpoint",
        "API Platform",
        "llm_provider_endpoint_selector.moonshot",
        "llm_provider_endpoint_selector_help.moonshot",
        "llm_provider_endpoint.moonshot.china",
        "llm_provider_endpoint.moonshot.global",
        "llm_provider_authentication_error.moonshot",
        "Local LLM Script Generation",
        "llm_provider_label.apimart",
        "llm_provider_label.openrouter",
        "llm_provider_label.api_route",
        "llm_provider_label.fluxionai",
        "llm_provider_label.cheaperinference",
        "llm_provider_label.requesty",
        "llm_provider_label.futureinfra",
        "llm_provider_label.yapi",
        "llm_provider_label.opper",
        "llm_provider_label.onomeo",
        "llm_provider_label.empiriolabs",
        "llm_provider_label.shengsuanyun",
        "llm_provider_label.muse_spark",
        "llm_provider_label.kimi_code",
        "kimi_code.oauth.login",
        "kimi_code.oauth.pending",
        "kimi_code.oauth.open_page",
        "kimi_code.oauth.success",
        "kimi_code.oauth.authorized",
        "kimi_code.oauth.disconnect",
        "kimi_code.oauth.expired",
        "kimi_code.oauth.denied",
        "kimi_code.oauth.reauth",
        "kimi_code.oauth.region_mismatch",
        "kimi_code.oauth.network_error",
        "kimi_code.oauth.poll_retry",
        "kimi_code.oauth.changed",
        "LoomLoom Poll Retry Pending",
        "LoomLoom Poll Retry Warning",
        "Resume LoomLoom Status Check",
        "Refresh AI Video Models",
        "Retry AI Video Quote",
        "LoomLoom Quote Summary Singular",
        "LoomLoom Video Terms Reuse Help",
        "Metaso MiniMax H3",
        "Metaso MiniMax H3 Help",
        "Metaso MiniMax API Key",
        "Metaso MiniMax API Key Help",
        "Metaso MiniMax Base URL",
        "Metaso MiniMax Resolution",
        "Metaso MiniMax Resolution Help",
        "Metaso MiniMax Invalid Resolution",
        "Select Metaso MiniMax Resolution",
        "Please Enter the Metaso MiniMax API Key",
        "Metaso MiniMax Billing Notice",
        "Metaso MiniMax Billing Notice Uploaded Audio",
        "Metaso MiniMax Billing Notice Without Script",
        "Confirm Metaso MiniMax Charge",
        "Confirm Metaso MiniMax Charge Help",
        "Confirm Metaso MiniMax Charge Required",
        "MuAPI AI Video",
        "MuAPI AI Video Help",
        "MuAPI API Key",
        "MuAPI API Key Help",
        "MuAPI Base URL",
        "MuAPI Base URL Help",
        "MuAPI Video Endpoint",
        "MuAPI Video Endpoint Help",
        "MuAPI Resolution",
        "MuAPI Resolution Help",
        "Please Enter the MuAPI API Key",
        "MuAPI Billing Notice",
        "MuAPI Billing Notice Without Script",
        "Confirm MuAPI Charge",
        "Confirm MuAPI Charge Help",
        "Confirm MuAPI Charge Required",
        "Script Generation Method",
        "Script Generation Method Help",
        "Shengsuan Cloud AI Video",
        "Shengsuan Cloud AI Video Help",
        "Shengsuan Cloud API Key",
        "Shengsuan Cloud API Key Help",
        "Shengsuan Cloud API Key Link",
        "Shengsuan Cloud API Key Placeholder",
        "Shengsuan Cloud API Key Required",
        "Shengsuan Cloud API Key Reused",
        "Shengsuan Cloud Batch Script Generation",
        "Selected AI Video Model Unavailable",
        "Selected AI Video Ratio Unavailable",
        "Stop Tracking LoomLoom Run",
        "Stop Tracking LoomLoom Run Help",
        "Unavailable AI Video Model",
        "VoxCPM Speed Not Supported",
        "VoxCPM Reference Audio",
        "VoxCPM Reference Audio Help",
        "VoxCPM Reference Audio Notice",
        "VoxCPM Reference Audio Empty",
        "VoxCPM Reference Audio Upload Too Large",
        "Validating VoxCPM Reference Audio",
        "VoxCPM Reference Audio Invalid",
        "VoxCPM High Fidelity Delivery",
        "VoxCPM High Fidelity Delivery Help",
        "VoxCPM Separate Prompt Audio",
        "VoxCPM Separate Prompt Audio Help",
        "VoxCPM Prompt Audio",
        "VoxCPM Prompt Audio Help",
        "VoxCPM Prompt Text",
        "VoxCPM Prompt Text Help",
        "Transcribe VoxCPM Prompt Audio",
        "Transcribe VoxCPM Prompt Audio Help",
        "Transcribing VoxCPM Prompt Audio",
        "VoxCPM Prompt Audio Transcribed",
        "VoxCPM Prompt Audio Transcription Failed",
        "VoxCPM Prompt Transcript Review",
        "VoxCPM Prompt Audio Empty",
        "VoxCPM Prompt Audio Upload Too Large",
        "Validating VoxCPM Prompt Audio",
        "VoxCPM Prompt Text Required",
        "VoxCPM Prompt Pair Required",
        "VoxCPM Prompt Invalid",
        "None (Animation)",
    }
)
FORMAT_PLACEHOLDER_PATTERN = re.compile(r"(?<!\{)\{([a-zA-Z_][a-zA-Z0-9_]*)\}(?!\})")
MARKDOWN_URL_PATTERN = re.compile(r"\[[^\]]+\]\((https?://[^)]+)\)")


class _TrKeyVisitor(ast.NodeVisitor):
    def __init__(self):
        self.keys = set()

    def visit_Call(self, node):
        if (
            isinstance(node.func, ast.Name)
            and node.func.id == "tr"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            self.keys.add(node.args[0].value)
        self.generic_visit(node)


def _load_translation(locale):
    data = json.loads((I18N_DIR / f"{locale}.json").read_text(encoding="utf-8"))
    return data.get("Translation", {})


def _duplicate_translation_keys(path):
    """Return duplicate keys defined in raw locale text; JSON parsing keeps only the last occurrence."""
    duplicates = []

    def collect(pairs):
        seen = set()
        for key, _ in pairs:
            if key in seen:
                duplicates.append(key)
            seen.add(key)
        return dict(pairs)

    json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=collect)
    return duplicates


def _required_translation_keys(translations):
    """Return keys that secondary languages must maintain; provider detailed descriptions fall back to English."""
    return {
        key
        for key in translations
        if key not in ENGLISH_FALLBACK_KEYS
        and not key.startswith(PROVIDER_TIPS_PREFIXES)
    }


def _format_placeholders(value):
    """Extract runtime format variables to prevent translation omissions or accidental variable renames."""
    return set(FORMAT_PLACEHOLDER_PATTERN.findall(value))


def _markdown_urls(value):
    """Extract Markdown link targets, allowing translation of link text while preventing URL corruption."""
    return set(MARKDOWN_URL_PATTERN.findall(value))


class TestWebuiI18n(unittest.TestCase):
    def test_catalan_locale_is_discovered_and_matches_browser_variants(self):
        """Locale files auto-register; regional variants fall back to ca without overriding saved user choices."""
        locales = utils.load_locales(str(I18N_DIR))
        self.assertEqual(locales["ca"]["Language"], "Català")
        for browser_locale in ("ca", "ca-ES", "ca_AD", "CA-es", "ca-ES-valencia"):
            with self.subTest(browser_locale=browser_locale):
                self.assertEqual(
                    utils.resolve_ui_language("", browser_locale, locales), "ca"
                )
        self.assertEqual(utils.resolve_ui_language("en", "ca-ES", locales), "en")
        self.assertEqual(utils.resolve_ui_language("ca", "en-US", locales), "ca")

    def test_catalan_pause_examples_use_supported_syntax(self):
        """Pause tags in help text must be parseable without translating keywords into unsupported syntax."""
        help_text = _load_translation("ca")["Video Script Help"]
        examples = re.findall(r"\[pausa: [^\]]+\]", help_text)
        self.assertTrue(examples)
        for example in examples:
            with self.subTest(example=example):
                self.assertTrue(utils.has_pause_tags(example))
                self.assertEqual(utils.remove_pause_tags(example).strip(), "")

    def test_saved_ui_language_takes_priority_over_browser_locale(self):
        language = utils.resolve_ui_language(
            saved_language="de",
            browser_locale="zh-CN",
            supported_languages=["zh", "en", "de"],
        )

        self.assertEqual(language, "de")

    def test_browser_locale_is_normalized_to_supported_base_language(self):
        self.assertEqual(
            utils.resolve_ui_language("", "zh-CN", ["zh", "en"]),
            "zh",
        )
        self.assertEqual(
            utils.resolve_ui_language(None, "pt_BR", ["en", "pt"]),
            "pt",
        )

    def test_unsupported_browser_locale_falls_back_to_english(self):
        language = utils.resolve_ui_language(
            saved_language="",
            browser_locale="fr-FR",
            supported_languages=["zh", "en"],
        )

        self.assertEqual(language, "en")

    def test_english_locale_covers_static_webui_labels(self):
        tree = ast.parse(WEBUI_MAIN.read_text(encoding="utf-8"))
        visitor = _TrKeyVisitor()
        visitor.visit(tree)

        en_keys = set(_load_translation("en"))

        self.assertEqual(sorted(visitor.keys - en_keys), [])

    def test_shengsuanyun_provider_tips_keep_registration_and_model_links(self):
        """Partner entry points and model catalogs are product configs; prevent accidental removal of tracking links when updating copy."""
        expected_urls = {
            "https://www.shengsuanyun.com/?from=CH_XUQ4OTSK",
            "https://global.modelmesh.info/model",
        }

        for locale in ("zh", "en"):
            with self.subTest(locale=locale):
                tips = _load_translation(locale)["llm_provider_tips.shengsuanyun"]
                provider = get_llm_provider("shengsuanyun")
                rendered = tips.format(
                    api_key_url=provider.effective_api_key_url(),
                    default_base_url=provider.effective_default_base_url,
                    default_model=provider.default_model,
                )
                self.assertEqual(_markdown_urls(rendered), expected_urls)

    def test_metaso_api_key_label_keeps_mpt_referral_link(self):
        """Metaso Key acquisition entry point must retain MPT referral parameters to preserve sponsorship conversion attribution."""
        expected_url = "https://metaso.cn/minimax-h3/?s=MPT"

        for locale in ("zh", "en"):
            with self.subTest(locale=locale):
                label = _load_translation(locale)["Metaso MiniMax API Key"]
                self.assertEqual(_markdown_urls(label), {expected_url})

    def test_secondary_locales_cover_english_locale(self):
        en_translations = _load_translation("en")
        required_en_keys = _required_translation_keys(en_translations)

        for locale in SECONDARY_LOCALES:
            with self.subTest(locale=locale):
                locale_keys = set(_load_translation(locale))
                self.assertEqual(sorted(required_en_keys - locale_keys), [])

    def test_secondary_locales_do_not_duplicate_provider_tips(self):
        # Provider configuration detailed descriptions are maintained in Chinese and English only; other languages fall back to English at runtime.
        # Copying these keys is prohibited to avoid unmaintained half-translated content.
        for locale in SECONDARY_LOCALES:
            with self.subTest(locale=locale):
                locale_keys = set(_load_translation(locale))
                duplicated_keys = sorted(
                    key for key in locale_keys if key.startswith(PROVIDER_TIPS_PREFIXES)
                )
                self.assertEqual(duplicated_keys, [])

    def test_secondary_locales_do_not_duplicate_english_fallback_keys(self):
        for locale in SECONDARY_LOCALES:
            with self.subTest(locale=locale):
                locale_keys = set(_load_translation(locale))
                self.assertEqual(sorted(ENGLISH_FALLBACK_KEYS & locale_keys), [])

    def test_secondary_locales_cover_static_webui_labels(self):
        tree = ast.parse(WEBUI_MAIN.read_text(encoding="utf-8"))
        visitor = _TrKeyVisitor()
        visitor.visit(tree)

        for locale in SECONDARY_LOCALES:
            with self.subTest(locale=locale):
                locale_keys = set(_load_translation(locale))
                self.assertEqual(
                    sorted(visitor.keys - locale_keys - ENGLISH_FALLBACK_KEYS),
                    [],
                )

    def test_secondary_locales_preserve_format_placeholders(self):
        en_translations = _load_translation("en")

        for locale in SECONDARY_LOCALES:
            locale_translations = _load_translation(locale)
            for key in _required_translation_keys(en_translations):
                with self.subTest(locale=locale, key=key):
                    self.assertEqual(
                        _format_placeholders(locale_translations[key]),
                        _format_placeholders(en_translations[key]),
                    )

    def test_secondary_locales_preserve_markdown_urls(self):
        en_translations = _load_translation("en")

        for locale in SECONDARY_LOCALES:
            locale_translations = _load_translation(locale)
            for key in _required_translation_keys(en_translations):
                with self.subTest(locale=locale, key=key):
                    self.assertEqual(
                        _markdown_urls(locale_translations[key]),
                        _markdown_urls(en_translations[key]),
                    )

    def test_script_language_options_include_russian_and_catalan(self):
        tree = ast.parse(WEBUI_MAIN.read_text(encoding="utf-8"))
        support_locales = None

        for node in tree.body:
            if not isinstance(node, ast.Assign):
                continue
            if any(
                isinstance(target, ast.Name) and target.id == "support_locales"
                for target in node.targets
            ):
                support_locales = ast.literal_eval(node.value)
                break

        self.assertIsNotNone(support_locales)
        self.assertIn("ru-RU", support_locales)
        self.assertIn("ca-ES", support_locales)
        self.assertIn("hi-IN", support_locales)

    def test_locale_files_do_not_redefine_a_translation_key(self):
        """
        When duplicate keys appear in the same JSON object, parsing retains only the last one, silently discarding preceding ones.
        Video transitions and subtitle animations previously shared the "None" key, causing the Chinese transition dropdown to display as "No Animation".
        Inspect raw locale text directly here to prevent similar overrides from escaping tr() key coverage tests.
        """

        for path in sorted(I18N_DIR.glob("*.json")):
            with self.subTest(locale=path.stem):
                self.assertEqual(_duplicate_translation_keys(path), [])
