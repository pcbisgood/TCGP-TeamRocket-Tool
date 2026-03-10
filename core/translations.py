# =========================================================================
# INTERNATIONALIZATION (i18n) SYSTEM
# =========================================================================

from typing import Optional
import json
import os
import locale


def get_available_languages():
    """Returns a list of available languages based on JSON files in /locales."""
    locales_dir = os.path.join(os.path.dirname(__file__), 'locales')
    languages = []

    if os.path.exists(locales_dir):
        for file in os.listdir(locales_dir):
            if file.endswith(".json"):
                lang_code = file.replace(".json", "")
                languages.append(lang_code)

    return languages


def get_system_language():
    """Detects the system language."""
    try:
        system_locale = locale.getdefaultlocale()[0]
        if system_locale:
            lang = system_locale.split('_')[0].lower()
            if lang in get_available_languages():
                return lang
    except:
        pass

    # Default language: first available, or 'en' fallback
    langs = get_available_languages()
    return langs[0] if langs else 'en'


class Translator:
    """Translation manager."""

    def __init__(self, language=None):
        self.available_languages = get_available_languages()
        self.language = language or get_system_language()
        self.translations = {}
        self.load_translations()

    def load_translations(self):
        """Loads translation data from the JSON files."""
        locales_dir = os.path.join(os.path.dirname(__file__), 'locales')
        lang_file = os.path.join(locales_dir, f'{self.language}.json')

        if os.path.exists(lang_file):
            try:
                with open(lang_file, 'r', encoding='utf-8') as f:
                    self.translations = json.load(f)
            except Exception as e:
                print(f"⚠️ Error loading translations: {e}")
                self.translations = {}
        else:
            # Try fallback language
            fallback_lang = 'en' if 'en' in self.available_languages else (
                self.available_languages[0] if self.available_languages else None
            )
            if fallback_lang:
                fallback_file = os.path.join(locales_dir, f'{fallback_lang}.json')
                try:
                    with open(fallback_file, 'r', encoding='utf-8') as f:
                        self.translations = json.load(f)
                except:
                    self.translations = {}
            else:
                self.translations = {}

    def t(self, key, **kwargs):
        """
        Translates a key.

        Args:
            key: Translation key (supports nested keys: "ui.channel_id")
            **kwargs: Variables to inject into the translated string

        Returns:
            Translated string or the key itself if missing
        """
        parts = key.split('.')
        value = self.translations

        for part in parts:
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return key

        if not isinstance(value, str):
            return key

        if kwargs:
            try:
                value = value.format(**kwargs)
            except:
                pass

        return value

    def set_language(self, language):
        """Changes the active language."""
        if language in self.available_languages:
            self.language = language
            self.load_translations()


# Global translator instance
_translator = Translator()


def t(key, **kwargs):
    """Global translation helper."""
    return _translator.t(key, **kwargs)


def set_language(language):
    """Sets the global language."""
    _translator.set_language(language)


def get_language():
    """Returns the current global language."""
    return _translator.language


def get_languages():
    """Returns available languages."""
    return _translator.available_languages
