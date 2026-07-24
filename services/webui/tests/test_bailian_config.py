from pathlib import Path
import tempfile
import unittest

from joy_interaction_webui.bailian_config import (
    BAILIAN_ASR_MODEL,
    BAILIAN_MODELS,
    BAILIAN_PROVIDER,
    BAILIAN_TTS_MODEL,
    BailianConfigurationError,
    load_bailian_speech_api_key,
    load_bailian_api_key,
    resolve_provider,
)
from joy_interaction_webui.asr import build_bailian_asr_payload
from joy_interaction_webui.tts import build_bailian_tts_payload, pcm16_to_wav_bytes, wav_to_pcm16


class BailianConfigTests(unittest.TestCase):
    def test_environment_key_overrides_file(self):
        with tempfile.TemporaryDirectory() as directory:
            key_file = Path(directory) / "API.txt"
            key_file.write_text("说明\nsk-file.value-from-file\n", encoding="utf-8")

            self.assertEqual(
                load_bailian_api_key(
                    key_file=key_file, environ={"OPENAI_API_KEY": "sk-env.value"}
                ),
                "sk-env.value",
            )

    def test_key_file_preserves_complete_key_line(self):
        with tempfile.TemporaryDirectory() as directory:
            key_file = Path(directory) / "API.txt"
            expected = "sk-test.segment.with.dots-and-dashes"
            key_file.write_text(f"说明文字\n  {expected}  \n", encoding="utf-8")

            self.assertEqual(load_bailian_api_key(key_file=key_file, environ={}), expected)

    def test_speech_key_prefers_dashscope_environment_key(self):
        self.assertEqual(
            load_bailian_speech_api_key(
                environ={"DASHSCOPE_API_KEY": "sk-speech.key.with.dots"}
            ),
            "sk-speech.key.with.dots",
        )

    def test_key_file_requires_sk_prefix(self):
        for content in ("说明文字\n", "not-a-key\n"):
            with self.subTest(content=content), tempfile.TemporaryDirectory() as directory:
                key_file = Path(directory) / "API.txt"
                key_file.write_text(content, encoding="utf-8")

                with self.assertRaisesRegex(BailianConfigurationError, "does not contain"):
                    load_bailian_api_key(key_file=key_file, environ={})

    def test_missing_key_file_is_safe_error(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(BailianConfigurationError, "was not found"):
                load_bailian_api_key(key_file=Path(directory) / "missing.txt", environ={})

    def test_provider_defaults_to_bailian_and_rejects_unknown(self):
        self.assertEqual(resolve_provider(""), BAILIAN_PROVIDER)
        with self.assertRaisesRegex(ValueError, "Unsupported VLM provider"):
            resolve_provider("unsupported")

    def test_supported_bailian_models_are_explicit(self):
        self.assertEqual(
            BAILIAN_MODELS,
            ("qwen3.7-plus", "qwen3.6-plus", "qwen3.6-flash"),
        )

    def test_bailian_asr_request_uses_wav_data_url_and_fixed_model(self):
        payload = build_bailian_asr_payload(b"\x00\x00\x01\x00")
        audio_data = payload["input"]["messages"][0]["content"][0]["audio"]

        self.assertEqual(BAILIAN_ASR_MODEL, "fun-asr-realtime")
        self.assertTrue(audio_data.startswith("data:audio/wav;base64,"))
        self.assertNotIn("127.0.0.1", audio_data)

    def test_bailian_tts_payload_and_pcm_conversion(self):
        payload = build_bailian_tts_payload("你好")
        pcm = b"\x00\x00\x01\x00"
        converted_pcm, sample_rate = wav_to_pcm16(pcm16_to_wav_bytes(pcm, 24000))

        self.assertEqual(payload["model"], BAILIAN_TTS_MODEL)
        self.assertEqual(BAILIAN_TTS_MODEL, "qwen-audio-3.0-tts-plus")
        self.assertEqual(payload["input"]["text"], "你好")
        self.assertEqual(payload["input"]["format"], "wav")
        self.assertEqual(converted_pcm, pcm)
        self.assertEqual(sample_rate, 24000)
