
"""TTS bridge for streaming VLM text responses as browser-playable PCM audio."""

import asyncio
import base64
import io
import json
import logging
import os
import uuid
import wave

import aiohttp
from aiohttp import web

from .bailian_config import (
    BAILIAN_PROVIDER,
    BAILIAN_TTS_API_URL,
    BAILIAN_TTS_MODEL,
    load_bailian_speech_api_key,
)

# TTS parameters
TTS_URL = os.getenv("TTS_URL", "ws://127.0.0.1:8992/ws/tts")
TTS_SAMPLE_RATE = int(os.getenv("TTS_SAMPLE_RATE", "24000"))
TTS_VOICE = os.getenv("TTS_VOICE", "default")
TTS_EMOTION = os.getenv("TTS_EMOTION", "{{高兴}}")
TTS_CHUNK_SIZE = int(os.getenv("TTS_CHUNK_SIZE", "12"))
TTS_OPEN_TIMEOUT = float(os.getenv("TTS_OPEN_TIMEOUT", "10.0"))
TTS_TIMEOUT = float(os.getenv("TTS_TIMEOUT", "120.0"))
TTS_STREAM_IDLE_TIMEOUT = float(os.getenv("TTS_STREAM_IDLE_TIMEOUT", "60.0"))
TTS_CANCEL_TIMEOUT = float(os.getenv("TTS_CANCEL_TIMEOUT", "1.0"))
TTS_MAX_TEXT_CHARS = int(os.getenv("TTS_MAX_TEXT_CHARS", "2000"))
TTS_TEMPERATURE = float(os.getenv("TTS_TEMPERATURE", "0.7"))
TTS_MAX_TOKENS = int(os.getenv("TTS_MAX_TOKENS", "1024"))
TTS_INSTRUCTIONS = os.getenv(
    "TTS_INSTRUCTIONS",
    (
        "Please speak at a slightly faster pace, around 1.2x normal speed, "
        "while keeping pronunciation clear and natural."
    ),
)
TTS_PROXY = os.getenv("TTS_PROXY", "").strip() or None
TTS_TRUST_ENV = os.getenv("TTS_TRUST_ENV", "1").lower() not in {
    "0",
    "false",
    "no",
}
BAILIAN_TTS_VOICE = os.getenv("BAILIAN_TTS_VOICE", "longanhuan_v3.6")
BAILIAN_TTS_HTTP_TIMEOUT = float(os.getenv("BAILIAN_TTS_HTTP_TIMEOUT", "90"))
BAILIAN_TTS_FORWARD_CHUNK_SIZE = int(os.getenv("BAILIAN_TTS_FORWARD_CHUNK_SIZE", "8192"))

logger = logging.getLogger(__name__)


def iter_text_chunks(text: str, chunk_size: int = TTS_CHUNK_SIZE):
    for start in range(0, len(text), chunk_size):
        yield text[start : start + chunk_size]


def normalize_tts_text(text: str) -> str:
    normalized = " ".join(str(text or "").split())
    if TTS_MAX_TEXT_CHARS > 0:
        return normalized[:TTS_MAX_TEXT_CHARS]
    return normalized


def build_tts_config(sample_rate: int = TTS_SAMPLE_RATE, voice: str = TTS_VOICE) -> dict:
    return {
        "config": {
            "modalities": ["text", "audio"],
            "voice": voice,
            "instructions": TTS_INSTRUCTIONS,
            "input_audio_format": "pcm16",
            "output_audio_format": "pcm16",
            "sample_rate": sample_rate,
            "temperature": TTS_TEMPERATURE,
            "max_tokens": TTS_MAX_TOKENS,
        }
    }


async def send_tts_text(websocket, text: str, chunk_size: int, emotion: str, reqid: str):
    for chunk in iter_text_chunks(text, chunk_size):
        await websocket.send_str(
            json.dumps(
                {
                    "type": "input_text.append",
                    "text": chunk,
                    "emotion": emotion,
                    "reqid": reqid,
                },
                ensure_ascii=False,
            )
        )
        await asyncio.sleep(0.05)

    await websocket.send_str(json.dumps({"type": "input_text.commit", "reqid": reqid}))


async def receive_tts_pcm(websocket) -> bytes:
    pcm = bytearray()

    while True:
        msg = await websocket.receive()

        if msg.type == aiohttp.WSMsgType.BINARY:
            pcm.extend(msg.data)
            continue

        if msg.type == aiohttp.WSMsgType.TEXT:
            try:
                message = json.loads(msg.data)
            except json.JSONDecodeError:
                logger.debug("[tts] skipped non-JSON text event")
                continue

            event_type = message.get("type")
            if event_type == "response.audio.delta":
                audio_b64 = message.get("delta", "")
                if audio_b64:
                    pcm.extend(base64.b64decode(audio_b64))
            elif event_type == "response.done":
                return bytes(pcm)
            elif event_type == "error":
                raise RuntimeError(f"TTS server error: {message}")
            else:
                logger.debug("[tts] event: %s", event_type)
            continue

        if msg.type in {
            aiohttp.WSMsgType.CLOSE,
            aiohttp.WSMsgType.CLOSED,
            aiohttp.WSMsgType.CLOSING,
        }:
            return bytes(pcm)

        if msg.type == aiohttp.WSMsgType.ERROR:
            raise websocket.exception() or RuntimeError("TTS upstream websocket error")


async def forward_tts_stream(
    websocket,
    client_ws,
    idle_timeout: float = TTS_STREAM_IDLE_TIMEOUT,
) -> int:
    total_audio_bytes = 0

    while True:
        try:
            msg = await websocket.receive(timeout=idle_timeout)
        except asyncio.TimeoutError as err:
            raise TimeoutError(f"TTS stream idle timeout after {idle_timeout}s") from err

        if msg.type == aiohttp.WSMsgType.BINARY:
            await client_ws.send_bytes(msg.data)
            total_audio_bytes += len(msg.data)
            continue

        if msg.type == aiohttp.WSMsgType.TEXT:
            try:
                message = json.loads(msg.data)
            except json.JSONDecodeError:
                logger.debug("[tts] skipped non-JSON text event")
                continue

            event_type = message.get("type")
            if event_type == "response.audio.delta":
                audio_b64 = message.get("delta", "")
                if audio_b64:
                    audio_bytes = base64.b64decode(audio_b64)
                    await client_ws.send_bytes(audio_bytes)
                    total_audio_bytes += len(audio_bytes)
            elif event_type == "response.done":
                return total_audio_bytes
            elif event_type == "error":
                raise RuntimeError(f"TTS server error: {message}")
            else:
                logger.debug("[tts] event: %s", event_type)
            continue

        if msg.type in {
            aiohttp.WSMsgType.CLOSE,
            aiohttp.WSMsgType.CLOSED,
            aiohttp.WSMsgType.CLOSING,
        }:
            return total_audio_bytes

        if msg.type == aiohttp.WSMsgType.ERROR:
            raise websocket.exception() or RuntimeError("TTS upstream websocket error")


def pcm16_to_wav_bytes(pcm: bytes, sample_rate: int = TTS_SAMPLE_RATE) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm)
    return buffer.getvalue()


async def synthesize_tts_pcm(
    text: str,
    *,
    url: str = TTS_URL,
    sample_rate: int = TTS_SAMPLE_RATE,
    voice: str = TTS_VOICE,
    open_timeout: float = TTS_OPEN_TIMEOUT,
    timeout: float = TTS_TIMEOUT,
    chunk_size: int = TTS_CHUNK_SIZE,
    emotion: str = TTS_EMOTION,
    reqid: str | None = None,
) -> bytes:
    text = normalize_tts_text(text)
    if not text:
        raise ValueError("text must not be empty")

    reqid = reqid or uuid.uuid4().hex
    logger.info("[tts] synthesize reqid=%s chars=%s url=%s", reqid, len(text), url)

    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=None),
        trust_env=TTS_TRUST_ENV,
    ) as session:
        websocket = await session.ws_connect(
            url,
            timeout=open_timeout,
            heartbeat=20,
            max_msg_size=0,
            proxy=TTS_PROXY,
        )
        try:
            await websocket.send_str(json.dumps(build_tts_config(sample_rate, voice)))
            await send_tts_text(websocket, text, chunk_size, emotion, reqid)
            pcm = await asyncio.wait_for(receive_tts_pcm(websocket), timeout=timeout)
        finally:
            if not websocket.closed:
                await websocket.close()

    if not pcm:
        raise RuntimeError("TTS returned no audio")
    logger.info("[tts] synthesized reqid=%s audio_bytes=%s", reqid, len(pcm))
    return pcm


async def synthesize_tts_wav(text: str, **kwargs) -> tuple[bytes, int]:
    sample_rate = int(kwargs.pop("sample_rate", None) or TTS_SAMPLE_RATE)
    pcm = await synthesize_tts_pcm(text, sample_rate=sample_rate, **kwargs)
    return pcm16_to_wav_bytes(pcm, sample_rate), len(pcm)


async def run_tts_stream_request(client_ws, data, provider=None):
    if provider == BAILIAN_PROVIDER:
        await run_bailian_tts_stream_request(client_ws, data)
        return

    upstream_session = None
    upstream_ws = None
    reqid = data.get("request_id") or data.get("reqid")

    try:
        text = normalize_tts_text(data.get("text", ""))
        if not text:
            await client_ws.send_json(
                {"type": "error", "request_id": reqid, "error": "Missing text"}
            )
            return

        try:
            sample_rate = int(data.get("sample_rate") or TTS_SAMPLE_RATE)
        except (TypeError, ValueError):
            await client_ws.send_json(
                {"type": "error", "request_id": reqid, "error": "Invalid sample_rate"}
            )
            return

        url = data.get("url") or TTS_URL
        voice = data.get("voice") or TTS_VOICE
        emotion = data.get("emotion") or TTS_EMOTION
        reqid = reqid or f"{data.get('session_id') or 'web'}-{uuid.uuid4().hex[:12]}"

        await client_ws.send_json(
            {
                "type": "start",
                "request_id": reqid,
                "format": "pcm16",
                "sample_rate": sample_rate,
                "channels": 1,
                "reqid": reqid,
            }
        )

        logger.info("[tts] stream reqid=%s chars=%s url=%s", reqid, len(text), url)
        upstream_session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=None),
            trust_env=TTS_TRUST_ENV,
        )
        upstream_ws = await upstream_session.ws_connect(
            url,
            timeout=TTS_OPEN_TIMEOUT,
            heartbeat=20,
            max_msg_size=0,
            proxy=TTS_PROXY,
        )
        await upstream_ws.send_str(json.dumps(build_tts_config(sample_rate, voice)))
        await send_tts_text(upstream_ws, text, TTS_CHUNK_SIZE, emotion, reqid)
        total_audio_bytes = await forward_tts_stream(upstream_ws, client_ws)
        await client_ws.send_json(
            {
                "type": "done",
                "request_id": reqid,
                "reqid": reqid,
                "audio_bytes": total_audio_bytes,
            }
        )
        logger.info("[tts] stream done reqid=%s audio_bytes=%s", reqid, total_audio_bytes)
    except asyncio.CancelledError:
        logger.info("[tts] stream cancelled reqid=%s", reqid)
        if not client_ws.closed:
            try:
                await client_ws.send_json({"type": "stopped", "request_id": reqid, "reqid": reqid})
            except (ConnectionResetError, RuntimeError):
                pass
        raise
    except asyncio.TimeoutError:
        logger.warning("[tts] stream timeout reqid=%s", reqid)
        if not client_ws.closed:
            await client_ws.send_json(
                {"type": "error", "request_id": reqid, "error": "TTS request timed out"}
            )
    except (aiohttp.ClientError, OSError, RuntimeError) as err:
        logger.warning("[tts] stream failed reqid=%s: %s", reqid, err)
        if not client_ws.closed:
            await client_ws.send_json(
                {"type": "error", "request_id": reqid, "error": f"TTS failed: {err}"}
            )
    finally:
        if upstream_ws is not None and not upstream_ws.closed:
            await upstream_ws.close()
        if upstream_session is not None:
            await upstream_session.close()


async def cancel_tts_stream_task(task):
    if task is None or task.done():
        return
    task.cancel()
    try:
        await asyncio.wait_for(task, timeout=TTS_CANCEL_TIMEOUT)
    except asyncio.CancelledError:
        pass
    except asyncio.TimeoutError:
        logger.warning("[tts] previous stream did not stop within %.2fs", TTS_CANCEL_TIMEOUT)


async def tts_websocket_handler(request):
    client_ws = web.WebSocketResponse(heartbeat=20, max_msg_size=0)
    await client_ws.prepare(request)

    stream_task = None

    try:
        async for msg in client_ws:
            if msg.type == web.WSMsgType.TEXT:
                try:
                    data = json.loads(msg.data)
                except json.JSONDecodeError:
                    await client_ws.send_json({"type": "error", "error": "Invalid JSON"})
                    continue

                message_type = data.get("type") or "speak"
                if message_type == "speak":
                    await cancel_tts_stream_task(stream_task)
                    stream_task = asyncio.create_task(
                        run_tts_stream_request(
                            client_ws, data, request.app.get("voice_provider")
                        )
                    )
                elif message_type == "stop":
                    await cancel_tts_stream_task(stream_task)
                    stream_task = None
                elif message_type == "ping":
                    await client_ws.send_json({"type": "pong", "id": data.get("id")})
                else:
                    await client_ws.send_json(
                        {"type": "error", "error": f"Unknown TTS message type: {message_type}"}
                    )
            elif msg.type == web.WSMsgType.ERROR:
                raise client_ws.exception() or RuntimeError("TTS client websocket error")
    except Exception as err:
        logger.warning("[tts] browser websocket failed: %s", err)
    finally:
        await cancel_tts_stream_task(stream_task)
        if not client_ws.closed:
            await client_ws.close()

    return client_ws


async def tts_config_handler(request):
    if request.app.get("voice_provider") == BAILIAN_PROVIDER:
        return web.json_response(
            {
                "provider": BAILIAN_PROVIDER,
                "model": BAILIAN_TTS_MODEL,
                "sample_rate": TTS_SAMPLE_RATE,
                "voice": BAILIAN_TTS_VOICE,
                "max_text_chars": TTS_MAX_TEXT_CHARS,
            }
        )
    return web.json_response(
        {
            "url": TTS_URL,
            "sample_rate": TTS_SAMPLE_RATE,
            "voice": TTS_VOICE,
            "emotion": TTS_EMOTION,
            "chunk_size": TTS_CHUNK_SIZE,
            "max_text_chars": TTS_MAX_TEXT_CHARS,
        }
    )


def setup_tts_routes(app):
    app.router.add_get("/api/tts/config", tts_config_handler)
    app.router.add_get("/api/tts", tts_websocket_handler)


def wav_to_pcm16(wav_data: bytes) -> tuple[bytes, int]:
    """Convert the cloud WAV result into the PCM16 protocol already used by the UI."""
    with wave.open(io.BytesIO(wav_data), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        if channels != 1 or sample_width != 2:
            raise ValueError("Bailian TTS returned an unsupported WAV format")
        return wav_file.readframes(wav_file.getnframes()), sample_rate


def build_bailian_tts_payload(text: str, voice: str = BAILIAN_TTS_VOICE) -> dict:
    return {
        "model": BAILIAN_TTS_MODEL,
        "input": {
            "text": text,
            "voice": voice,
            "format": "wav",
            "sample_rate": TTS_SAMPLE_RATE,
        },
    }


async def synthesize_bailian_tts_pcm(text: str, voice: str = BAILIAN_TTS_VOICE) -> tuple[bytes, int]:
    """Call Qwen TTS and return mono PCM16 suitable for the current browser player."""
    text = normalize_tts_text(text)
    if not text:
        raise ValueError("text must not be empty")

    timeout = aiohttp.ClientTimeout(total=BAILIAN_TTS_HTTP_TIMEOUT)
    headers = {
        "Authorization": f"Bearer {load_bailian_speech_api_key()}",
        "Content-Type": "application/json",
    }
    async with aiohttp.ClientSession(timeout=timeout, trust_env=TTS_TRUST_ENV) as session:
        async with session.post(BAILIAN_TTS_API_URL, headers=headers, json=build_bailian_tts_payload(text, voice)) as response:
            if response.status >= 400:
                raise RuntimeError("Bailian TTS request was rejected")
            payload = await response.json(content_type=None)
        audio_url = ((payload.get("output") or {}).get("audio") or {}).get("url")
        if not isinstance(audio_url, str) or not audio_url:
            raise RuntimeError("Bailian TTS returned no audio URL")
        async with session.get(audio_url) as audio_response:
            if audio_response.status >= 400:
                raise RuntimeError("Bailian TTS audio download failed")
            wav_data = await audio_response.read()
    return wav_to_pcm16(wav_data)


async def run_bailian_tts_stream_request(client_ws, data):
    """Keep the existing browser WebSocket contract while using HTTP cloud TTS."""
    reqid = data.get("request_id") or data.get("reqid") or uuid.uuid4().hex
    try:
        text = normalize_tts_text(data.get("text", ""))
        if not text:
            await client_ws.send_json({"type": "error", "request_id": reqid, "error": "Missing text"})
            return
        voice = data.get("voice") or BAILIAN_TTS_VOICE
        pcm, sample_rate = await synthesize_bailian_tts_pcm(text, voice)
        await client_ws.send_json(
            {"type": "start", "request_id": reqid, "format": "pcm16", "sample_rate": sample_rate, "channels": 1, "reqid": reqid}
        )
        for offset in range(0, len(pcm), BAILIAN_TTS_FORWARD_CHUNK_SIZE):
            await client_ws.send_bytes(pcm[offset : offset + BAILIAN_TTS_FORWARD_CHUNK_SIZE])
        await client_ws.send_json({"type": "done", "request_id": reqid, "reqid": reqid, "audio_bytes": len(pcm)})
    except asyncio.CancelledError:
        raise
    except ValueError as err:
        await client_ws.send_json({"type": "error", "request_id": reqid, "error": str(err)})
    except Exception:
        logger.warning("[tts] Bailian TTS request failed reqid=%s", reqid)
        await client_ws.send_json(
            {"type": "error", "request_id": reqid, "error": "Bailian TTS request failed. Check the service configuration."}
        )
