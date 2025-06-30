import os
import time
import asyncio
import requests
import tempfile
import numpy as np
from typing import Dict, Any, Optional, List, Callable, Awaitable
from dotenv import load_dotenv
from loguru import logger

# Import Murf client
from murf import Murf

# Import OpenAI for OpenAI TTS
import openai

# Import pipecat base classes
from pipecat.services.tts_service import TTSService
from pipecat.frames.frames import TTSTextFrame
from pyee.base import EventEmitter

load_dotenv()

# Test function to demonstrate both TTS services
async def test_tts_services():
    # Test Murf TTS
    murf_service = MurfTTSService(
        api_key=os.getenv("MURF_API_KEY"),
        voice_id="en-UK-ruby",
        style="Narration",
        multi_native_locale="id-ID"
    )
    
    murf_audio = await murf_service.synthesize("Selamat Datang di Indonesia dengan Murf TTS!")
    print(f"Murf audio length: {len(murf_audio) if murf_audio else 0} bytes")
    
    # Test OpenAI TTS
    openai_service = OpenAITTSService(
        api_key=os.getenv("OPENAI_API_KEY"),
        voice="shimmer",
        model="tts-1"
    )
    
    openai_audio = await openai_service.synthesize("Selamat Datang di Indonesia dengan OpenAI TTS!")
    print(f"OpenAI audio length: {len(openai_audio) if openai_audio else 0} bytes")


class MurfTTSService(TTSService, EventEmitter):
    """Murf TTS Service for text-to-speech synthesis."""

    def __init__(
        self,
        api_key: str,
        voice_id: str = "en-UK-ruby",
        style: str = "Narration",
        multi_native_locale: str = "id-ID",
    ):
        """Initialize the Murf TTS service.

        Args:
            api_key: Murf API key
            voice_id: Voice ID to use
            style: Voice style to use
            multi_native_locale: Locale for multilingual support
        """
        BaseTTSService.__init__(self)
        EventEmitter.__init__(self)
        
        self.api_key = api_key
        self.voice_id = voice_id
        self.style = style
        self.multi_native_locale = multi_native_locale
        self.client = Murf(api_key=self.api_key)
        
        logger.info(f"Initialized Murf TTS service with voice {voice_id}")

    async def synthesize(self, text: str) -> Optional[bytes]:
        """Synthesize text to speech using Murf API.

        Args:
            text: Text to synthesize

        Returns:
            Audio data as bytes or None if synthesis failed
        """
        try:
            # Emit TTS start event
            await self.emit("on_tts_start", self, text)
            start_time = time.time()
            
            # Call Murf API
            response = self.client.text_to_speech.generate(
                text=text,
                voice_id=self.voice_id,
                style=self.style,
                multi_native_locale=self.multi_native_locale
            )
            
            # Download the audio file
            audio_response = requests.get(response.audio_file)
            audio_data = audio_response.content
            
            # Calculate latency
            latency = time.time() - start_time
            logger.info(f"Murf TTS synthesis completed in {latency:.2f}s")
            
            # Emit TTS end event
            await self.emit("on_tts_end", self, audio_data)
            
            return audio_data
        except Exception as e:
            logger.error(f"Murf TTS synthesis failed: {e}")
            await self.emit("on_tts_error", self, str(e))
            return None

    async def process_frame(self, frame: TTSTextFrame) -> Optional[bytes]:
        """Process a TTSTextFrame and return audio data.

        Args:
            frame: TTSTextFrame containing text to synthesize

        Returns:
            Audio data as bytes or None if synthesis failed
        """
        return await self.synthesize(frame.text)


class OpenAITTSService(TTSService, EventEmitter):
    """OpenAI TTS Service for text-to-speech synthesis."""

    def __init__(
        self,
        api_key: str,
        voice: str = "shimmer",
        model: str = "tts-1",
        speed: float = 1.0,
    ):
        """Initialize the OpenAI TTS service.

        Args:
            api_key: OpenAI API key
            voice: Voice to use (alloy, echo, fable, onyx, nova, shimmer)
            model: Model to use (tts-1, tts-1-hd)
            speed: Speech speed (0.25 to 4.0)
        """
        BaseTTSService.__init__(self)
        EventEmitter.__init__(self)
        
        self.api_key = api_key
        self.voice = voice
        self.model = model
        self.speed = speed
        self.client = openai.OpenAI(api_key=self.api_key)
        
        logger.info(f"Initialized OpenAI TTS service with voice {voice} and model {model}")

    async def synthesize(self, text: str) -> Optional[bytes]:
        """Synthesize text to speech using OpenAI API.

        Args:
            text: Text to synthesize

        Returns:
            Audio data as bytes or None if synthesis failed
        """
        try:
            # Emit TTS start event
            await self.emit("on_tts_start", self, text)
            start_time = time.time()
            
            # Call OpenAI API
            response = self.client.audio.speech.create(
                model=self.model,
                voice=self.voice,
                input=text,
                speed=self.speed,
                response_format="wav"
            )
            
            # Get audio data
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_file:
                response.stream_to_file(temp_file.name)
                with open(temp_file.name, "rb") as f:
                    audio_data = f.read()
            
            # Calculate latency
            latency = time.time() - start_time
            logger.info(f"OpenAI TTS synthesis completed in {latency:.2f}s")
            
            # Emit TTS end event
            await self.emit("on_tts_end", self, audio_data)
            
            return audio_data
        except Exception as e:
            logger.error(f"OpenAI TTS synthesis failed: {e}")
            await self.emit("on_tts_error", self, str(e))
            return None

    async def process_frame(self, frame: TTSTextFrame) -> Optional[bytes]:
        """Process a TTSTextFrame and return audio data.

        Args:
            frame: TTSTextFrame containing text to synthesize

        Returns:
            Audio data as bytes or None if synthesis failed
        """
        return await self.synthesize(frame.text)


# Run test if script is executed directly
if __name__ == "__main__":
    asyncio.run(test_tts_services())
