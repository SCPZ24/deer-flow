import base64
from abc import ABC, abstractmethod

import requests
from pydantic import BaseModel, Field


class ImageModelConfig(BaseModel):
    """
    Config for image generation models.
    """
    name: str = Field(default=None, description="Unique identifier for the image model")
    model: str = Field(default=None, description="Name of the image model")
    api_base: str = Field(default=None, description="Base URL for the image model API")
    api_key: str = Field(default=None, description="API key for authenticating with the image model API")


class BaseImageGenerator(ABC):
    def __init__(self, cfg: ImageModelConfig):
        self.cfg = cfg

    @abstractmethod
    def _generate(
        self,
        prompt: str,
        reference_images: list[dict],
        aspect_ratio: str = "16:9",
    ) -> bytes:
        raise NotImplementedError


class GeminiImageGenerator(BaseImageGenerator):
    def __init__(self, cfg: ImageModelConfig):
        super().__init__(cfg)

    def _generate(
        self,
        prompt: str,
        reference_images: list[dict],
        aspect_ratio: str = "16:9",
    ) -> bytes:
        api_key = self.cfg.api_key
        if not api_key:
            raise ValueError("api_key required")

        model = self.cfg.model or "gemini-3-flash-image" 
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    
        headers = {"Content-Type": "application/json"}
        params = {"key": api_key}

        payload = {
            "contents": [{
                "parts": reference_images + [{"text": prompt}]
            }],
            "generationConfig": {
                "imageConfig": {
                    "aspectRatio": aspect_ratio,
                    "outputMimeType": "image/jpeg"
                }
         }
        }

        resp = requests.post(url, headers=headers, params=params, json=payload, timeout=90)
        resp.raise_for_status()
    
        data = resp.json()
    
        candidate = data.get("candidates", [{}])[0]
        content_parts = candidate.get("content", {}).get("parts", [])
    
        for part in content_parts:
            if "inlineData" in part:
                return base64.b64decode(part["inlineData"]["data"])
            
        # If no inlineData found, raise error
        reason = candidate.get("finishReason", "UNKNOWN")
        raise RuntimeError(f"No image generated. Finish Reason: {reason}")


class VolcengineSeedreamImageGenerator(BaseImageGenerator):
    def __init__(self, cfg: ImageModelConfig):
        super().__init__(cfg)
        self.cfg = cfg

    def _generate(
        self,
        prompt: str,
        reference_images: list[dict],
        aspect_ratio: str = "16:9",
    ) -> bytes:
        if not self.cfg.api_base:
            raise ValueError("api_base required")
        if not self.cfg.api_key:
            raise ValueError("api_key required")

        url = f"{self.cfg.api_base.rstrip('/')}"
        headers = {"Authorization": f"Bearer {self.cfg.api_key}", "Content-Type": "application/json"}
    
        size_map = {"1:1": "1920x1920", "16:9": "2560x1440", "2:3": "1600x2400"}
        size = size_map.get(aspect_ratio, "2560x1440")
        
        payload = {
            "model": self.cfg.model,
            "prompt": prompt,
            "sequential_image_generation": "disabled",
            "response_format": "b64_json",
            "size": size,
            "stream": False,
            "watermark": False
        }

        resp = requests.post(url, headers=headers, json=payload, timeout=120)
        try:
            resp.raise_for_status()
        except Exception as e:
            error_msg = f"Request failed: {e}\nURL: {url}\nStatus: {resp.status_code}\nResponse: {resp.text}"
            raise RuntimeError(error_msg)
        data = resp.json()

        images = data.get("data") or []
        if not images:
            raise RuntimeError("No image in response")
        
        b64img = images[0].get("b64_json")
        if b64img:
            return base64.b64decode(b64img)
        
        img_url = images[0].get("url")
        if img_url:
            img_resp = requests.get(img_url, timeout=30)
            img_resp.raise_for_status()
            return img_resp.content
        
        raise RuntimeError(f"No image data found in response: {data}")

def get_image_generate_fn(cfg: ImageModelConfig):
    name = cfg.name.lower()
    if "gemini" in name:
        return GeminiImageGenerator(cfg)._generate
    if "seedream" in name:
        return VolcengineSeedreamImageGenerator(cfg)._generate
    return None
