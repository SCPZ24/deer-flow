from pydantic import BaseModel, Field
import base64
import os
import requests
from abc import ABC, abstractmethod

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
        api_key = self.cfg.api_key or os.getenv("GEMINI_API_KEY") or ""
        if not api_key:
            raise ValueError("Missing GEMINI_API_KEY")
        model = self.cfg.model or "gemini-3-pro-image-preview"
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        headers = {"x-goog-api-key": api_key, "Content-Type": "application/json"}
        payload = {
            "generationConfig": {"imageConfig": {"aspectRatio": aspect_ratio}},
            "contents": [{"parts": [*(reference_images or []), {"text": prompt}]}],
        }
        resp = requests.post(url, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        cand = data.get("candidates") or []
        if not cand:
            raise RuntimeError("No candidates")
        parts_out = cand[0].get("content", {}).get("parts") or []
        imgs = [p for p in parts_out if isinstance(p, dict) and p.get("inlineData")]
        if not imgs:
            raise RuntimeError("No image in response")
        b64img = imgs[0]["inlineData"]["data"]
        return base64.b64decode(b64img)


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
            # 输出详细错误信息帮助调试
            error_msg = f"Request failed: {e}\nURL: {url}\nStatus: {resp.status_code}\nResponse: {resp.text}"
            raise RuntimeError(error_msg)
        data = resp.json()

        # 响应格式: {"created": 1234567890, "data": [{"b64_json": "base64..."}]} 
        # 成功curl返回url格式，我们这里用b64_json更方便
        images = data.get("data") or []
        if not images:
            raise RuntimeError("No image in response")
        
        # 尝试获取b64_json
        b64img = images[0].get("b64_json")
        if b64img:
            return base64.b64decode(b64img)
        
        # 如果是url格式，下载图片
        img_url = images[0].get("url")
        if img_url:
            img_resp = requests.get(img_url, timeout=30)
            img_resp.raise_for_status()
            return img_resp.content
        
        raise RuntimeError(f"No image data found in response: {data}")

class OpenAICompatibleImageGenerator(BaseImageGenerator):
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
        api_key = self.cfg.api_key or os.getenv("OPENAI_API_KEY") or ""
        if not api_key:
            raise ValueError("api_key required")
        url = f"{self.cfg.api_base.rstrip('/')}"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {
            "model": self.cfg.model or "gpt-image-1",
            "prompt": prompt,
            "size": "1024x576" if aspect_ratio == "16:9" else "1024x1024",
        }
        resp = requests.post(url, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        images = data.get("data") or []
        if not images:
            raise RuntimeError("No image in response")
        b64img = images[0].get("b64_json")
        if not b64img:
            raise RuntimeError("No base64 data")
        return base64.b64decode(b64img)


def get_image_generate_fn(cfg: ImageModelConfig):
    name = cfg.name.lower()
    if "gemini" in name:
        return GeminiImageGenerator(cfg)._generate
    if "seedream" in name:
        return VolcengineSeedreamImageGenerator(cfg)._generate
    if "openai" in name:
        return OpenAICompatibleImageGenerator(cfg)._generate
    return None
