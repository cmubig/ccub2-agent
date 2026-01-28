"""Wikimedia Commons downloader for curated cultural images.

Uses the MediaWiki API (no API key required) to search and download
freely-licensed images from Wikimedia Commons.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any
from urllib.parse import unquote

import requests

from ccub2_agent.data.curation.base_downloader import BaseDownloader, DownloadResult
from ccub2_agent.schemas.provenance_schema import CurationPlatform, License

logger = logging.getLogger(__name__)

# Wikimedia API endpoints
COMMONS_API = "https://commons.wikimedia.org/w/api.php"

# Map Wikimedia license templates to our License enum
LICENSE_MAP: dict[str, License] = {
    "cc0": License.CC0_1_0,
    "cc-zero": License.CC0_1_0,
    "cc-by-4.0": License.CC_BY_4_0,
    "cc-by-sa-4.0": License.CC_BY_SA_4_0,
    "pd": License.PUBLIC_DOMAIN,
    "public domain": License.PUBLIC_DOMAIN,
    "pd-old": License.PUBLIC_DOMAIN,
    "pd-us": License.PUBLIC_DOMAIN,
    "pd-self": License.PUBLIC_DOMAIN,
}


class WikimediaDownloader(BaseDownloader):
    """Download cultural images from Wikimedia Commons."""

    platform = CurationPlatform.WIKIMEDIA_COMMONS

    def __init__(
        self,
        output_dir: Path,
        country: str,
        category: str | None = None,
        limit: int = 50,
        delay: float = 1.0,
    ):
        super().__init__(output_dir, country, category, limit)
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "WorldCCUB-Agent/1.0 (https://github.com/cmubig/ccub2-agent; research@worldccub.org)"
        })

    def search(self, query: str, limit: int = 50) -> list[dict[str, Any]]:
        """Search Wikimedia Commons for images matching the query."""
        params = {
            "action": "query",
            "format": "json",
            "generator": "search",
            "gsrsearch": f"filetype:bitmap {query}",
            "gsrnamespace": "6",  # File namespace
            "gsrlimit": str(min(limit, 50)),
            "prop": "imageinfo",
            "iiprop": "url|extmetadata|size|mime",
            "iiurlwidth": "1024",
        }

        try:
            resp = self.session.get(COMMONS_API, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as e:
            logger.error(f"Wikimedia API error: {e}")
            return []

        pages = data.get("query", {}).get("pages", {})
        results: list[dict[str, Any]] = []

        for page_id, page in pages.items():
            if int(page_id) < 0:
                continue

            imageinfo = page.get("imageinfo", [{}])[0]
            extmeta = imageinfo.get("extmetadata", {})

            # Extract license
            license_str = extmeta.get("LicenseShortName", {}).get("value", "").lower()
            mapped_license = self._map_license(license_str)
            if mapped_license is None:
                continue

            # Extract metadata
            title = page.get("title", "")
            author = extmeta.get("Artist", {}).get("value", "Unknown")
            # Strip HTML tags from author
            if "<" in author:
                import re
                author = re.sub(r"<[^>]+>", "", author).strip()

            url = imageinfo.get("thumburl") or imageinfo.get("url", "")
            if not url:
                continue

            # Filter by minimum size
            width = imageinfo.get("width", 0)
            height = imageinfo.get("height", 0)
            if width < 256 or height < 256:
                continue

            results.append({
                "url": url,
                "source_id": title,
                "license": mapped_license.value,
                "attribution": f"Photo by {author}, Wikimedia Commons",
                "title": unquote(title.replace("File:", "")),
                "width": width,
                "height": height,
                "ext": self._get_extension(imageinfo.get("mime", "image/jpeg")),
                "description": extmeta.get("ImageDescription", {}).get("value", ""),
            })

        time.sleep(self.delay)
        return results

    def download_image(self, url: str, dest: Path) -> bool:
        """Download a single image from Wikimedia Commons."""
        try:
            resp = self.session.get(url, timeout=60, stream=True)
            resp.raise_for_status()

            with open(dest, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)

            time.sleep(self.delay)
            return True
        except requests.RequestException as e:
            logger.error(f"Failed to download {url}: {e}")
            return False

    def _map_license(self, license_str: str) -> License | None:
        """Map a Wikimedia license string to our License enum."""
        license_lower = license_str.lower().strip()
        for key, value in LICENSE_MAP.items():
            if key in license_lower:
                return value
        return None

    @staticmethod
    def _get_extension(mime: str) -> str:
        """Get file extension from MIME type."""
        mime_map = {
            "image/jpeg": ".jpg",
            "image/png": ".png",
            "image/gif": ".gif",
            "image/webp": ".webp",
            "image/tiff": ".tiff",
        }
        return mime_map.get(mime, ".jpg")
