"""Abstract base class for cultural image downloaders.

Each platform-specific downloader inherits from this class and implements
the search and download methods for its respective API.
"""

from __future__ import annotations

import hashlib
import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from ccub2_agent.schemas.provenance_schema import (
    CuratedProvenance,
    CurationPlatform,
    License,
)

logger = logging.getLogger(__name__)


# Cultural categories and their search terms per country
CULTURAL_SEARCH_TERMS: dict[str, dict[str, list[str]]] = {
    "korea": {
        "food": ["Korean cuisine", "kimchi", "bibimbap", "Korean BBQ", "tteok"],
        "clothing": ["hanbok", "Korean traditional dress"],
        "architecture": ["hanok", "Korean temple", "Korean palace"],
        "city": ["Seoul skyline", "Busan", "Korean street market"],
        "nature": ["Korean landscape", "Jeju Island", "Seoraksan"],
        "religion": ["Chuseok", "Seollal", "Korean Buddhist ceremony"],
        "art": ["Korean pottery", "Korean calligraphy", "celadon"],
        "people": ["Korean daily life", "Korean street fashion", "Korean elderly"],
    },
    "china": {
        "food": ["Chinese cuisine", "dim sum", "Peking duck", "Chinese tea ceremony"],
        "clothing": ["hanfu", "cheongsam", "qipao", "Chinese traditional dress"],
        "architecture": ["Chinese temple", "pagoda", "hutong", "forbidden city"],
        "city": ["Shanghai skyline", "Beijing", "Chinese street"],
        "nature": ["Guilin karst", "Great Wall", "Chinese garden"],
        "religion": ["Chinese New Year", "Mid-Autumn Festival", "Dragon Boat"],
        "art": ["Chinese calligraphy", "Chinese painting", "porcelain"],
        "people": ["Chinese daily life", "Chinese tea culture", "Chinese market vendor"],
    },
    "japan": {
        "food": ["Japanese cuisine", "sushi", "ramen", "tempura", "wagashi"],
        "clothing": ["kimono", "yukata", "Japanese traditional dress"],
        "architecture": ["Japanese temple", "torii gate", "machiya"],
        "city": ["Tokyo skyline", "Kyoto street", "Osaka"],
        "nature": ["Mount Fuji", "Japanese garden", "bamboo forest"],
        "religion": ["Matsuri", "Hanami", "Shinto shrine ceremony"],
        "art": ["ukiyo-e", "Japanese pottery", "ikebana"],
        "people": ["Japanese daily life", "Japanese commuter", "Japanese craftsman"],
    },
    "india": {
        "food": ["Indian cuisine", "biryani", "dosa", "chai", "thali"],
        "clothing": ["sari", "kurta", "Indian traditional dress", "lehenga"],
        "architecture": ["Indian temple", "Mughal architecture", "stepwell"],
        "city": ["Mumbai skyline", "Delhi street", "Indian bazaar"],
        "nature": ["Himalaya landscape", "Indian countryside", "Kerala backwaters"],
        "religion": ["Diwali", "Holi", "Indian wedding ceremony"],
        "art": ["Rangoli", "Indian miniature painting", "Madhubani art"],
        "people": ["Indian daily life", "Indian farmer", "Indian street vendor"],
    },
    "indonesia": {
        "food": ["Indonesian cuisine", "nasi goreng", "rendang", "satay"],
        "clothing": ["batik", "kebaya", "Indonesian traditional dress"],
        "architecture": ["Borobudur", "Balinese temple", "Indonesian mosque"],
        "city": ["Jakarta skyline", "Indonesian market", "Yogyakarta"],
        "nature": ["Bali rice terrace", "Komodo Island", "Indonesian rainforest"],
        "religion": ["Nyepi", "Waisak", "Indonesian ceremony"],
        "art": ["Wayang puppet", "Indonesian batik art", "Balinese dance"],
        "people": ["Indonesian daily life", "Balinese ceremony participant", "Indonesian fisherman"],
    },
    "vietnam": {
        "food": ["Vietnamese cuisine", "pho", "banh mi", "spring rolls"],
        "clothing": ["ao dai", "Vietnamese traditional dress"],
        "architecture": ["Vietnamese pagoda", "Hoi An ancient town", "Imperial City Hue"],
        "city": ["Hanoi street", "Ho Chi Minh City", "Vietnamese floating market"],
        "nature": ["Ha Long Bay", "Sapa rice terrace", "Mekong Delta"],
        "religion": ["Tet festival", "Vietnamese Buddhist ceremony", "Vietnamese ancestor worship"],
        "art": ["Vietnamese lacquer art", "Vietnamese silk painting", "water puppetry"],
        "people": ["Vietnamese daily life", "Vietnamese street vendor", "Vietnamese farmer"],
    },
    "saudi_arabia": {
        "food": ["Saudi cuisine", "kabsa", "Arabic coffee", "dates"],
        "clothing": ["thobe", "abaya", "Saudi traditional dress"],
        "architecture": ["Masjid al-Haram", "Diriyah", "Saudi Arabian fort"],
        "city": ["Riyadh skyline", "Jeddah", "Saudi Arabian market"],
        "nature": ["Saudi desert", "Red Sea coral", "Al Ula landscape"],
        "religion": ["Hajj pilgrimage", "Eid celebration", "Saudi mosque"],
        "art": ["Arabic calligraphy", "Saudi pottery", "Islamic geometric art"],
        "people": ["Saudi daily life", "Bedouin culture", "Saudi Arabian market"],
    },
    "turkey": {
        "food": ["Turkish cuisine", "kebab", "baklava", "Turkish tea"],
        "clothing": ["Turkish traditional dress", "Ottoman clothing"],
        "architecture": ["Hagia Sophia", "Blue Mosque", "Cappadocia"],
        "city": ["Istanbul skyline", "Grand Bazaar", "Turkish street"],
        "nature": ["Pamukkale", "Cappadocia landscape", "Turkish coast"],
        "religion": ["Whirling dervish", "Turkish mosque ceremony", "Ramadan in Turkey"],
        "art": ["Turkish carpet", "Iznik tile", "Turkish ceramics"],
        "people": ["Turkish daily life", "Turkish tea culture", "Turkish bazaar vendor"],
    },
    "nigeria": {
        "food": ["Nigerian cuisine", "jollof rice", "pounded yam", "suya"],
        "clothing": ["agbada", "aso oke", "gele", "Nigerian fashion"],
        "architecture": ["Nigerian architecture", "Benin City walls"],
        "city": ["Lagos skyline", "Abuja", "Nigerian market"],
        "nature": ["Niger Delta", "Nigerian savanna", "Yankari"],
        "religion": ["Eyo festival", "Durbar festival", "Nigerian ceremony"],
        "art": ["Nok sculpture", "Benin bronze", "Nigerian art"],
        "people": ["Nigerian daily life", "Lagos street scene", "Nigerian market woman"],
    },
    "kenya": {
        "food": ["Kenyan cuisine", "ugali", "nyama choma", "Kenyan tea"],
        "clothing": ["Maasai shuka", "Kenyan traditional dress", "kikoy"],
        "architecture": ["Kenyan architecture", "Fort Jesus Mombasa", "Lamu old town"],
        "city": ["Nairobi skyline", "Kenyan market", "Mombasa street"],
        "nature": ["Masai Mara", "Mount Kenya", "Great Rift Valley"],
        "religion": ["Kenyan ceremony", "Maasai ritual", "Kenyan church"],
        "art": ["Kenyan beadwork", "Maasai art", "soapstone carving"],
        "people": ["Kenyan daily life", "Maasai people", "Kenyan farmer"],
    },
    "ethiopia": {
        "food": ["Ethiopian cuisine", "injera", "doro wot", "Ethiopian coffee ceremony"],
        "clothing": ["Ethiopian traditional dress", "habesha kemis", "netela"],
        "architecture": ["Lalibela rock church", "Ethiopian monastery", "Axum obelisk"],
        "city": ["Addis Ababa", "Ethiopian market", "Ethiopian street"],
        "nature": ["Simien Mountains", "Danakil Depression", "Blue Nile Falls"],
        "religion": ["Timkat festival", "Ethiopian Orthodox ceremony", "Meskel"],
        "art": ["Ethiopian painting", "Ethiopian cross", "Ethiopian basket"],
        "people": ["Ethiopian daily life", "Ethiopian coffee ceremony", "Ethiopian farmer"],
    },
    "south_africa": {
        "food": ["South African cuisine", "braai", "bobotie", "biltong"],
        "clothing": ["Zulu traditional dress", "Xhosa beadwork dress", "Ndebele dress"],
        "architecture": ["Cape Dutch architecture", "Ndebele house", "South African township"],
        "city": ["Cape Town skyline", "Johannesburg", "South African market"],
        "nature": ["Table Mountain", "Kruger National Park", "Drakensberg"],
        "religion": ["Zulu ceremony", "South African church", "Reed Dance"],
        "art": ["Ndebele mural", "Zulu beadwork", "South African wire art"],
        "people": ["South African daily life", "township life", "South African street"],
    },
    "egypt": {
        "food": ["Egyptian cuisine", "koshari", "ful medames", "Egyptian bread"],
        "clothing": ["Egyptian traditional dress", "galabeya", "hijab Egypt"],
        "architecture": ["Egyptian pyramid", "Egyptian mosque", "Cairo Islamic architecture"],
        "city": ["Cairo skyline", "Egyptian bazaar", "Alexandria"],
        "nature": ["Nile River", "Egyptian desert", "Sinai Peninsula"],
        "religion": ["Egyptian mosque ceremony", "Coptic church", "Ramadan Egypt"],
        "art": ["Egyptian calligraphy", "pharaonic art", "Egyptian pottery"],
        "people": ["Egyptian daily life", "Cairo street scene", "Egyptian fisherman"],
    },
    "germany": {
        "food": ["German cuisine", "bratwurst", "pretzel", "beer garden"],
        "clothing": ["dirndl", "lederhosen", "German traditional dress"],
        "architecture": ["German castle", "half-timbered house", "German cathedral"],
        "city": ["Berlin skyline", "Munich", "German Christmas market"],
        "nature": ["Black Forest", "Bavarian Alps", "Rhine Valley"],
        "religion": ["Oktoberfest", "German Christmas tradition", "German carnival"],
        "art": ["German expressionism", "Bauhaus design", "German woodcarving"],
        "people": ["German daily life", "beer garden culture", "German market"],
    },
    "poland": {
        "food": ["Polish cuisine", "pierogi", "bigos", "Polish bread"],
        "clothing": ["Polish folk dress", "Kraków costume", "Polish highland dress"],
        "architecture": ["Polish castle", "Kraków old town", "wooden church Poland"],
        "city": ["Warsaw skyline", "Kraków", "Polish market square"],
        "nature": ["Tatra Mountains", "Białowieża Forest", "Mazury lakes"],
        "religion": ["Polish Easter", "Corpus Christi Poland", "Polish church ceremony"],
        "art": ["Wycinanki paper cut", "Polish pottery Bolesławiec", "amber craft"],
        "people": ["Polish daily life", "Polish countryside", "Polish market vendor"],
    },
    "brazil": {
        "food": ["Brazilian cuisine", "feijoada", "açaí", "churrasco"],
        "clothing": ["carnival costume Brazil", "baiana dress", "Brazilian fashion"],
        "architecture": ["Brazilian colonial architecture", "favela", "Oscar Niemeyer"],
        "city": ["Rio de Janeiro skyline", "São Paulo", "Brazilian street"],
        "nature": ["Amazon rainforest", "Iguazu Falls", "Brazilian coast"],
        "religion": ["Carnival Brazil", "Candomblé ceremony", "Brazilian Festa Junina"],
        "art": ["Brazilian street art", "capoeira", "samba dance"],
        "people": ["Brazilian daily life", "Rio beach culture", "Brazilian street vendor"],
    },
    "mexico": {
        "food": ["Mexican cuisine", "tacos", "mole", "tamales"],
        "clothing": ["Mexican traditional dress", "huipil", "charro suit"],
        "architecture": ["Aztec pyramid", "Mexican colonial church", "hacienda"],
        "city": ["Mexico City skyline", "Oaxaca", "Mexican market"],
        "nature": ["Copper Canyon", "cenote", "Mexican desert"],
        "religion": ["Dia de los Muertos", "Virgin of Guadalupe", "Mexican fiesta"],
        "art": ["Mexican mural art", "alebrijes", "Talavera pottery"],
        "people": ["Mexican daily life", "Mexican market vendor", "Mexican artisan"],
    },
    "argentina": {
        "food": ["Argentine cuisine", "asado", "empanada", "mate"],
        "clothing": ["gaucho clothing", "Argentine tango dress"],
        "architecture": ["Buenos Aires architecture", "Argentine estancia", "La Boca"],
        "city": ["Buenos Aires skyline", "Argentine street", "San Telmo market"],
        "nature": ["Patagonia", "Iguazu Falls Argentina", "Andes Argentina"],
        "religion": ["Argentine carnival", "gaucho festival", "Argentine tradition"],
        "art": ["Argentine tango", "fileteado art", "Argentine silverwork"],
        "people": ["Argentine daily life", "gaucho culture", "Buenos Aires cafe"],
    },
    "peru": {
        "food": ["Peruvian cuisine", "ceviche", "lomo saltado", "pisco sour"],
        "clothing": ["Peruvian traditional dress", "Quechua clothing", "chullo hat"],
        "architecture": ["Machu Picchu", "Cusco architecture", "Chan Chan"],
        "city": ["Lima skyline", "Cusco", "Peruvian market"],
        "nature": ["Andes Peru", "Amazon Peru", "Lake Titicaca"],
        "religion": ["Inti Raymi", "Peruvian Semana Santa", "Andean ceremony"],
        "art": ["Peruvian textile", "retablo art", "Peruvian pottery"],
        "people": ["Peruvian daily life", "Andean farmer", "Peruvian market vendor"],
    },
    "australia": {
        "food": ["Australian cuisine", "meat pie", "lamington", "bush tucker"],
        "clothing": ["Aboriginal Australian dress", "akubra hat"],
        "architecture": ["Sydney Opera House", "Australian homestead", "Aboriginal rock art site"],
        "city": ["Sydney skyline", "Melbourne street", "Australian market"],
        "nature": ["Uluru", "Great Barrier Reef", "Australian outback"],
        "religion": ["Aboriginal ceremony", "NAIDOC Week", "Australian bush festival"],
        "art": ["Aboriginal dot painting", "Aboriginal bark art", "didgeridoo"],
        "people": ["Australian daily life", "Aboriginal elder", "Australian beach culture"],
    },
}


class DownloadResult:
    """Result of a single image download."""

    def __init__(
        self,
        success: bool,
        image_path: Path | None = None,
        provenance: CuratedProvenance | None = None,
        error: str | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        self.success = success
        self.image_path = image_path
        self.provenance = provenance
        self.error = error
        self.metadata = metadata or {}


class BaseDownloader(ABC):
    """Abstract base class for platform-specific image downloaders."""

    platform: CurationPlatform

    def __init__(
        self,
        output_dir: Path,
        country: str,
        category: str | None = None,
        limit: int = 50,
    ):
        self.output_dir = Path(output_dir)
        self.country = country.lower()
        self.category = category
        self.limit = limit
        self._staging_dir = self.output_dir / "staging" / self.country
        if self.category:
            self._staging_dir = self._staging_dir / self.category
        self._staging_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def search(self, query: str, limit: int = 50) -> list[dict[str, Any]]:
        """Search the platform for images matching the query.

        Returns a list of result dicts with at minimum:
          - url: str (image URL)
          - source_id: str (platform identifier)
          - license: str (license string)
          - attribution: str
          - title: str
        """

    @abstractmethod
    def download_image(self, url: str, dest: Path) -> bool:
        """Download a single image to the destination path.

        Returns True on success.
        """

    def get_search_terms(self) -> list[str]:
        """Get cultural search terms for the configured country/category."""
        country_terms = CULTURAL_SEARCH_TERMS.get(self.country, {})
        if self.category:
            return country_terms.get(self.category, [f"{self.country} {self.category}"])
        # Return all terms for all categories
        all_terms: list[str] = []
        for terms in country_terms.values():
            all_terms.extend(terms)
        return all_terms

    def generate_image_id(self, source_id: str) -> str:
        """Generate a deterministic image ID from source platform and ID."""
        raw = f"{self.platform.value}:{source_id}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def run(self) -> list[DownloadResult]:
        """Execute the download pipeline for configured country/category.

        Steps:
        1. Get search terms for country/category
        2. Search platform for each term
        3. Download images to staging directory
        4. Create provenance records
        5. Save provenance metadata
        """
        results: list[DownloadResult] = []
        search_terms = self.get_search_terms()
        downloaded_count = 0

        for term in search_terms:
            if downloaded_count >= self.limit:
                break

            remaining = self.limit - downloaded_count
            logger.info(f"Searching '{term}' (remaining: {remaining})")

            try:
                search_results = self.search(term, limit=min(remaining, 20))
            except Exception as e:
                logger.error(f"Search failed for '{term}': {e}")
                continue

            for item in search_results:
                if downloaded_count >= self.limit:
                    break

                image_id = self.generate_image_id(item["source_id"])
                ext = item.get("ext", ".jpg")
                dest = self._staging_dir / f"{image_id}{ext}"

                if dest.exists():
                    logger.debug(f"Skipping existing: {dest}")
                    continue

                try:
                    success = self.download_image(item["url"], dest)
                except Exception as e:
                    logger.error(f"Download failed for {item['url']}: {e}")
                    results.append(DownloadResult(success=False, error=str(e)))
                    continue

                if success:
                    try:
                        license_val = License(item["license"])
                    except ValueError:
                        logger.warning(f"Unknown license '{item['license']}', skipping")
                        dest.unlink(missing_ok=True)
                        continue

                    provenance = CuratedProvenance(
                        original_url=item["url"],
                        license=license_val,
                        attribution=item["attribution"],
                        source_platform=self.platform,
                        source_id=item["source_id"],
                    )
                    results.append(
                        DownloadResult(
                            success=True,
                            image_path=dest,
                            provenance=provenance,
                            metadata=item,
                        )
                    )
                    downloaded_count += 1

        # Save provenance records
        self._save_provenance(results)
        logger.info(f"Downloaded {downloaded_count} images for {self.country}/{self.category or 'all'}")
        return results

    def _save_provenance(self, results: list[DownloadResult]) -> None:
        """Append provenance records to the JSONL file."""
        provenance_file = self.output_dir / "metadata" / "provenance.jsonl"
        provenance_file.parent.mkdir(parents=True, exist_ok=True)

        with open(provenance_file, "a") as f:
            for result in results:
                if result.success and result.provenance:
                    record = {
                        "image_id": self.generate_image_id(result.provenance.source_id),
                        "image_path": str(result.image_path),
                        "country": self.country,
                        "category": self.category or "unknown",
                        **result.provenance.model_dump(mode="json"),
                    }
                    f.write(json.dumps(record, default=str) + "\n")
