"""
Agent job creator - automatically creates data collection jobs on Firebase.

Now uses FirebaseClient for unified Firebase access.
"""

from typing import Dict, List, Optional
import logging
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class AgentJobCreator:
    """
    Automatically create data collection jobs when data gaps are detected.

    This is the key component that makes the system self-improving!
    When the agent detects missing data, it creates a job on WorldCCUB
    to collect that specific type of data.
    """

    def __init__(self, firebase_config: Optional[Path] = None):
        """
        Initialize job creator with Firebase connection.

        Args:
            firebase_config: Path to Firebase config JSON (optional)
                           If None, uses default .firebase_config.json
        """
        self.firebase_config = firebase_config
        self.firebase = None

        try:
            from .firebase_client import get_firebase_client
            self.firebase = get_firebase_client(config_path=firebase_config)
            logger.info("Firebase client initialized for job creation")
        except Exception as e:
            logger.warning(f"Failed to initialize Firebase client: {e}")
            logger.warning("Running in dry-run mode (jobs will be logged but not created)")

    @property
    def db(self):
        """Compatibility property for db access."""
        return self.firebase.db if self.firebase else None

    @property
    def storage(self):
        """Compatibility property for storage access."""
        return self.firebase.storage if self.firebase else None

    def check_duplicate_job(
        self,
        country: str,
        category: str,
        keywords: List[str],
    ) -> Optional[str]:
        """
        Check if a similar job already exists.

        Args:
            country: Target country
            category: Data category
            keywords: List of keywords

        Returns:
            Existing job ID if duplicate found, None otherwise
        """
        if not self.db:
            return None

        try:
            # Query for active jobs with same country and category
            jobs_ref = self.db.collection("jobs")
            query = (
                jobs_ref.where("country", "==", country)
                .where("category", "==", category)
                .where("status", "in", ["IN_PROGRESS", "PENDING"])
            )

            docs = query.stream()

            for doc in docs:
                job_data = doc.to_dict()

                # Check keyword overlap
                existing_keywords = job_data.get("keywords_metadata", [])
                if not existing_keywords:
                    continue

                # Calculate overlap
                common_keywords = set(keywords) & set(existing_keywords)
                overlap_ratio = len(common_keywords) / max(len(keywords), len(existing_keywords))

                # If > 70% overlap, consider it duplicate
                if overlap_ratio > 0.7:
                    logger.info(
                        f"Found duplicate job {doc.id}: {overlap_ratio:.1%} keyword overlap"
                    )
                    return doc.id

            return None

        except Exception as e:
            logger.error(f"Failed to check duplicate: {e}")
            return None

    def create_job(
        self,
        country: str,
        category: str,
        keywords: List[str],
        description: str,
        min_level: int = 2,
        points: int = 50,
        target_count: int = 100,
        thumbnail_url: Optional[str] = None,
        skip_duplicate_check: bool = False,
    ) -> Optional[str]:
        """
        Create a data collection job.

        Args:
            country: Target country (e.g., "korea")
            category: Data category (e.g., "traditional_clothing", "text_hangul")
            keywords: List of keywords
            description: Job description
            min_level: Minimum user level required
            points: Points awarded per contribution
            target_count: Target number of data items
            thumbnail_url: Optional thumbnail URL
            skip_duplicate_check: Skip duplicate checking

        Returns:
            Job ID if created, None if duplicate found
        """
        logger.info(
            f"Creating job for country={country}, category={category}, keywords={keywords}"
        )

        # Check for duplicates
        if not skip_duplicate_check:
            existing_job_id = self.check_duplicate_job(country, category, keywords)
            if existing_job_id:
                logger.warning(
                    f"Duplicate job detected! Existing job ID: {existing_job_id}"
                )
                logger.warning("Skipping job creation to avoid duplicates")
                return None

        # Generate job ID
        job_id = self._generate_job_id()

        # Create job data
        job_data = {
            "title": self._generate_title(country, category),
            "description": description,
            "keywords": [],  # Keep empty as per existing system
            "thumbnail": thumbnail_url or self._get_default_thumbnail(country),
            "minimumLevel": min_level,
            "point": points,
            "status": "PENDING",  # Start as PENDING until reviewed
            "qualificationTestId": f"test_{job_id}",
            "createdBy": "agent",
            "createdAt": self._get_timestamp(),
            "category": category,
            "country": country,
            "targetCount": target_count,
            "currentCount": 0,
            "keywords_metadata": keywords,  # Store separately for agent use
        }

        # Create qualification test
        test_data = {
            "id": f"test_{job_id}",
            "jobId": job_id,
            "questions": self._generate_qualification_questions(
                country, category, keywords
            ),
        }

        # Upload to Firebase
        if self.db:
            try:
                # Save job
                self.db.collection("jobs").document(job_id).set(job_data)

                # Save qualification test
                self.db.collection("qualification-tests").document(
                    f"test_{job_id}"
                ).set(test_data)

                logger.info(f"Job created successfully with ID: {job_id}")
            except Exception as e:
                logger.error(f"Failed to create job: {e}")
                raise
        else:
            logger.info(f"Dry-run mode - job data: {json.dumps(job_data, indent=2)}")

        return job_id

    def _generate_job_id(self) -> str:
        """Generate next available job ID."""
        if self.firebase:
            try:
                return self.firebase._generate_job_id()
            except Exception as e:
                logger.error(f"Failed to generate job ID: {e}")
                # Fallback to timestamp
                import time
                return str(int(time.time()))
        else:
            # Dry-run mode
            import time
            return str(int(time.time()))

    def _generate_title(self, country: str, category: str) -> str:
        """Generate job title."""
        country_names = {
            "korea": "Korean",
            "japan": "Japanese",
            "china": "Chinese",
            "india": "Indian",
            "nigeria": "Nigerian",
            "kenya": "Kenyan",
        }

        category_names = {
            "traditional_clothing": "Traditional Clothing",
            "text": "Text and Writing",
            "architecture": "Architecture",
            "food": "Food and Cuisine",
            "symbols": "Cultural Symbols",
            "festivals": "Festivals and Celebrations",
        }

        country_name = country_names.get(country, country.title())
        category_name = category_names.get(category, category.replace("_", " ").title())

        return f"{country_name} {category_name} Data Collection"

    def _generate_qualification_questions(
        self, country: str, category: str, keywords: List[str]
    ) -> List[Dict]:
        """
        Generate qualification test questions.

        These ensure that contributors understand the cultural context.
        """
        # Question templates by country and category
        questions = []

        if country == "korea":
            if "traditional_clothing" in category or "hanbok" in keywords:
                questions.append(
                    {
                        "question": "Which of these is a traditional Korean garment?",
                        "options": ["Hanbok", "Kimono", "Sari", "Ao Dai"],
                        "answer": 0,
                    }
                )
                questions.append(
                    {
                        "question": "What is the name of the Korean jacket in hanbok?",
                        "options": ["Jeogori", "Haori", "Cheongsam", "Kurta"],
                        "answer": 0,
                    }
                )

            if "text" in category or "hangul" in keywords:
                questions.append(
                    {
                        "question": "What is the Korean writing system called?",
                        "options": ["Hangul", "Kanji", "Hiragana", "Pinyin"],
                        "answer": 0,
                    }
                )

        # Generic fallback questions
        if not questions:
            questions.append(
                {
                    "question": f"Are you familiar with {country.title()} culture?",
                    "options": ["Yes", "Somewhat", "Learning", "No"],
                    "answer": 0,
                }
            )
            questions.append(
                {
                    "question": f"Can you identify authentic {country.title()} cultural elements?",
                    "options": ["Yes, confidently", "Yes, mostly", "Somewhat", "No"],
                    "answer": 0,
                }
            )

        return questions

    def _get_default_thumbnail(self, country: str) -> str:
        """Get default thumbnail URL for country."""
        # TODO: Upload default thumbnails to Firebase Storage
        default_thumbnails = {
            "korea": "https://firebasestorage.googleapis.com/v0/b/worldccub-app.appspot.com/o/default%2Fkorea.png?alt=media",
            "japan": "https://firebasestorage.googleapis.com/v0/b/worldccub-app.appspot.com/o/default%2Fjapan.png?alt=media",
            "china": "https://firebasestorage.googleapis.com/v0/b/worldccub-app.appspot.com/o/default%2Fchina.png?alt=media",
            "india": "https://firebasestorage.googleapis.com/v0/b/worldccub-app.appspot.com/o/default%2Findia.png?alt=media",
            "nigeria": "https://firebasestorage.googleapis.com/v0/b/worldccub-app.appspot.com/o/default%2Fnigeria.png?alt=media",
            "kenya": "https://firebasestorage.googleapis.com/v0/b/worldccub-app.appspot.com/o/default%2Fkenya.png?alt=media",
        }

        return default_thumbnails.get(
            country,
            "https://firebasestorage.googleapis.com/v0/b/worldccub-app.appspot.com/o/default%2Fgeneric.png?alt=media",
        )

    def _get_timestamp(self) -> str:
        """Get current timestamp in ISO format."""
        from datetime import datetime

        return datetime.utcnow().isoformat() + "Z"

    def get_job_status(self, job_id: str) -> Dict:
        """
        Get current status of a job.

        Args:
            job_id: Job ID

        Returns:
            Job status dict
        """
        if not self.db:
            return {"error": "Firebase not initialized"}

        try:
            doc = self.db.collection("jobs").document(job_id).get()
            if doc.exists:
                return doc.to_dict()
            else:
                return {"error": f"Job {job_id} not found"}
        except Exception as e:
            logger.error(f"Failed to get job status: {e}")
            return {"error": str(e)}

    def get_approved_contributions(
        self,
        country: Optional[str] = None,
        category: Optional[str] = None,
        since_timestamp: Optional[str] = None,
    ) -> List[Dict]:
        """
        Get approved contributions from Firebase.

        Args:
            country: Filter by country (optional)
            category: Filter by category (optional)
            since_timestamp: Get contributions after this timestamp (optional)

        Returns:
            List of approved contribution dicts with:
                - id: Contribution ID
                - jobId: Associated job ID
                - userId: User ID
                - imageUrl: Uploaded image URL
                - metadata: Additional metadata
                - approvedAt: Approval timestamp
                - country: Country
                - category: Category
        """
        if not self.db:
            logger.warning("Firebase not initialized")
            return []

        try:
            # Query contributions
            contributions_ref = self.db.collection("contributions")
            query = contributions_ref.where("status", "==", "APPROVED")

            if country:
                query = query.where("country", "==", country)

            if category:
                query = query.where("category", "==", category)

            if since_timestamp:
                query = query.where("approvedAt", ">=", since_timestamp)

            # Order by approval time
            query = query.order_by("approvedAt", direction="DESCENDING")

            docs = query.stream()

            contributions = []
            for doc in docs:
                data = doc.to_dict()
                data['id'] = doc.id
                contributions.append(data)

            logger.info(f"Found {len(contributions)} approved contributions")
            return contributions

        except Exception as e:
            logger.error(f"Failed to get approved contributions: {e}")
            return []

    def download_approved_image(self, image_url: str, local_path: Path) -> bool:
        """
        Download an approved contribution image from Firebase Storage.

        Args:
            image_url: Firebase Storage URL
            local_path: Local path to save image

        Returns:
            True if successful, False otherwise
        """
        if not self.storage:
            logger.warning("Firebase Storage not initialized")
            return False

        try:
            import requests

            # Download image
            response = requests.get(image_url, timeout=30)
            response.raise_for_status()

            # Save to local path
            local_path.parent.mkdir(parents=True, exist_ok=True)
            with open(local_path, 'wb') as f:
                f.write(response.content)

            logger.info(f"Downloaded image to {local_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to download image: {e}")
            return False
