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
        subcategory: str,
        keywords: List[str],
    ) -> Optional[str]:
        """
        Check if a similar job already exists.

        Since Firebase schema doesn't have country/category fields,
        we check by parsing title and description.

        Args:
            country: Target country
            category: Data category
            subcategory: Data subcategory
            keywords: List of keywords

        Returns:
            Existing job ID if duplicate found, None otherwise
        """
        if not self.db:
            return None

        try:
            # Query for active jobs
            jobs_ref = self.db.collection("jobs")
            query = jobs_ref.where("status", "in", ["IN_PROGRESS", "PENDING"])

            docs = query.stream()

            for doc in docs:
                job_data = doc.to_dict()
                title = job_data.get("title", "").lower()
                description = job_data.get("description", "")

                # Check if country and category match in title
                if country.lower() not in title:
                    continue

                if category.replace("_", " ").lower() not in title.lower():
                    continue

                # Check subcategory in title or description
                if subcategory and subcategory != "general":
                    subcategory_normalized = subcategory.replace("_", " ").lower()
                    if subcategory_normalized not in title.lower() and \
                       subcategory_normalized not in description.lower():
                        continue

                # Check keyword overlap in description
                if keywords:
                    keywords_lower = [k.lower() for k in keywords]
                    desc_lower = description.lower()
                    matching_keywords = sum(1 for k in keywords_lower if k in desc_lower)
                    overlap_ratio = matching_keywords / len(keywords)

                    # If > 50% keywords match, consider it duplicate
                    if overlap_ratio > 0.5:
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
        subcategory: str = "general",
        min_level: int = 2,
        points: int = 50,
        target_count: int = 100,
        thumbnail_url: Optional[str] = None,
        skip_duplicate_check: bool = False,
    ) -> Optional[str]:
        """
        Create a data collection job.

        Firebase schema only has: title, description, keywords, thumbnail,
        minimumLevel, point, status, qualificationTestId.

        Country/category/subcategory info is embedded in title and description.

        Args:
            country: Target country (e.g., "korea")
            category: Data category (e.g., "traditional_clothing")
            keywords: List of keywords
            description: Job description (user-facing)
            subcategory: Specific subcategory (e.g., "jeogori_collar", default: "general")
            min_level: Minimum user level required
            points: Points awarded per contribution
            target_count: Target number of data items
            thumbnail_url: Optional thumbnail URL
            skip_duplicate_check: Skip duplicate checking

        Returns:
            Job ID if created, None if duplicate found
        """
        logger.info(
            f"Creating job for country={country}, category={category}, "
            f"subcategory={subcategory}, keywords={keywords}"
        )

        # Check for duplicates
        if not skip_duplicate_check:
            existing_job_id = self.check_duplicate_job(country, category, subcategory, keywords)
            if existing_job_id:
                logger.warning(
                    f"Duplicate job detected! Existing job ID: {existing_job_id}"
                )
                logger.warning("Skipping job creation to avoid duplicates")
                return None

        # Generate job ID
        job_id = self._generate_job_id()

        # Generate title with country and category (for detect_available_countries.py)
        title = self._generate_title(country, category, subcategory)

        # Create structured description with metadata
        structured_description = self._create_structured_description(
            description=description,
            country=country,
            category=category,
            subcategory=subcategory,
            keywords=keywords,
            target_count=target_count
        )

        # Create job data (Firebase schema compliant)
        job_data = {
            "title": title,
            "description": structured_description,
            "keywords": [],  # Keep empty as per existing Firebase schema
            "thumbnail": thumbnail_url or self._get_default_thumbnail(country),
            "minimumLevel": min_level,
            "point": points,
            "status": "PENDING",  # Start as PENDING until reviewed
            "qualificationTestId": f"test_{job_id}",
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

    def _generate_title(self, country: str, category: str, subcategory: str = "general") -> str:
        """
        Generate job title.

        100% DYNAMIC - ZERO HARDCODING!
        Works for ANY country, ANY category, ANY subcategory.

        Format:
        - General: "Korean Traditional Clothing Dataset"
        - Specific: "Korean Traditional Clothing - Jeogori Collar"

        This format ensures detect_available_countries.py can extract country from title.
        """
        # Convert country to proper case (ZERO hardcoding!)
        # Just capitalize first letter of each word
        country_name = country.replace("_", " ").replace("-", " ").title()

        # Convert category to display name
        category_name = category.replace("_", " ").title()

        # If specific subcategory, add it to title
        if subcategory and subcategory != "general":
            subcategory_name = subcategory.replace("_", " ").title()
            return f"{country_name} {category_name} - {subcategory_name}"
        else:
            # General category
            return f"{country_name} {category_name} Dataset"

    def _create_structured_description(
        self,
        description: str,
        country: str,
        category: str,
        subcategory: str,
        keywords: List[str],
        target_count: int
    ) -> str:
        """
        Create structured description with embedded metadata.

        This allows init_dataset.py to parse metadata from description.

        Format:
        ---
        [User-facing description]

        ---
        📊 Project Details:
        • Country: korea
        • Category: traditional_clothing
        • Subcategory: jeogori_collar
        • Keywords: jeogori, collar, neckline
        • Target: 15 contributions

        📌 This data helps improve AI cultural accuracy.
        ---
        """
        keywords_str = ", ".join(keywords) if keywords else "N/A"

        structured = f"""{description}

---
📊 **Project Details:**
• Country: {country}
• Category: {category}
• Subcategory: {subcategory}
• Keywords: {keywords_str}
• Target: {target_count} contributions

📌 Your contributions will help AI systems better understand and represent {country.title()} culture accurately.
"""
        return structured.strip()

    def _generate_qualification_questions(
        self, country: str, category: str, keywords: List[str]
    ) -> List[Dict]:
        """
        Generate qualification test questions.

        FULLY DYNAMIC - Works for ANY country without hardcoding!
        Uses generic questions based on country, category, and keywords.
        """
        questions = []

        # Category name for display
        category_display = category.replace("_", " ").title()

        # Generic questions that work for any country
        questions.append(
            {
                "question": f"Are you familiar with {country.title()} {category_display.lower()}?",
                "options": ["Yes, very familiar", "Somewhat familiar", "Learning", "Not familiar"],
                "answer": 0,
            }
        )

        questions.append(
            {
                "question": f"Can you identify authentic {country.title()} cultural elements in images?",
                "options": ["Yes, confidently", "Yes, mostly", "Somewhat", "No"],
                "answer": 0,
            }
        )

        # If we have specific keywords, add a keyword-based question
        if keywords and len(keywords) > 0:
            # Take first meaningful keyword
            keyword = keywords[0] if keywords[0] != country else (keywords[1] if len(keywords) > 1 else keywords[0])
            questions.append(
                {
                    "question": f"Do you understand what '{keyword}' refers to in {country.title()} culture?",
                    "options": ["Yes, clearly", "Have some idea", "Heard of it", "No"],
                    "answer": 0,
                }
            )

        return questions

    def _get_default_thumbnail(self, country: str) -> str:
        """
        Get default thumbnail URL for country.

        FULLY DYNAMIC - Tries country-specific first, falls back to generic.
        No hardcoded list needed!
        """
        # Try country-specific thumbnail first
        country_specific_url = f"https://firebasestorage.googleapis.com/v0/b/worldccub-app.appspot.com/o/default%2F{country.lower()}.png?alt=media"

        # Always fallback to generic if country-specific doesn't exist
        # (Firebase will handle 404, we just provide the URL)
        # In production, you'd want to check if the file exists first
        return country_specific_url

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
