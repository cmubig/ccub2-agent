"""
Firebase client for direct data access.

Replaces CSV files with direct Firebase Firestore access.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
import json
import logging

logger = logging.getLogger(__name__)


class FirebaseClient:
    """
    Firebase Firestore client for CCUB2 Agent.

    Provides direct access to:
    - contributions collection (replaces contributions.csv)
    - jobs collection (replaces _jobs.csv)
    """

    def __init__(self, config_path: Optional[Path] = None, use_admin_sdk: bool = True):
        """
        Initialize Firebase client.

        Args:
            config_path: Path to Firebase config JSON
            use_admin_sdk: Use firebase-admin SDK (requires service account)
                          If False, uses REST API
        """
        self.config_path = config_path or Path(__file__).parent.parent.parent / ".firebase_config.json"
        self.use_admin_sdk = use_admin_sdk

        # Load config
        with open(self.config_path, 'r') as f:
            self.config = json.load(f)

        self.project_id = self.config['projectId']
        self.storage_bucket = self.config['storageBucket']

        # Initialize Firebase
        if use_admin_sdk:
            self._init_admin_sdk()
        else:
            self._init_rest_api()

    def _init_admin_sdk(self):
        """Initialize firebase-admin SDK."""
        try:
            import firebase_admin
            from firebase_admin import credentials, firestore, storage

            # Check if already initialized
            try:
                app = firebase_admin.get_app()
                logger.info("Firebase already initialized")
            except ValueError:
                # Not initialized yet
                # For admin SDK, we need service account key
                # Try to find it in common locations
                service_account_paths = [
                    Path(__file__).parent.parent.parent / "firebase-service-account.json",
                    Path.home() / ".firebase" / "service-account.json",
                    Path("/etc/firebase/service-account.json"),
                ]

                service_account_key = None
                for path in service_account_paths:
                    if path.exists():
                        service_account_key = str(path)
                        break

                if service_account_key:
                    logger.info(f"Using service account: {service_account_key}")
                    cred = credentials.Certificate(service_account_key)
                    firebase_admin.initialize_app(cred, {
                        'storageBucket': self.storage_bucket
                    })
                else:
                    # Try Application Default Credentials
                    logger.info("Trying Application Default Credentials...")
                    cred = credentials.ApplicationDefault()
                    firebase_admin.initialize_app(cred, {
                        'projectId': self.project_id,
                        'storageBucket': self.storage_bucket
                    })

            self.db = firestore.client()
            self.storage = storage.bucket()
            self.use_admin_sdk = True

            logger.info("Firebase Admin SDK initialized successfully")

        except Exception as e:
            logger.warning(f"Failed to initialize Admin SDK: {e}")
            logger.info("Falling back to REST API...")
            self.use_admin_sdk = False
            self._init_rest_api()

    def _init_rest_api(self):
        """Initialize Firestore REST API client."""
        logger.info("Using Firestore REST API")
        self.db = None
        self.storage = None
        self.base_url = f"https://firestore.googleapis.com/v1/projects/{self.project_id}/databases/(default)/documents"

    def get_contributions(
        self,
        country: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get contributions from Firestore.

        Replaces reading from contributions.csv.

        Args:
            country: Filter by country (optional)
            limit: Max number of items (optional)

        Returns:
            List of contribution dicts with keys:
            - jobId
            - imageURL
            - description
            - category
            - likeCount
        """
        if self.use_admin_sdk and self.db:
            return self._get_contributions_admin(country, limit)
        else:
            return self._get_contributions_rest(country, limit)

    def _get_contributions_admin(
        self,
        country: Optional[str],
        limit: Optional[int]
    ) -> List[Dict[str, Any]]:
        """Get contributions using Admin SDK."""
        try:
            # Query contributions collection
            query = self.db.collection('contributions')

            if country:
                # Need to join with jobs to filter by country
                # For now, fetch all and filter in Python
                pass

            if limit:
                query = query.limit(limit)

            docs = query.stream()

            contributions = []
            for doc in docs:
                data = doc.to_dict()
                contributions.append({
                    'jobId': data.get('jobId', ''),
                    'imageURL': data.get('imageURL', ''),
                    'description': data.get('description', ''),
                    'category': data.get('category', 'general'),
                    'likeCount': data.get('likeCount', 0),
                })

            logger.info(f"Fetched {len(contributions)} contributions from Firebase")
            return contributions

        except Exception as e:
            logger.error(f"Failed to fetch contributions: {e}")
            return []

    def _get_contributions_rest(
        self,
        country: Optional[str],
        limit: Optional[int]
    ) -> List[Dict[str, Any]]:
        """Get contributions using REST API."""
        import requests

        try:
            url = f"{self.base_url}/contributions"

            params = {}
            if limit:
                params['pageSize'] = limit

            response = requests.get(url, params=params)
            response.raise_for_status()

            data = response.json()
            documents = data.get('documents', [])

            contributions = []
            for doc in documents:
                fields = doc.get('fields', {})
                contributions.append({
                    'jobId': fields.get('jobId', {}).get('stringValue', ''),
                    'imageURL': fields.get('imageURL', {}).get('stringValue', ''),
                    'description': fields.get('description', {}).get('stringValue', ''),
                    'category': fields.get('category', {}).get('stringValue', 'general'),
                    'likeCount': int(fields.get('likeCount', {}).get('integerValue', 0)),
                })

            logger.info(f"Fetched {len(contributions)} contributions via REST API")
            return contributions

        except Exception as e:
            logger.error(f"Failed to fetch contributions via REST: {e}")
            return []

    def get_jobs(self) -> List[Dict[str, Any]]:
        """
        Get jobs from Firestore.

        Replaces reading from _jobs.csv.

        Returns:
            List of job dicts with keys:
            - __id__
            - title
            - country
            - category
            - keywords_metadata
        """
        if self.use_admin_sdk and self.db:
            return self._get_jobs_admin()
        else:
            return self._get_jobs_rest()

    def _get_jobs_admin(self) -> List[Dict[str, Any]]:
        """Get jobs using Admin SDK."""
        try:
            docs = self.db.collection('jobs').stream()

            jobs = []
            for doc in docs:
                data = doc.to_dict()
                jobs.append({
                    '__id__': doc.id,
                    'title': data.get('title', ''),
                    'country': data.get('country', ''),
                    'category': data.get('category', ''),
                    'keywords_metadata': data.get('keywords_metadata', []),
                })

            logger.info(f"Fetched {len(jobs)} jobs from Firebase")
            return jobs

        except Exception as e:
            logger.error(f"Failed to fetch jobs: {e}")
            return []

    def _get_jobs_rest(self) -> List[Dict[str, Any]]:
        """Get jobs using REST API."""
        import requests

        try:
            url = f"{self.base_url}/jobs"
            response = requests.get(url)
            response.raise_for_status()

            data = response.json()
            documents = data.get('documents', [])

            jobs = []
            for doc in documents:
                doc_id = doc['name'].split('/')[-1]
                fields = doc.get('fields', {})

                keywords = []
                if 'keywords_metadata' in fields:
                    array_value = fields['keywords_metadata'].get('arrayValue', {})
                    values = array_value.get('values', [])
                    keywords = [v.get('stringValue', '') for v in values]

                jobs.append({
                    '__id__': doc_id,
                    'title': fields.get('title', {}).get('stringValue', ''),
                    'country': fields.get('country', {}).get('stringValue', ''),
                    'category': fields.get('category', {}).get('stringValue', ''),
                    'keywords_metadata': keywords,
                })

            logger.info(f"Fetched {len(jobs)} jobs via REST API")
            return jobs

        except Exception as e:
            logger.error(f"Failed to fetch jobs via REST: {e}")
            return []

    def create_job(self, job_data: Dict[str, Any]) -> Optional[str]:
        """
        Create a new job in Firestore.

        Args:
            job_data: Job data dict

        Returns:
            Job ID if successful, None otherwise
        """
        if self.use_admin_sdk and self.db:
            return self._create_job_admin(job_data)
        else:
            return self._create_job_rest(job_data)

    def _create_job_admin(self, job_data: Dict[str, Any]) -> Optional[str]:
        """Create job using Admin SDK."""
        try:
            job_id = job_data.get('__id__') or self._generate_job_id()

            doc_ref = self.db.collection('jobs').document(job_id)
            doc_ref.set(job_data)

            logger.info(f"Created job {job_id} in Firebase")
            return job_id

        except Exception as e:
            logger.error(f"Failed to create job: {e}")
            return None

    def _create_job_rest(self, job_data: Dict[str, Any]) -> Optional[str]:
        """Create job using REST API."""
        import requests

        try:
            job_id = job_data.get('__id__') or self._generate_job_id()
            url = f"{self.base_url}/jobs?documentId={job_id}"

            # Convert to Firestore format
            fields = {}
            for key, value in job_data.items():
                if isinstance(value, str):
                    fields[key] = {'stringValue': value}
                elif isinstance(value, int):
                    fields[key] = {'integerValue': value}
                elif isinstance(value, list):
                    fields[key] = {
                        'arrayValue': {
                            'values': [{'stringValue': v} for v in value]
                        }
                    }

            payload = {'fields': fields}

            response = requests.post(url, json=payload)
            response.raise_for_status()

            logger.info(f"Created job {job_id} via REST API")
            return job_id

        except Exception as e:
            logger.error(f"Failed to create job via REST: {e}")
            return None

    def _generate_job_id(self) -> str:
        """Generate next available job ID."""
        jobs = self.get_jobs()
        if not jobs:
            return "1"

        # Find highest numeric ID
        highest_id = 0
        for job in jobs:
            try:
                job_id = int(job['__id__'])
                if job_id > highest_id:
                    highest_id = job_id
            except (ValueError, KeyError):
                continue

        return str(highest_id + 1)

    def count_contributions(self, country: Optional[str] = None) -> int:
        """
        Count total contributions.

        Args:
            country: Filter by country (optional)

        Returns:
            Total count
        """
        contributions = self.get_contributions(country=country)
        return len(contributions)


# Convenience functions
def get_firebase_client(config_path: Optional[Path] = None) -> FirebaseClient:
    """Get or create Firebase client singleton."""
    return FirebaseClient(config_path=config_path)
