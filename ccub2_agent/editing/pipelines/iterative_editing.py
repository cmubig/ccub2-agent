"""
Iterative cultural editing pipeline.

This pipeline:
1. Generates an initial image from prompt
2. Uses VLM to detect cultural issues
3. Generates specific editing instructions
4. Applies image-to-image editing
5. Re-evaluates and repeats until quality threshold met
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class EditingStep:
    """Single step in the editing process."""
    step_num: int
    image_path: Path
    cultural_score: int  # 1-5
    prompt_score: int  # 1-5
    issues: List[Dict]
    editing_prompt: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class EditingResult:
    """Result of iterative editing process."""
    original_prompt: str
    country: str
    category: Optional[str]
    steps: List[EditingStep]
    final_image: Path
    final_cultural_score: int
    final_prompt_score: int
    total_iterations: int
    success: bool


class IterativeEditingPipeline:
    """
    Pipeline for iteratively improving cultural accuracy through editing.

    Workflow:
        Initial Prompt → T2I Generation
             ↓
        VLM Evaluation → Issues Detection
             ↓
        Generate Editing Instructions
             ↓
        I2I Editing → Improved Image
             ↓
        Re-evaluate → Repeat if needed
    """

    def __init__(
        self,
        vlm_detector,
        image_generator,
        output_dir: Path,
        max_iterations: int = 5,
        target_cultural_score: int = 8,
        target_prompt_score: int = 8,
    ):
        """
        Initialize iterative editing pipeline.

        Args:
            vlm_detector: VLMCulturalDetector instance
            image_generator: Image generation model interface
            output_dir: Directory to save intermediate results
            max_iterations: Maximum editing iterations
            target_cultural_score: Target cultural score (1-10)
            target_prompt_score: Target prompt score (1-10)
        """
        self.vlm = vlm_detector
        self.generator = image_generator
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.max_iterations = max_iterations
        self.target_cultural = target_cultural_score
        self.target_prompt = target_prompt_score

        logger.info(f"Initialized IterativeEditingPipeline")
        logger.info(f"  Max iterations: {max_iterations}")
        logger.info(f"  Target scores: cultural={target_cultural_score}, prompt={target_prompt_score}")

    def run(
        self,
        prompt: str,
        country: str,
        category: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> EditingResult:
        """
        Run iterative editing pipeline.

        Args:
            prompt: Initial generation prompt
            country: Target country
            category: Optional category for context
            session_id: Optional session ID for tracking

        Returns:
            EditingResult with full editing history
        """
        if session_id is None:
            session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        session_dir = self.output_dir / session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        logger.info("="*70)
        logger.info("ITERATIVE CULTURAL EDITING PIPELINE")
        logger.info("="*70)
        logger.info(f"Session: {session_id}")
        logger.info(f"Prompt: {prompt}")
        logger.info(f"Country: {country}")
        logger.info(f"Category: {category or 'None'}")
        logger.info("")

        steps = []
        current_prompt = prompt
        editing_prompt = None

        # Step 0: Generate initial image
        logger.info(">>> STEP 0: Initial Generation")
        logger.info(f"Prompt: {current_prompt}")

        image_path = self._generate_image(
            prompt=current_prompt,
            output_path=session_dir / "step_0_initial.png",
            is_editing=False,
        )

        # Evaluate initial image
        cultural_score, prompt_score, issues = self._evaluate_image(
            image_path=image_path,
            prompt=prompt,
            country=country,
            category=category,
            editing_prompt=None,
        )

        steps.append(EditingStep(
            step_num=0,
            image_path=image_path,
            cultural_score=cultural_score,
            prompt_score=prompt_score,
            issues=issues,
            editing_prompt=None,
        ))

        self._log_evaluation(0, cultural_score, prompt_score, issues)

        # Iterative improvement
        for iteration in range(1, self.max_iterations + 1):
            if (cultural_score >= self.target_cultural and
                prompt_score >= self.target_prompt and
                len(issues) == 0):
                logger.info(f"\n✓ Target quality achieved at iteration {iteration-1}!")
                break

            logger.info(f"\n>>> STEP {iteration}: Iterative Editing")

            editing_prompt = self._generate_editing_prompt(
                original_prompt=prompt,
                issues=issues,
                cultural_score=cultural_score,
                prompt_score=prompt_score,
                country=country,
            )

            logger.info(f"Editing instructions: {editing_prompt}")

            prev_image = steps[-1].image_path
            new_image_path = self._generate_image(
                prompt=editing_prompt,
                output_path=session_dir / f"step_{iteration}_edited.png",
                is_editing=True,
                base_image=prev_image,
            )

            cultural_score, prompt_score, issues = self._evaluate_image(
                image_path=new_image_path,
                prompt=prompt,
                country=country,
                category=category,
                editing_prompt=editing_prompt,
            )

            steps.append(EditingStep(
                step_num=iteration,
                image_path=new_image_path,
                cultural_score=cultural_score,
                prompt_score=prompt_score,
                issues=issues,
                editing_prompt=editing_prompt,
            ))

            self._log_evaluation(iteration, cultural_score, prompt_score, issues)

        final_step = steps[-1]
        success = (
            final_step.cultural_score >= self.target_cultural and
            final_step.prompt_score >= self.target_prompt and
            len(final_step.issues) == 0
        )

        result = EditingResult(
            original_prompt=prompt,
            country=country,
            category=category,
            steps=steps,
            final_image=final_step.image_path,
            final_cultural_score=final_step.cultural_score,
            final_prompt_score=final_step.prompt_score,
            total_iterations=len(steps) - 1,
            success=success,
        )

        self._save_summary(session_dir, result)

        logger.info("\n" + "="*70)
        logger.info("FINAL RESULT")
        logger.info("="*70)
        logger.info(f"Total iterations: {result.total_iterations}")
        logger.info(f"Final cultural score: {result.final_cultural_score}/5")
        logger.info(f"Final prompt score: {result.final_prompt_score}/5")
        logger.info(f"Success: {result.success}")
        logger.info(f"Final image: {result.final_image}")
        logger.info("")

        return result

    def _generate_image(
        self,
        prompt: str,
        output_path: Path,
        is_editing: bool = False,
        base_image: Optional[Path] = None,
    ) -> Path:
        """Generate image (T2I or I2I)."""
        if is_editing and base_image:
            logger.info(f"  Editing image from {base_image.name}...")
            image = self.generator.edit(
                prompt=prompt,
                image_path=base_image,
            )
        else:
            logger.info(f"  Generating new image...")
            image = self.generator.generate(prompt=prompt)

        image.save(output_path)
        logger.info(f"  Saved to: {output_path.name}")

        return output_path

    def _evaluate_image(
        self,
        image_path: Path,
        prompt: str,
        country: str,
        category: Optional[str],
        editing_prompt: Optional[str],
    ) -> Tuple[int, int, List[Dict]]:
        """Evaluate image with VLM."""
        logger.info(f"  Evaluating with VLM...")

        cultural_score, prompt_score = self.vlm.score_cultural_quality(
            image_path=image_path,
            prompt=prompt,
            country=country,
            editing_prompt=editing_prompt,
        )

        issues = self.vlm.detect(
            image_path=image_path,
            prompt=prompt,
            country=country,
            editing_prompt=editing_prompt,
            category=category,
        )

        return cultural_score, prompt_score, issues

    def _generate_editing_prompt(
        self,
        original_prompt: str,
        issues: List[Dict],
        cultural_score: int,
        prompt_score: int,
        country: str,
    ) -> str:
        """Generate specific editing instructions from detected issues."""
        if not issues:
            return f"Improve cultural accuracy for {country}"

        sorted_issues = sorted(issues, key=lambda x: x.get('severity', 5), reverse=True)
        instructions = []

        for issue in sorted_issues[:3]:  # Top 3 issues
            desc = issue['description']
            category = issue['category']

            if category == 'cultural_representation':
                instructions.append(f"enhance {country} cultural authenticity")
            elif category == 'prompt_alignment':
                instructions.append(f"better match the prompt '{original_prompt}'")
            elif 'authentic' in desc.lower():
                instructions.append(f"use more authentic {country} elements")
            elif 'western' in desc.lower() or 'foreign' in desc.lower():
                instructions.append(f"remove non-{country} elements")
            elif category == 'food':
                instructions.append(f"show traditional {country} food presentation")
            elif category == 'traditional_clothing':
                instructions.append(f"depict accurate {country} traditional clothing")
            elif category == 'architecture':
                instructions.append(f"use authentic {country} architectural style")
            else:
                instructions.append(desc.lower())

        edit_prompt = f"Edit the image to: " + ", ".join(instructions[:2])

        return edit_prompt

    def _log_evaluation(
        self,
        step: int,
        cultural_score: int,
        prompt_score: int,
        issues: List[Dict],
    ):
        """Log evaluation results."""
        logger.info(f"  Cultural Score: {cultural_score}/5")
        logger.info(f"  Prompt Score: {prompt_score}/5")
        logger.info(f"  Issues Found: {len(issues)}")

        if issues:
            for i, issue in enumerate(issues[:3], 1):
                logger.info(f"    {i}. [{issue['type']}] {issue['description'][:60]}...")

    def _save_summary(self, session_dir: Path, result: EditingResult):
        """Save editing summary to file."""
        import json

        summary = {
            'original_prompt': result.original_prompt,
            'country': result.country,
            'category': result.category,
            'total_iterations': result.total_iterations,
            'success': result.success,
            'final_cultural_score': result.final_cultural_score,
            'final_prompt_score': result.final_prompt_score,
            'steps': [
                {
                    'step': s.step_num,
                    'cultural_score': s.cultural_score,
                    'prompt_score': s.prompt_score,
                    'num_issues': len(s.issues),
                    'editing_prompt': s.editing_prompt,
                }
                for s in result.steps
            ],
        }

        summary_path = session_dir / "summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        logger.info(f"Summary saved to: {summary_path}")
