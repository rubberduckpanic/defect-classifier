"""
Module M5: Drift Simulation
Simulates distribution shift by applying realistic image transformations
that mimic production environment changes.

Simulated Scenarios:
1. Lighting change — Gradually increase/decrease brightness
2. Angle variation — Apply rotation transforms
3. Camera degradation — Add noise, blur
4. Mixed shift — Combination of the above

These simulations model real-world manufacturing scenarios:
- Factory lighting changes (day/night shifts, bulb degradation)
- Camera alignment drift over time
- Lens fouling or sensor aging
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
from torchvision import transforms

from src.monitoring.drift_detector import DriftDetector

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class DriftSimulator:
    """
    Simulates various types of data drift on production images.
    
    Provides controlled degradation scenarios to test monitoring system.
    """

    def __init__(self, random_seed: int = 42):
        self.rng = np.random.default_rng(random_seed)

    def simulate_lighting_change(
        self, image: Image.Image, severity: float = 0.5
    ) -> Image.Image:
        """
        Simulate gradual lighting change.
        
        Scenario: Factory shifts from day to night, or lighting fixtures degrade.
        
        Args:
            image: Input PIL image
            severity: 0.0 (no change) to 1.0 (extreme darkening/brightening)
        """
        # Random direction: darker or brighter
        direction = self.rng.choice([-1, 1])
        factor = 1.0 + direction * severity * 0.6  # Range: [0.4, 1.6]

        enhancer = ImageEnhance.Brightness(image)
        return enhancer.enhance(factor)

    def simulate_angle_variation(
        self, image: Image.Image, severity: float = 0.5
    ) -> Image.Image:
        """
        Simulate camera angle drift.
        
        Scenario: Camera mount loosens over time, slight rotation occurs.
        
        Args:
            image: Input PIL image
            severity: 0.0 (no rotation) to 1.0 (up to 20 degrees)
        """
        max_angle = severity * 20
        angle = self.rng.uniform(-max_angle, max_angle)
        return image.rotate(angle, fillcolor=(128, 128, 128))

    def simulate_noise(
        self, image: Image.Image, severity: float = 0.5
    ) -> Image.Image:
        """
        Simulate sensor noise / image degradation.
        
        Scenario: Camera sensor aging, electromagnetic interference.
        
        Args:
            image: Input PIL image
            severity: 0.0 (no noise) to 1.0 (heavy noise)
        """
        img_array = np.array(image).astype(np.float32)
        noise_std = severity * 50  # Up to 50 pixel value units of noise
        noise = self.rng.normal(0, noise_std, img_array.shape)
        noisy = np.clip(img_array + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy)

    def simulate_blur(
        self, image: Image.Image, severity: float = 0.5
    ) -> Image.Image:
        """
        Simulate lens fouling / focus drift.
        
        Scenario: Dust accumulation on lens, vibration causing defocus.
        """
        radius = severity * 3  # Blur radius up to 3 pixels
        if radius > 0.5:
            return image.filter(ImageFilter.GaussianBlur(radius=radius))
        return image

    def simulate_contrast_change(
        self, image: Image.Image, severity: float = 0.5
    ) -> Image.Image:
        """
        Simulate contrast degradation.
        
        Scenario: Display/capture settings drift, environmental haze.
        """
        factor = 1.0 - severity * 0.5  # Reduce contrast
        enhancer = ImageEnhance.Contrast(image)
        return enhancer.enhance(factor)

    def apply_gradual_drift(
        self,
        image: Image.Image,
        drift_type: str = "lighting",
        step: int = 0,
        total_steps: int = 100
    ) -> Image.Image:
        """
        Apply drift that gradually increases over time.
        
        Models real-world degradation that happens incrementally.
        
        Args:
            image: Input image
            drift_type: Type of drift to simulate
            step: Current time step
            total_steps: Total simulation steps
        """
        severity = min(step / total_steps, 1.0)  # Linear increase

        if drift_type == "lighting":
            return self.simulate_lighting_change(image, severity)
        elif drift_type == "angle":
            return self.simulate_angle_variation(image, severity)
        elif drift_type == "noise":
            return self.simulate_noise(image, severity)
        elif drift_type == "blur":
            return self.simulate_blur(image, severity)
        elif drift_type == "mixed":
            # Apply multiple drift types simultaneously
            img = self.simulate_lighting_change(image, severity * 0.5)
            img = self.simulate_noise(img, severity * 0.3)
            img = self.simulate_angle_variation(img, severity * 0.3)
            return img
        else:
            raise ValueError(f"Unknown drift type: {drift_type}")


def run_drift_simulation(
    model_path: str = "models/best_resnet18.pt",
    test_dir: str = "data/splits/test",
    drift_type: str = "lighting",
    num_steps: int = 100,
    output_dir: str = "logs"
) -> Dict:
    """
    Run a full drift simulation experiment.
    
    Applies gradual drift to test images and monitors model confidence.
    
    Args:
        model_path: Path to trained model
        test_dir: Directory with test images
        drift_type: Type of drift to simulate
        num_steps: Number of drift steps
        output_dir: Directory to save results
        
    Returns:
        Simulation results dictionary
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint.get("config", {})
    model_config = config.get("model", {})

    from src.training.models import get_model
    model = get_model(
        architecture=model_config.get("architecture", "resnet18"),
        num_classes=model_config.get("num_classes", 2),
        pretrained=False,
        dropout=model_config.get("dropout", 0.3)
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    # Setup transforms (no augmentation, just normalize)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Collect test images
    test_path = Path(test_dir)
    test_images = []
    for class_dir in test_path.iterdir():
        if class_dir.is_dir():
            for img_file in list(class_dir.iterdir())[:50]:  # Sample 50 per class
                if img_file.suffix.lower() in {'.png', '.jpg', '.jpeg'}:
                    test_images.append((img_file, class_dir.name))

    logger.info(f"Loaded {len(test_images)} test images for simulation")

    # Setup drift detector with reference confidence
    # First, get reference (no-drift) confidence scores
    reference_confidences = []
    simulator = DriftSimulator()

    with torch.no_grad():
        for img_path, label in test_images:
            img = Image.open(img_path).convert("RGB")
            tensor = transform(img).unsqueeze(0).to(device)
            output = model(tensor)
            prob = torch.softmax(output, dim=1).max().item()
            reference_confidences.append(prob)

    detector = DriftDetector(
        reference_confidence=reference_confidences,
        window_size=min(50, len(test_images)),
    )

    # Run simulation with gradual drift
    results_per_step = []

    for step in range(num_steps):
        step_confidences = []
        step_correct = 0
        step_total = 0

        with torch.no_grad():
            for img_path, true_label in test_images:
                img = Image.open(img_path).convert("RGB")

                # Apply drift
                drifted_img = simulator.apply_gradual_drift(
                    img, drift_type=drift_type, step=step, total_steps=num_steps
                )

                # Get prediction
                tensor = transform(drifted_img).unsqueeze(0).to(device)
                output = model(tensor)
                probs = torch.softmax(output, dim=1)
                confidence, pred_idx = probs.max(dim=1)

                # Compute image brightness for monitoring
                brightness = np.array(drifted_img.convert("L")).mean() / 255.0

                # Log prediction
                pred_label = "defective" if pred_idx.item() == 1 else "non_defective"
                detector.log_prediction({
                    "confidence": confidence.item(),
                    "label": pred_label,
                    "true_label": true_label,
                    "timestamp": f"step_{step}",
                    "image_brightness": brightness,
                })

                step_confidences.append(confidence.item())
                if pred_label == true_label:
                    step_correct += 1
                step_total += 1

        step_accuracy = step_correct / step_total if step_total > 0 else 0
        step_result = {
            "step": step,
            "severity": step / num_steps,
            "mean_confidence": float(np.mean(step_confidences)),
            "accuracy": step_accuracy,
            "num_alerts": len(detector.drift_alerts),
        }
        results_per_step.append(step_result)

        if step % 10 == 0:
            logger.info(
                f"Step {step}/{num_steps}: Confidence={np.mean(step_confidences):.3f}, "
                f"Accuracy={step_accuracy:.3f}, Alerts={len(detector.drift_alerts)}"
            )

    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    simulation_results = {
        "drift_type": drift_type,
        "num_steps": num_steps,
        "num_test_images": len(test_images),
        "reference_mean_confidence": float(np.mean(reference_confidences)),
        "final_mean_confidence": results_per_step[-1]["mean_confidence"],
        "final_accuracy": results_per_step[-1]["accuracy"],
        "total_alerts": len(detector.drift_alerts),
        "results_per_step": results_per_step,
        "drift_alerts": detector.drift_alerts,
        "summary": detector.get_summary(),
    }

    results_file = output_path / f"drift_simulation_{drift_type}.json"
    with open(results_file, "w") as f:
        json.dump(simulation_results, f, indent=2, default=str)

    # Save monitoring log
    detector.save_monitoring_log(str(output_path / "monitoring_log.json"))

    logger.info(f"Simulation complete. Results saved to {results_file}")
    return simulation_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run drift simulation")
    parser.add_argument("--model", type=str, default="models/best_resnet18.pt",
                        help="Path to model checkpoint")
    parser.add_argument("--test-dir", type=str, default="data/splits/test",
                        help="Test data directory")
    parser.add_argument("--drift-type", type=str, default="lighting",
                        choices=["lighting", "angle", "noise", "blur", "mixed"],
                        help="Type of drift to simulate")
    parser.add_argument("--steps", type=int, default=50,
                        help="Number of drift steps")
    args = parser.parse_args()

    results = run_drift_simulation(
        model_path=args.model,
        test_dir=args.test_dir,
        drift_type=args.drift_type,
        num_steps=args.steps,
    )

    print(f"\nSimulation Summary:")
    print(f"  Drift Type: {results['drift_type']}")
    print(f"  Reference Confidence: {results['reference_mean_confidence']:.3f}")
    print(f"  Final Confidence: {results['final_mean_confidence']:.3f}")
    print(f"  Final Accuracy: {results['final_accuracy']:.3f}")
    print(f"  Total Alerts: {results['total_alerts']}")


