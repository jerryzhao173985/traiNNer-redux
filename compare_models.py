import json
import os
from datetime import datetime
from pathlib import Path

import cv2
import lpips
import torch
from inference import get_device, load_model, process_image
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm


class ImageQualityMetrics:
    def __init__(self, device) -> None:
        self.device = device
        self.lpips_model = lpips.LPIPS(net="alex").to(device)

    def calculate_metrics(self, img1_path, img2_path):
        """Calculate various image quality metrics between two images."""
        # Load images
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)

        # Ensure same size for comparison
        if img1.shape != img2.shape:
            img1 = cv2.resize(img1, (img2.shape[1], img2.shape[0]))

        # Convert to RGB
        img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

        # Calculate PSNR
        psnr_value = psnr(img1_rgb, img2_rgb)

        # Calculate SSIM
        ssim_value = ssim(img1_rgb, img2_rgb, channel_axis=2)

        # Calculate LPIPS
        img1_tensor = (
            torch.from_numpy(img1_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        )
        img2_tensor = (
            torch.from_numpy(img2_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        )
        img1_tensor = img1_tensor.to(self.device)
        img2_tensor = img2_tensor.to(self.device)
        lpips_value = float(self.lpips_model(img1_tensor, img2_tensor))

        # Calculate sharpness (using Laplacian variance)
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        sharpness1 = cv2.Laplacian(gray1, cv2.CV_64F).var()
        sharpness2 = cv2.Laplacian(gray2, cv2.CV_64F).var()
        sharpness_ratio = sharpness2 / sharpness1

        return {
            "psnr": float(psnr_value),
            "ssim": float(ssim_value),
            "lpips": float(lpips_value),
            "sharpness_ratio": float(sharpness_ratio),
        }


def process_with_all_models(input_image, models_dir, output_dir, device):
    """Process an image with all available models."""
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = Path(input_image).stem
    result_dir = Path(output_dir) / f"{base_name}_{timestamp}"
    result_dir.mkdir(parents=True, exist_ok=True)

    # Get all model files
    model_files = list(Path(models_dir).glob("*.safetensors"))
    results = {}

    # Process with each model
    print(f"Processing {input_image} with {len(model_files)} models...")
    for model_path in tqdm(model_files):
        model_name = model_path.stem
        output_path = result_dir / f"{model_name}.png"

        try:
            # Load and process
            model, scale = load_model(str(model_path), device)
            output_img = process_image(model, input_image, scale, device=device)
            output_img.save(output_path)

            # Store basic info
            results[model_name] = {
                "path": str(output_path),
                "scale": scale,
                "size": output_img.size,
                "model_size_mb": os.path.getsize(model_path) / (1024 * 1024),
            }

        except Exception as e:
            print(f"Error processing with {model_name}: {e}")
            continue

    return result_dir, results


def compare_results(input_image, results_dict, result_dir, device):
    """Compare all processed images and generate a report."""
    metrics = ImageQualityMetrics(device)
    comparison_results = {}

    # Calculate metrics for each result
    for model_name, info in results_dict.items():
        try:
            metrics_result = metrics.calculate_metrics(input_image, info["path"])
            comparison_results[model_name] = {**info, **metrics_result}
        except Exception as e:
            print(f"Error calculating metrics for {model_name}: {e}")
            continue

    # Calculate overall scores (weighted combination of metrics)
    for model_name in comparison_results:
        metrics = comparison_results[model_name]
        # Higher is better for PSNR, SSIM, and sharpness_ratio
        # Lower is better for LPIPS
        overall_score = (
            0.3 * metrics["psnr"] / 50.0  # Normalize PSNR
            + 0.3 * metrics["ssim"]
            + 0.2 * (1 - metrics["lpips"])  # Invert LPIPS
            + 0.2 * min(metrics["sharpness_ratio"], 2.0) / 2.0  # Cap sharpness ratio
        )
        comparison_results[model_name]["overall_score"] = float(overall_score)

    # Sort results by overall score
    sorted_results = dict(
        sorted(
            comparison_results.items(),
            key=lambda x: x[1]["overall_score"],
            reverse=True,
        )
    )

    # Save detailed report
    report_path = result_dir / "comparison_report.json"
    with open(report_path, "w") as f:
        json.dump(sorted_results, f, indent=4)

    # Generate summary report
    summary = ["Model Comparison Summary:\n"]
    for model_name, metrics in sorted_results.items():
        summary.append(f"\n{model_name}:")
        summary.append(f"Overall Score: {metrics['overall_score']:.4f}")
        summary.append(f"PSNR: {metrics['psnr']:.2f}")
        summary.append(f"SSIM: {metrics['ssim']:.4f}")
        summary.append(f"LPIPS: {metrics['lpips']:.4f}")
        summary.append(f"Sharpness Ratio: {metrics['sharpness_ratio']:.2f}")
        summary.append(f"Model Size: {metrics['model_size_mb']:.1f}MB")
        summary.append(f"Output Size: {metrics['size']}")

    summary_path = result_dir / "summary.txt"
    with open(summary_path, "w") as f:
        f.write("\n".join(summary))

    return sorted_results, summary_path


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Compare super-resolution models")
    parser.add_argument("--input", type=str, required=True, help="Input image path")
    parser.add_argument(
        "--models_dir",
        type=str,
        default="pretrained_model",
        help="Directory containing model files",
    )
    parser.add_argument(
        "--output_dir", type=str, default="results", help="Directory to save results"
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["mps", "cuda", "cpu"],
        help="Device to use (default: auto-detect)",
    )
    args = parser.parse_args()

    # Setup device
    device = torch.device(args.device) if args.device else get_device()
    print(f"Using device: {device}")

    # Process image with all models
    result_dir, results = process_with_all_models(
        args.input, args.models_dir, args.output_dir, device
    )

    # Compare results
    sorted_results, summary_path = compare_results(
        args.input, results, result_dir, device
    )

    # Print summary
    print(f"\nResults saved in: {result_dir}")
    print(f"Summary saved to: {summary_path}")
    print("\nTop 3 Models:")
    for i, (model_name, metrics) in enumerate(list(sorted_results.items())[:3]):
        print(f"{i + 1}. {model_name}")
        print(f"   Score: {metrics['overall_score']:.4f}")
        print(f"   PSNR: {metrics['psnr']:.2f}")
        print(f"   SSIM: {metrics['ssim']:.4f}")
        print(f"   LPIPS: {metrics['lpips']:.4f}")


if __name__ == "__main__":
    main()
