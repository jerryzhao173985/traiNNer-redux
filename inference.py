import argparse
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file
from traiNNer.archs import ARCH_REGISTRY
from traiNNer.utils import img2tensor, tensor2img


def get_device():
    """Get the most efficient available device."""
    if not torch.backends.mps.is_available():
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    # Check if MPS is actually working
    try:
        torch.zeros(1).to(torch.device("mps"))
        return torch.device("mps")
    except Exception as e:
        print(f"MPS is available but not working properly: {e}")
        return torch.device("cpu")


def convert_model_to_fp32(model):
    """Convert model parameters and buffers to float32."""
    for param in model.parameters():
        param.data = param.data.float()
        if param._grad is not None:
            param._grad.data = param._grad.data.float()
    for buf in model.buffers():
        buf.data = buf.data.float()
    return model


def load_model(model_path, device):
    """Load model with optimized settings for the device."""
    # Load the model state dict
    if model_path.endswith(".safetensors"):
        state_dict = load_file(model_path)
    else:
        state_dict = torch.load(model_path, map_location="cpu")

    # Extract model parameters from filename
    model_name = Path(model_path).stem
    scale = 2 if "2x" in model_name else 4  # Default to 4x if not specified

    # Determine model type and configuration from filename
    if "RCAN" in model_name:
        model = ARCH_REGISTRY.get("rcan")(
            scale=scale,
            n_resgroups=10,
            n_resblocks=20,
            n_feats=64,
            n_colors=3,
            rgb_range=255,
            norm=False,
        )
    elif "RealPLKSR" in model_name:
        model = ARCH_REGISTRY.get("realplksr")(
            scale=scale,
            dim=96,
            n_blocks=28,
            kernel_size=17,
            split_ratio=0.25,
            use_ea=True,
            layer_norm=True,
        )
    elif "Compact" in model_name:
        # The Compact model uses a VGG-style architecture
        model = ARCH_REGISTRY.get("compact")(
            scale=scale,
            num_feat=64,
            num_conv=16,
            act_type="prelu",
            learn_residual=True,
        )
    elif "ArtCNN" in model_name:
        # Parse model configuration from filename
        # Format: R{num_blocks}F{num_features}
        import re

        match = re.search(r"R(\d+)F(\d+)", model_name)
        if not match:
            raise ValueError(f"Invalid ArtCNN model name format: {model_name}")

        n_block = int(match.group(1))
        filters = int(match.group(2))

        # Use the specific ArtCNN model based on configuration
        if n_block == 16 and filters == 96:
            model = ARCH_REGISTRY.get("artcnn_r16f96")(scale=scale)
        elif n_block == 8 and filters == 64:
            model = ARCH_REGISTRY.get("artcnn_r8f64")(scale=scale)
        elif n_block == 8 and filters == 48:
            model = ARCH_REGISTRY.get("artcnn_r8f48")(scale=scale)
        else:
            model = ARCH_REGISTRY.get("artcnn")(
                scale=scale,
                filters=filters,
                n_block=n_block,
                kernel_size=3,
            )
    else:
        raise ValueError(f"Unknown model type in {model_path}")

    # Convert model to float32 for consistent behavior
    model = convert_model_to_fp32(model)

    # Load state dict and move to device
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model, scale


def process_tile(model, tile, scale, device):
    """Process a single tile with optimized memory handling."""
    # Move to device and ensure float32 for consistent behavior
    tile = tile.to(device).float()

    # Add batch dimension
    tile = tile.unsqueeze(0)

    with torch.no_grad():
        output = model(tile)

    return output.squeeze(0)


def process_image(model, img_path, scale, tile_size=512, overlap=32, device=None):
    """Process image with optimized memory and device handling."""
    if device is None:
        device = get_device()

    # Load and preprocess image
    img = Image.open(img_path).convert("RGB")
    img_np = np.array(img)

    # Convert to tensor with optimal format
    img_tensor = img2tensor(img_np, bgr2rgb=True, float32=True)

    # Get dimensions
    _, h, w = img_tensor.shape

    # If image is small enough, process it directly
    if h <= tile_size and w <= tile_size:
        return process_single_image(model, img_tensor, device)

    # Calculate tiles
    h_tiles = math.ceil(h / (tile_size - overlap))
    w_tiles = math.ceil(w / (tile_size - overlap))

    # Initialize output tensor
    out_h, out_w = h * scale, w * scale
    output = torch.zeros((3, out_h, out_w), device=device)
    output_weights = torch.zeros((out_h, out_w), device=device)

    # Process tiles with progress indication
    total_tiles = h_tiles * w_tiles
    current_tile = 0

    for i in range(h_tiles):
        for j in range(w_tiles):
            current_tile += 1
            print(f"Processing tile {current_tile}/{total_tiles}...")

            # Calculate tile boundaries
            h_start = i * (tile_size - overlap)
            w_start = j * (tile_size - overlap)
            h_end = min(h_start + tile_size, h)
            w_end = min(w_start + tile_size, w)

            # Extract tile
            tile = img_tensor[:, h_start:h_end, w_start:w_end]

            # Process tile
            tile_output = process_tile(model, tile, scale, device)

            # Create weight mask for blending and scale it
            mask = create_weight_mask(h_end - h_start, w_end - w_start, overlap)
            mask = (
                torch.nn.functional.interpolate(
                    mask.unsqueeze(0).unsqueeze(0),
                    scale_factor=scale,
                    mode="bilinear",
                    align_corners=False,
                )
                .squeeze(0)
                .squeeze(0)
                .to(device)
            )

            # Calculate output tile boundaries
            out_h_start = h_start * scale
            out_w_start = w_start * scale
            out_h_end = h_end * scale
            out_w_end = w_end * scale

            # Add processed tile to output
            output[:, out_h_start:out_h_end, out_w_start:out_w_end] += (
                tile_output * mask
            )
            output_weights[out_h_start:out_h_end, out_w_start:out_w_end] += mask

            # Clear cache periodically
            if device.type in ["mps", "cuda"]:
                torch.mps.empty_cache() if device.type == "mps" else torch.cuda.empty_cache()

    # Normalize output by weights
    output = output / (output_weights + 1e-8)

    # Convert back to image
    output_img = tensor2img(output.cpu(), rgb2bgr=True, min_max=(0, 1))

    return Image.fromarray(output_img)


def create_weight_mask(h, w, overlap):
    """Create a weight mask for tile blending."""
    mask = torch.ones((h, w))
    r = overlap // 2

    if r > 0:
        # Create linear ramps for blending
        ramp_h = torch.linspace(0, 1, r)
        ramp_w = torch.linspace(0, 1, r)

        # Apply ramps to edges
        mask[:r, :] *= ramp_h.view(-1, 1)
        mask[-r:, :] *= (1 - ramp_h).view(-1, 1)
        mask[:, :r] *= ramp_w.view(1, -1)
        mask[:, -r:] *= (1 - ramp_w).view(1, -1)

    return mask


def process_single_image(model, img_tensor, device):
    """Process a single image without tiling."""
    # Move to device and ensure float32
    img_tensor = img_tensor.to(device).float()

    # Add batch dimension
    img_tensor = img_tensor.unsqueeze(0)

    with torch.no_grad():
        output = model(img_tensor)

    # Convert back to image
    output_img = tensor2img(output.squeeze(0).cpu(), rgb2bgr=True, min_max=(0, 1))

    return Image.fromarray(output_img)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Super-resolution using traiNNer models"
    )
    parser.add_argument("--model", type=str, required=True, help="Path to model file")
    parser.add_argument("--input", type=str, required=True, help="Path to input image")
    parser.add_argument(
        "--output", type=str, required=True, help="Path to save output image"
    )
    parser.add_argument(
        "--tile_size",
        type=int,
        default=512,
        help="Tile size for processing large images (default: 512)",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=32,
        help="Overlap size between tiles (default: 32)",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["mps", "cuda", "cpu"],
        help="Device to use for processing (default: auto-detect)",
    )

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set device
    if args.device:
        device = torch.device(args.device)
    else:
        device = get_device()

    print(f"Using device: {device}")

    try:
        # Load model
        print("Loading model...")
        model, scale = load_model(args.model, device)

        # Process image
        print("Processing image...")
        output_img = process_image(
            model,
            args.input,
            scale,
            tile_size=args.tile_size,
            overlap=args.overlap,
            device=device,
        )

        # Save result
        print("Saving result...")
        output_img.save(args.output)
        print(f"Processed image saved to {args.output}")

    except Exception as e:
        print(f"Error during processing: {e}")
        if device.type == "mps":
            print("MPS error encountered. Falling back to CPU...")
            device = torch.device("cpu")
            model, scale = load_model(args.model, device)
            output_img = process_image(
                model,
                args.input,
                scale,
                tile_size=args.tile_size,
                overlap=args.overlap,
                device=device,
            )
            output_img.save(args.output)
            print(f"Processed image saved to {args.output} using CPU")


if __name__ == "__main__":
    main()
