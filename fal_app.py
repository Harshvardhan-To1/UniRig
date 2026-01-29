"""
UniRig FAL App - Automatic 3D Model Rigging

This app provides automatic skeleton prediction and skinning weight generation
for 3D models using the UniRig framework from VAST-AI-Research.

UniRig is a unified framework for automatic 3D model rigging that:
- Predicts topologically valid skeleton structures using an autoregressive model
- Generates per-vertex skinning weights using bone-point cross attention
- Handles diverse 3D models (humans, animals, objects) with a single model

Paper: "One Model to Rig Them All: Diverse Skeleton Rigging with UniRig" (SIGGRAPH'25)
GitHub: https://github.com/VAST-AI-Research/UniRig
Model: https://huggingface.co/VAST-AI/UniRig
"""

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Literal

import fal
from fal.exceptions import FieldException
from fal.toolkit import FAL_MODEL_WEIGHTS_DIR, File, clone_repository
from fastapi import Request, Response
from pydantic import BaseModel, Field


# Output format options
OUTPUT_FORMAT_LITERAL = Literal["fbx", "glb"]
DEFAULT_OUTPUT_FORMAT: OUTPUT_FORMAT_LITERAL = "fbx"
DEFAULT_SEED: int = 12345
DEFAULT_FACES_TARGET_COUNT: int = 50000

# Supported input formats
SUPPORTED_FORMATS = ["obj", "fbx", "glb", "gltf", "vrm", "dae"]


def get_seed(seed: int | None) -> int:
    """Get seed value, generating random one if not provided."""
    if seed is None:
        import random
        return random.randint(0, 2**32 - 1)
    return seed


def safe_hf_download(
    repo_id: str,
    filename: str,
    local_dir: str,
    max_retries: int = 3,
) -> str:
    """Download a file from HuggingFace with retry logic."""
    from huggingface_hub import hf_hub_download
    import time
    
    last_error = None
    for attempt in range(max_retries):
        try:
            return hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=local_dir,
            )
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                wait_time = 4 * (2 ** attempt)  # 4s, 8s, 16s
                print(f"Download attempt {attempt + 1} failed: {e}, retrying in {wait_time}s...")
                time.sleep(wait_time)
    
    raise RuntimeError(f"Failed to download {filename} from {repo_id}: {last_error}")


class UniRigInput(BaseModel):
    """Input schema for UniRig rigging."""
    
    mesh_file: str = Field(
        description="URL to the input 3D mesh file (supports .obj, .fbx, .glb, .gltf, .vrm, .dae)",
        examples=[
            "https://example.com/model.glb",
            "https://example.com/character.fbx",
        ],
    )
    seed: int | None = Field(
        default=None,
        description="Random seed for reproducible skeleton generation.",
        examples=[42, 12345],
    )
    output_format: OUTPUT_FORMAT_LITERAL = Field(
        default=DEFAULT_OUTPUT_FORMAT,
        description="Output format: 'fbx' or 'glb'.",
    )
    faces_target_count: int = Field(
        default=DEFAULT_FACES_TARGET_COUNT,
        ge=1000,
        le=200000,
        description="Target number of faces for mesh simplification (used for processing large meshes).",
    )


class SkeletonOnlyInput(BaseModel):
    """Input schema for skeleton-only prediction."""
    
    mesh_file: str = Field(
        description="URL to the input 3D mesh file (supports .obj, .fbx, .glb, .gltf, .vrm, .dae)",
        examples=[
            "https://example.com/model.glb",
        ],
    )
    seed: int | None = Field(
        default=None,
        description="Random seed for reproducible skeleton generation.",
        examples=[42, 12345],
    )
    output_format: OUTPUT_FORMAT_LITERAL = Field(
        default=DEFAULT_OUTPUT_FORMAT,
        description="Output format: 'fbx' or 'glb'.",
    )
    faces_target_count: int = Field(
        default=DEFAULT_FACES_TARGET_COUNT,
        ge=1000,
        le=200000,
        description="Target number of faces for mesh simplification.",
    )


class SkinOnlyInput(BaseModel):
    """Input schema for skinning prediction (requires mesh with skeleton)."""
    
    mesh_file: str = Field(
        description="URL to the input 3D mesh file with skeleton (typically .fbx from skeleton endpoint)",
        examples=[
            "https://example.com/model_with_skeleton.fbx",
        ],
    )
    original_mesh_file: str | None = Field(
        default=None,
        description="URL to the original mesh file to merge textures/materials from. If not provided, uses mesh_file.",
        examples=[
            "https://example.com/original_model.glb",
        ],
    )
    output_format: OUTPUT_FORMAT_LITERAL = Field(
        default=DEFAULT_OUTPUT_FORMAT,
        description="Output format: 'fbx' or 'glb'.",
    )
    faces_target_count: int = Field(
        default=DEFAULT_FACES_TARGET_COUNT,
        ge=1000,
        le=200000,
        description="Target number of faces for mesh simplification.",
    )


class UniRigOutput(BaseModel):
    """Output schema for full UniRig rigging."""
    
    rigged_file: File = Field(
        description="The fully rigged 3D model file with skeleton and skinning weights.",
    )
    skeleton_file: File | None = Field(
        default=None,
        description="Intermediate skeleton-only file.",
    )
    seed: int = Field(
        description="Seed used for generation.",
        examples=[42],
    )


class SkeletonOutput(BaseModel):
    """Output schema for skeleton-only prediction."""
    
    skeleton_file: File = Field(
        description="The predicted skeleton file (FBX or GLB format).",
    )
    seed: int = Field(
        description="Seed used for generation.",
    )


class SkinOutput(BaseModel):
    """Output schema for skinning prediction."""
    
    skinned_file: File = Field(
        description="The skinned 3D model file with weights applied.",
    )


class HealthOutput(BaseModel):
    """Output schema for health check."""
    
    status: str = Field(description="Service status")
    version: str = Field(description="UniRig version info")
    gpu_available: bool = Field(description="Whether GPU is available")


class UniRig(
    fal.App,
    name="unirig",
    min_concurrency=0,
    max_concurrency=2,
    keep_alive=1800,
    startup_timeout=1800,
    request_timeout=3600,
):  # type: ignore

    machine_type = "GPU-A100"
    num_gpus = 1

    requirements = [
        # Core ML dependencies
        "torch==2.5.1+cu124",
        "torchvision==0.20.1+cu124",
        "transformers==4.51.3",
        "huggingface_hub>=0.20.0",
        "lightning>=2.0.0",
        "pytorch_lightning>=2.0.0",
        # Flash attention
        "flash_attn>=2.5.0",
        # 3D processing
        "trimesh>=4.0.0",
        "open3d>=0.18.0",
        "fast-simplification>=0.1.0",
        "pyrender>=0.1.45",
        # Blender Python API
        "bpy==4.2",
        # Other dependencies
        "python-box>=7.0.0",
        "einops>=0.7.0",
        "omegaconf>=2.3.0",
        "addict>=2.4.0",
        "timm>=0.9.0",
        "scipy>=1.10.0",
        "numpy>=1.24.0,<2.0",
        "wandb>=0.16.0",
        "requests>=2.28.0",
        "PyYAML>=6.0",
        "psutil>=5.9.0",
        # PyTorch Geometric dependencies
        "torch_scatter",
        "torch_cluster",
        # Spconv for sparse convolutions
        "spconv-cu124>=2.3.0",
        # Extra index URLs
        "--extra-index-url",
        "https://download.pytorch.org/whl/cu124",
        "--extra-index-url",
        "https://data.pyg.org/whl/torch-2.5.0+cu124.html",
    ]

    GITHUB_REPO = "VAST-AI-Research/UniRig"
    GITHUB_COMMIT = "main"
    HF_REPO_ID = "VAST-AI/UniRig"
    
    # Model checkpoint paths in HuggingFace repo
    SKELETON_CKPT = "skeleton/articulation-xl_quantization_256/model.ckpt"
    SKIN_CKPT = "skin/articulation-xl/model.ckpt"

    async def setup(self) -> None:
        """Initialize the UniRig models and environment."""
        import torch
        
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
        
        # Clone the UniRig repository
        print("Cloning UniRig repository...")
        self.repo_dir = str(
            clone_repository(
                f"https://github.com/{self.GITHUB_REPO}.git",
                commit_hash=self.GITHUB_COMMIT,
                include_to_path=True,
                repo_name="unirig",
            )
        )
        print(f"Repository cloned to: {self.repo_dir}")
        
        # Add repo to Python path
        if self.repo_dir not in sys.path:
            sys.path.insert(0, self.repo_dir)
        
        # Download model checkpoints from HuggingFace with retry logic
        print("Downloading model checkpoints from HuggingFace...")
        weights_dir = Path(FAL_MODEL_WEIGHTS_DIR) / "unirig"
        weights_dir.mkdir(parents=True, exist_ok=True)
        
        # Download skeleton model checkpoint
        self.skeleton_ckpt_path = safe_hf_download(
            repo_id=self.HF_REPO_ID,
            filename=self.SKELETON_CKPT,
            local_dir=str(weights_dir),
        )
        print(f"Skeleton checkpoint downloaded: {self.skeleton_ckpt_path}")
        
        # Download skin model checkpoint
        self.skin_ckpt_path = safe_hf_download(
            repo_id=self.HF_REPO_ID,
            filename=self.SKIN_CKPT,
            local_dir=str(weights_dir),
        )
        print(f"Skin checkpoint downloaded: {self.skin_ckpt_path}")
        
        # Set up symlinks so the default checkpoint paths in configs work
        experiments_dir = Path(self.repo_dir) / "experiments"
        experiments_dir.mkdir(parents=True, exist_ok=True)
        
        # Symlink skeleton checkpoint
        skeleton_exp_dir = experiments_dir / "skeleton" / "articulation-xl_quantization_256"
        skeleton_exp_dir.mkdir(parents=True, exist_ok=True)
        skeleton_ckpt_link = skeleton_exp_dir / "model.ckpt"
        if not skeleton_ckpt_link.exists() and not skeleton_ckpt_link.is_symlink():
            skeleton_ckpt_link.symlink_to(self.skeleton_ckpt_path)
        
        # Symlink skin checkpoint
        skin_exp_dir = experiments_dir / "skin" / "articulation-xl"
        skin_exp_dir.mkdir(parents=True, exist_ok=True)
        skin_ckpt_link = skin_exp_dir / "model.ckpt"
        if not skin_ckpt_link.exists() and not skin_ckpt_link.is_symlink():
            skin_ckpt_link.symlink_to(self.skin_ckpt_path)
        
        # Configure PyTorch
        torch.set_grad_enabled(False)
        torch.set_float32_matmul_precision('high')
        
        # Store config paths for later use
        self.skeleton_task_config = "configs/task/quick_inference_skeleton_articulationxl_ar_256.yaml"
        self.skin_task_config = "configs/task/quick_inference_unirig_skin.yaml"
        
        # Verify configs exist
        os.chdir(self.repo_dir)
        if not os.path.exists(self.skeleton_task_config):
            raise RuntimeError(f"Skeleton config not found: {self.skeleton_task_config}")
        if not os.path.exists(self.skin_task_config):
            raise RuntimeError(f"Skin config not found: {self.skin_task_config}")
        
        # Warm up by importing necessary modules
        print("Importing UniRig modules...")
        self._import_modules()
        
        print("UniRig setup complete!")

    def _import_modules(self) -> None:
        """Import UniRig modules to warm up the system."""
        os.chdir(self.repo_dir)
        try:
            from src.data.extract import clean_bpy, load, process_mesh, get_arranged_bones, process_armature, save_raw_data
            from src.inference.merge import transfer, clean_bpy as merge_clean_bpy
            import bpy
            # Clear any existing scene
            clean_bpy()
        except ImportError as e:
            print(f"Warning: Could not import some modules: {e}")

    def _download_input_file(self, url: str, work_dir: Path) -> Path:
        """Download input file from URL."""
        import requests
        
        # Determine filename from URL
        url_path = url.split("?")[0]  # Remove query params
        filename = os.path.basename(url_path)
        
        if not filename or "." not in filename:
            # Generate a filename if URL doesn't have one
            filename = "input_mesh.glb"
        
        # Validate format
        ext = filename.split(".")[-1].lower()
        if ext not in SUPPORTED_FORMATS:
            raise FieldException(
                "mesh_file",
                f"Unsupported format: .{ext}. Supported formats: {', '.join(SUPPORTED_FORMATS)}",
            )
        
        local_path = work_dir / filename
        
        # Download the file
        response = requests.get(url, stream=True, timeout=300)
        response.raise_for_status()
        
        with open(local_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        if not local_path.exists() or local_path.stat().st_size == 0:
            raise FieldException("mesh_file", "Failed to download mesh file")
        
        return local_path

    def _extract_mesh(
        self,
        input_file: Path,
        output_dir: Path,
        faces_target_count: int,
    ) -> Path:
        """Extract and simplify mesh from input file using Blender."""
        os.chdir(self.repo_dir)
        
        from src.data.extract import clean_bpy, load, process_mesh, get_arranged_bones, process_armature, save_raw_data
        
        clean_bpy()
        
        try:
            armature = load(str(input_file))
        except Exception as e:
            raise FieldException("mesh_file", f"Failed to load mesh: {e}")
        
        if armature is not None:
            arranged_bones = get_arranged_bones(armature)
        else:
            arranged_bones = None
            
        vertices, faces, skin = process_mesh(arranged_bones)
        
        if armature is not None:
            joints, tails, parents, names, matrix_local = process_armature(armature, arranged_bones)
        else:
            joints, tails, parents, names, matrix_local = None, None, None, None, None
        
        output_dir.mkdir(parents=True, exist_ok=True)
        npz_path = output_dir / "raw_data.npz"
        
        save_raw_data(
            path=str(npz_path),
            vertices=vertices,
            faces=faces - 1,  # Blender uses 1-based indexing
            skin=skin,
            joints=joints,
            tails=tails,
            parents=parents,
            names=names,
            matrix_local=matrix_local,
            target_count=faces_target_count,
        )
        
        return output_dir

    def _run_skeleton_prediction(
        self,
        input_file: Path,
        npz_dir: Path,
        output_file: Path,
        seed: int,
    ) -> Path:
        """Run skeleton prediction using the AR model."""
        os.chdir(self.repo_dir)
        
        # Build command to run skeleton prediction
        cmd = [
            sys.executable, "run.py",
            f"--task={self.skeleton_task_config}",
            f"--seed={seed}",
            f"--input={input_file}",
            f"--npz_dir={npz_dir}",
            f"--output={output_file}",
        ]
        
        env = os.environ.copy()
        env["PYTHONPATH"] = self.repo_dir + ":" + env.get("PYTHONPATH", "")
        
        result = subprocess.run(
            cmd,
            cwd=self.repo_dir,
            env=env,
            capture_output=True,
            text=True,
            timeout=600,
        )
        
        if result.returncode != 0:
            error_msg = result.stderr or result.stdout or "Unknown error"
            raise RuntimeError(f"Skeleton prediction failed: {error_msg[:500]}")
        
        # Look for output file in various locations
        if output_file.exists():
            return output_file
        
        # Check npz_dir for skeleton output
        possible_outputs = list(npz_dir.rglob("*skeleton*.fbx"))
        if possible_outputs:
            return possible_outputs[0]
        
        raise RuntimeError("Skeleton prediction did not produce output file")

    def _run_skin_prediction(
        self,
        input_file: Path,
        npz_dir: Path,
        output_file: Path,
    ) -> Path:
        """Run skin prediction using the skin model."""
        os.chdir(self.repo_dir)
        
        cmd = [
            sys.executable, "run.py",
            f"--task={self.skin_task_config}",
            f"--input={input_file}",
            f"--npz_dir={npz_dir}",
            f"--output={output_file}",
            "--data_name=predict_skeleton.npz",
        ]
        
        env = os.environ.copy()
        env["PYTHONPATH"] = self.repo_dir + ":" + env.get("PYTHONPATH", "")
        
        result = subprocess.run(
            cmd,
            cwd=self.repo_dir,
            env=env,
            capture_output=True,
            text=True,
            timeout=600,
        )
        
        if result.returncode != 0:
            error_msg = result.stderr or result.stdout or "Unknown error"
            raise RuntimeError(f"Skin prediction failed: {error_msg[:500]}")
        
        if output_file.exists():
            return output_file
        
        # Check for alternative output locations
        possible_outputs = (
            list(npz_dir.rglob("*result_fbx*.fbx")) + 
            list(npz_dir.rglob("*skin*.fbx"))
        )
        if possible_outputs:
            return possible_outputs[0]
        
        raise RuntimeError("Skin prediction did not produce output file")

    def _merge_with_original(
        self,
        source_file: Path,
        target_file: Path,
        output_file: Path,
    ) -> Path:
        """Merge predicted skeleton/skin with original mesh to preserve textures."""
        os.chdir(self.repo_dir)
        
        from src.inference.merge import transfer
        
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        transfer(
            source=str(source_file),
            target=str(target_file),
            output=str(output_file),
            add_root=False,
        )
        
        if not output_file.exists():
            raise RuntimeError("Merge operation did not produce output file")
        
        return output_file

    def _convert_to_glb(self, fbx_file: Path) -> Path:
        """Convert FBX to GLB format using Blender."""
        import bpy
        from src.data.extract import clean_bpy
        
        clean_bpy()
        
        # Import FBX
        bpy.ops.import_scene.fbx(filepath=str(fbx_file), ignore_leaf_bones=False)
        
        # Export as GLB
        glb_file = fbx_file.with_suffix(".glb")
        bpy.ops.export_scene.gltf(filepath=str(glb_file))
        
        return glb_file

    def _run_full_pipeline(
        self,
        input_file: Path,
        work_dir: Path,
        seed: int,
        faces_target_count: int,
        output_format: str,
    ) -> tuple[Path, Path]:
        """Run the complete skeleton + skin pipeline."""
        
        npz_dir = work_dir / "npz"
        output_dir = work_dir / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        model_name = input_file.stem
        model_npz_dir = npz_dir / model_name
        
        # Step 1: Extract mesh to NPZ format
        self._extract_mesh(input_file, model_npz_dir, faces_target_count)
        
        # Step 2: Run skeleton prediction
        skeleton_output = output_dir / "skeleton.fbx"
        skeleton_file = self._run_skeleton_prediction(
            input_file=input_file,
            npz_dir=npz_dir,
            output_file=skeleton_output,
            seed=seed,
        )
        
        # Step 3: Run skin prediction
        skinned_output = output_dir / "skinned.fbx"
        skinned_file = self._run_skin_prediction(
            input_file=skeleton_file,
            npz_dir=npz_dir,
            output_file=skinned_output,
        )
        
        # Step 4: Merge with original mesh to preserve textures/materials
        merged_fbx = output_dir / "rigged.fbx"
        final_fbx = self._merge_with_original(
            source_file=skinned_file,
            target_file=input_file,
            output_file=merged_fbx,
        )
        
        # Step 5: Convert format if needed
        if output_format == "glb":
            final_output = self._convert_to_glb(final_fbx)
        else:
            final_output = final_fbx
        
        return final_output, skeleton_file

    @fal.endpoint("/")
    def generate(
        self,
        input: UniRigInput,
        request: Request,
        response: Response,
    ) -> UniRigOutput:
        """
        Generate a fully rigged 3D model with skeleton and skinning weights.
        
        This is the main endpoint that performs the complete rigging pipeline:
        1. Mesh extraction and simplification
        2. Skeleton prediction using the autoregressive model
        3. Skinning weight prediction using bone-point cross attention
        4. Merging with original mesh to preserve textures/materials
        
        The generated skeleton is optimized for the mesh geometry and includes
        proper bone hierarchy and naming. Skinning weights are automatically
        computed for smooth deformation.
        """
        seed = get_seed(input.seed)
        
        # Create temporary working directory
        work_dir = Path(tempfile.mkdtemp(prefix="unirig_"))
        
        try:
            # Download input file
            input_file = self._download_input_file(input.mesh_file, work_dir)
            
            # Run full pipeline
            rigged_file, skeleton_file = self._run_full_pipeline(
                input_file=input_file,
                work_dir=work_dir,
                seed=seed,
                faces_target_count=input.faces_target_count,
                output_format=input.output_format,
            )
            
            response.headers["x-fal-billable-units"] = "1"
            
            return UniRigOutput(
                rigged_file=File.from_path(str(rigged_file), request=request),
                skeleton_file=File.from_path(str(skeleton_file), request=request) if skeleton_file.exists() else None,
                seed=seed,
            )
            
        finally:
            # Cleanup temporary files
            shutil.rmtree(work_dir, ignore_errors=True)

    @fal.endpoint("/skeleton")
    def generate_skeleton(
        self,
        input: SkeletonOnlyInput,
        request: Request,
        response: Response,
    ) -> SkeletonOutput:
        """
        Generate skeleton only (without skinning weights).
        
        Use this endpoint when you want to:
        - Preview or manually edit the skeleton before skinning
        - Use your own skinning solution
        - Generate multiple skeleton variations with different seeds
        - Inspect the predicted bone structure
        
        The output skeleton can be used as input to the /skin endpoint
        after any desired manual adjustments.
        """
        seed = get_seed(input.seed)
        
        work_dir = Path(tempfile.mkdtemp(prefix="unirig_skeleton_"))
        
        try:
            # Download input file
            input_file = self._download_input_file(input.mesh_file, work_dir)
            
            npz_dir = work_dir / "npz"
            output_dir = work_dir / "output"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            model_npz_dir = npz_dir / input_file.stem
            
            # Extract mesh
            self._extract_mesh(input_file, model_npz_dir, input.faces_target_count)
            
            # Run skeleton prediction
            skeleton_output = output_dir / "skeleton.fbx"
            skeleton_file = self._run_skeleton_prediction(
                input_file=input_file,
                npz_dir=npz_dir,
                output_file=skeleton_output,
                seed=seed,
            )
            
            # Convert format if needed
            if input.output_format == "glb":
                skeleton_file = self._convert_to_glb(skeleton_file)
            
            response.headers["x-fal-billable-units"] = "1"
            
            return SkeletonOutput(
                skeleton_file=File.from_path(str(skeleton_file), request=request),
                seed=seed,
            )
            
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)

    @fal.endpoint("/skin")
    def generate_skin(
        self,
        input: SkinOnlyInput,
        request: Request,
        response: Response,
    ) -> SkinOutput:
        """
        Generate skinning weights for a mesh with existing skeleton.
        
        Use this endpoint when you:
        - Have manually edited a skeleton from /skeleton endpoint
        - Want to apply UniRig's skinning to your own skeleton
        - Need to re-skin an existing rigged model with different weights
        
        Input requirements:
        - mesh_file: Must be a file containing both mesh AND skeleton (e.g., FBX with armature)
        - original_mesh_file (optional): Original mesh to merge textures/materials from
        
        The skinning algorithm uses bone-point cross attention to compute
        smooth deformation weights based on mesh geometry and bone positions.
        """
        work_dir = Path(tempfile.mkdtemp(prefix="unirig_skin_"))
        
        try:
            # Download input file (mesh with skeleton)
            input_file = self._download_input_file(input.mesh_file, work_dir)
            
            # Download original mesh if provided (for texture/material merging)
            if input.original_mesh_file:
                original_file = self._download_input_file(input.original_mesh_file, work_dir / "original")
            else:
                original_file = input_file
            
            npz_dir = work_dir / "npz"
            output_dir = work_dir / "output"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            model_npz_dir = npz_dir / input_file.stem
            
            # Extract mesh with skeleton
            self._extract_mesh(input_file, model_npz_dir, input.faces_target_count)
            
            # Run skin prediction
            skinned_output = output_dir / "skinned.fbx"
            skinned_file = self._run_skin_prediction(
                input_file=input_file,
                npz_dir=npz_dir,
                output_file=skinned_output,
            )
            
            # Optionally merge with original to preserve textures
            if input.original_mesh_file and original_file != input_file:
                merged_output = output_dir / "merged.fbx"
                skinned_file = self._merge_with_original(
                    source_file=skinned_file,
                    target_file=original_file,
                    output_file=merged_output,
                )
            
            # Convert format if needed
            if input.output_format == "glb":
                skinned_file = self._convert_to_glb(skinned_file)
            
            response.headers["x-fal-billable-units"] = "1"
            
            return SkinOutput(
                skinned_file=File.from_path(str(skinned_file), request=request),
            )
            
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)

    @fal.endpoint("/health")
    def health_check(
        self,
        request: Request,
        response: Response,
    ) -> HealthOutput:
        """
        Health check endpoint.
        
        Returns the service status, version information, and GPU availability.
        Use this to verify the service is ready to accept requests.
        """
        import torch
        
        return HealthOutput(
            status="healthy",
            version="UniRig v1.0 (Articulation-XL)",
            gpu_available=torch.cuda.is_available(),
        )


if __name__ == "__main__":
    app = fal.wrap_app(UniRig)
    app()
