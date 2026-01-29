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
import sys
import tempfile
from pathlib import Path
from typing import Literal, Dict, Any, Optional

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


class DictToAttr(dict):
    """A dict subclass that allows attribute-style access to keys."""
    def __getattr__(self, key):
        try:
            value = self[key]
            if isinstance(value, dict) and not isinstance(value, DictToAttr):
                value = DictToAttr(value)
                self[key] = value
            return value
        except KeyError:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{key}'")
    
    def __setattr__(self, key, value):
        self[key] = value
    
    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{key}'")


def dict_to_attr(d):
    """Recursively convert a dict to DictToAttr for attribute-style access."""
    if isinstance(d, dict):
        return DictToAttr({k: dict_to_attr(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [dict_to_attr(item) for item in d]
    return d


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
                wait_time = 4 * (2 ** attempt)
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
        description="Target number of faces for mesh simplification.",
    )


class SkeletonOnlyInput(BaseModel):
    """Input schema for skeleton-only prediction."""
    
    mesh_file: str = Field(
        description="URL to the input 3D mesh file (supports .obj, .fbx, .glb, .gltf, .vrm, .dae)",
    )
    seed: int | None = Field(
        default=None,
        description="Random seed for reproducible skeleton generation.",
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
    )
    original_mesh_file: str | None = Field(
        default=None,
        description="URL to the original mesh file to merge textures/materials from.",
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
    )


class SkeletonOutput(BaseModel):
    """Output schema for skeleton-only prediction."""
    
    skeleton_file: File = Field(
        description="The predicted skeleton file.",
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
        "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl",
        # 3D processing
        "trimesh>=4.0.0",
        "open3d>=0.18.0",
        "fast-simplification>=0.1.0",
        "pyrender>=0.1.45",
        # Blender Python API
        "bpy==4.2",
        # Other dependencies
        "python-box>=7.0.0",  # Required by UniRig internals
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
        "https://data.pyg.org/whl/torch-2.5.0%2Bcu124/torch_cluster-1.6.3%2Bpt25cu124-cp311-cp311-linux_x86_64.whl",
        "https://data.pyg.org/whl/torch-2.5.0%2Bcu124/torch_scatter-2.1.2%2Bpt25cu124-cp311-cp311-linux_x86_64.whl",
        # Spconv for sparse convolutions
        "spconv-cu124>=2.3.0",
        # Extra index URLs
        "--extra-index-url",
        "https://download.pytorch.org/whl/cu124",
    ]

    GITHUB_REPO = "VAST-AI-Research/UniRig"
    GITHUB_COMMIT = "main"
    HF_REPO_ID = "VAST-AI/UniRig"
    
    SKELETON_CKPT = "skeleton/articulation-xl_quantization_256/model.ckpt"
    SKIN_CKPT = "skin/articulation-xl/model.ckpt"

    async def setup(self) -> None:
        """Initialize the UniRig models and environment."""
        import torch
        import yaml
        
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
        
        os.chdir(self.repo_dir)
        
        # Download model checkpoints
        print("Downloading model checkpoints from HuggingFace...")
        weights_dir = Path(FAL_MODEL_WEIGHTS_DIR) / "unirig"
        weights_dir.mkdir(parents=True, exist_ok=True)
        
        self.skeleton_ckpt_path = safe_hf_download(
            repo_id=self.HF_REPO_ID,
            filename=self.SKELETON_CKPT,
            local_dir=str(weights_dir),
        )
        print(f"Skeleton checkpoint: {self.skeleton_ckpt_path}")
        
        self.skin_ckpt_path = safe_hf_download(
            repo_id=self.HF_REPO_ID,
            filename=self.SKIN_CKPT,
            local_dir=str(weights_dir),
        )
        print(f"Skin checkpoint: {self.skin_ckpt_path}")
        
        # Configure PyTorch
        torch.set_grad_enabled(False)
        torch.set_float32_matmul_precision('high')
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load configurations
        print("Loading model configurations...")
        self._load_configs()
        
        # Load models
        print("Loading skeleton model...")
        self._load_skeleton_model()
        
        print("Loading skin model...")
        self._load_skin_model()
        
        print("UniRig setup complete!")

    def _load_yaml(self, path: str) -> dict:
        """Load a YAML config file."""
        import yaml
        full_path = os.path.join(self.repo_dir, path)
        return yaml.safe_load(open(full_path, 'r'))

    def _load_configs(self) -> None:
        """Load all required configurations."""
        from src.tokenizer.spec import TokenizerConfig
        from src.data.transform import TransformConfig
        from src.data.order import OrderConfig
        
        # Skeleton model configs
        # Use dict_to_attr for attribute-style access that UniRig parsers expect
        self.skeleton_tokenizer_config = TokenizerConfig.parse(
            dict_to_attr(self._load_yaml("configs/tokenizer/tokenizer_parts_articulationxl_256.yaml"))
        )
        ar_transform_yaml = self._load_yaml("configs/transform/inference_ar_transform.yaml")
        self.skeleton_transform_config = TransformConfig.parse(
            dict_to_attr(ar_transform_yaml.get('predict_transform_config', {}))
        )
        self.skeleton_model_config = self._load_yaml("configs/model/unirig_ar_350m_1024_81920_float32.yaml")
        self.skeleton_system_config = self._load_yaml("configs/system/ar_inference_articulationxl.yaml")
        
        # Skin model configs
        skin_transform_yaml = self._load_yaml("configs/transform/inference_skin_transform.yaml")
        self.skin_transform_config = TransformConfig.parse(
            dict_to_attr(skin_transform_yaml.get('predict_transform_config', {}))
        )
        self.skin_model_config = self._load_yaml("configs/model/unirig_skin.yaml")

    def _load_skeleton_model(self) -> None:
        """Load the skeleton prediction model."""
        import torch
        from src.tokenizer.parse import get_tokenizer
        from src.model.parse import get_model
        from src.system.ar import ARSystem
        from src.data.order import get_order
        
        # Create tokenizer
        self.skeleton_tokenizer = get_tokenizer(config=self.skeleton_tokenizer_config)
        
        # Create model - convert nested dicts to attr-accessible objects
        model_config = dict_to_attr(self.skeleton_model_config)
        model_kwargs = dict(model_config)
        model_kwargs['tokenizer'] = self.skeleton_tokenizer
        self.skeleton_model = get_model(**model_kwargs)
        
        # Load checkpoint
        checkpoint = torch.load(self.skeleton_ckpt_path, map_location=self.device)
        
        # Extract model state dict from Lightning checkpoint
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            # Remove 'model.' prefix if present
            state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
        else:
            state_dict = checkpoint
        
        self.skeleton_model.load_state_dict(state_dict, strict=False)
        self.skeleton_model.to(self.device)
        self.skeleton_model.eval()
        
        # Get generation kwargs from system config
        self.skeleton_generate_kwargs = dict(self.skeleton_system_config.get('generate_kwargs', {}))
        
        # Get order for name generation
        if self.skeleton_transform_config.order_config is not None:
            self.skeleton_order = get_order(config=self.skeleton_transform_config.order_config)
        else:
            self.skeleton_order = None
        
        print(f"Skeleton model loaded on {self.device}")

    def _load_skin_model(self) -> None:
        """Load the skin prediction model."""
        import torch
        from src.model.parse import get_model
        
        # Create model - convert nested dicts to attr-accessible objects
        model_config = dict_to_attr(self.skin_model_config)
        model_kwargs = dict(model_config)
        self.skin_model = get_model(**model_kwargs)
        
        # Load checkpoint
        checkpoint = torch.load(self.skin_ckpt_path, map_location=self.device)
        
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
        else:
            state_dict = checkpoint
        
        self.skin_model.load_state_dict(state_dict, strict=False)
        self.skin_model.to(self.device)
        self.skin_model.eval()
        
        print(f"Skin model loaded on {self.device}")

    def _download_input_file(self, url: str, work_dir: Path) -> Path:
        """Download input file from URL."""
        import requests
        
        url_path = url.split("?")[0]
        filename = os.path.basename(url_path)
        
        if not filename or "." not in filename:
            filename = "input_mesh.glb"
        
        ext = filename.split(".")[-1].lower()
        if ext not in SUPPORTED_FORMATS:
            raise FieldException(
                "mesh_file",
                f"Unsupported format: .{ext}. Supported: {', '.join(SUPPORTED_FORMATS)}",
            )
        
        local_path = work_dir / filename
        
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
        faces_target_count: int,
    ) -> 'RawData':
        """Extract and process mesh from input file."""
        os.chdir(self.repo_dir)
        
        from src.data.extract import clean_bpy, load, process_mesh, get_arranged_bones, process_armature
        from src.data.raw_data import RawData
        import numpy as np
        import trimesh
        import fast_simplification
        
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
        
        # Simplify mesh if needed
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces - 1)  # 1-indexed to 0-indexed
        vertices = np.array(mesh.vertices, dtype=np.float32)
        faces = np.array(mesh.faces, dtype=np.int64)
        
        if faces.shape[0] > faces_target_count:
            vertices, faces = fast_simplification.simplify(vertices, faces, target_count=faces_target_count)
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        
        raw_data = RawData(
            vertices=np.array(mesh.vertices, dtype=np.float32),
            vertex_normals=np.array(mesh.vertex_normals, dtype=np.float32),
            faces=np.array(mesh.faces, dtype=np.int64),
            face_normals=np.array(mesh.face_normals, dtype=np.float32),
            joints=np.array(joints, dtype=np.float32) if joints is not None else None,
            tails=tails,
            skin=np.array(skin, dtype=np.float32) if skin is not None else None,
            no_skin=None,
            parents=parents,
            names=names,
            matrix_local=matrix_local,
        )
        
        return raw_data

    def _predict_skeleton(
        self,
        raw_data: 'RawData',
        seed: int,
    ) -> 'RawData':
        """Run skeleton prediction on the mesh."""
        import torch
        import numpy as np
        import lightning as L
        from src.data.asset import Asset
        from src.data.transform import transform_asset
        from src.data.raw_data import RawData, RawSkeleton
        
        L.seed_everything(seed, workers=True)
        
        # Create asset from raw data
        asset = Asset.from_raw_data(raw_data=raw_data, tokenizer=self.skeleton_tokenizer)
        
        # Apply transforms
        transform_asset(asset=asset, transform_config=self.skeleton_transform_config)
        
        # Prepare input tensors
        vertices = torch.from_numpy(asset.sampled_vertices).float().to(self.device)
        normals = torch.from_numpy(asset.sampled_normals).float().to(self.device)
        
        # Run inference
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
            result = self.skeleton_model.generate(
                vertices=vertices,
                normals=normals,
                cls=None,
                **self.skeleton_generate_kwargs,
            )
        
        # Create skeleton from result
        skeleton = RawSkeleton.from_detokenize_output(res=result, order=self.skeleton_order)
        
        # Create new RawData with skeleton
        skeleton_data = RawData(
            vertices=raw_data.vertices,
            vertex_normals=raw_data.vertex_normals,
            faces=raw_data.faces,
            face_normals=raw_data.face_normals,
            joints=skeleton.joints,
            tails=skeleton.tails,
            skin=None,
            no_skin=skeleton.no_skin,
            parents=skeleton.parents,
            names=skeleton.names,
            matrix_local=None,
            cls=result.cls,
        )
        
        return skeleton_data

    def _predict_skin(
        self,
        raw_data: 'RawData',
    ) -> 'RawData':
        """Run skin prediction on mesh with skeleton."""
        import torch
        import numpy as np
        from src.data.asset import Asset
        from src.data.transform import transform_asset
        from src.data.raw_data import RawData, RawSkin
        
        # Create asset
        asset = Asset.from_raw_data(raw_data=raw_data, tokenizer=None)
        
        # Apply transforms
        transform_asset(asset=asset, transform_config=self.skin_transform_config)
        
        # Prepare batch
        batch = {
            'vertices': torch.from_numpy(asset.sampled_vertices).float().unsqueeze(0).to(self.device),
            'normals': torch.from_numpy(asset.sampled_normals).float().unsqueeze(0).to(self.device),
            'joints': torch.from_numpy(raw_data.joints).float().unsqueeze(0).to(self.device),
            'tails': torch.from_numpy(raw_data.tails).float().unsqueeze(0).to(self.device),
            'num_bones': torch.tensor([len(raw_data.joints)]).to(self.device),
            'path': ['inference'],
            'cls': [raw_data.cls if hasattr(raw_data, 'cls') and raw_data.cls else 'unknown'],
        }
        
        # Add vertex groups if present
        if hasattr(asset, 'vertex_groups') and asset.vertex_groups:
            for key, value in asset.vertex_groups.items():
                if isinstance(value, np.ndarray):
                    batch[key] = torch.from_numpy(value).float().unsqueeze(0).to(self.device)
        
        # Run inference
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
            result = self.skin_model.predict_step(batch)
        
        # Extract skin weights
        if isinstance(result, dict) and 'skin' in result:
            skin = result['skin'].cpu().numpy()[0]
        elif isinstance(result, np.ndarray):
            skin = result
        else:
            skin = result[0] if isinstance(result, (list, tuple)) else result
            if hasattr(skin, 'cpu'):
                skin = skin.cpu().numpy()
        
        # Create output with skin
        skinned_data = RawData(
            vertices=raw_data.vertices,
            vertex_normals=raw_data.vertex_normals,
            faces=raw_data.faces,
            face_normals=raw_data.face_normals,
            joints=raw_data.joints,
            tails=raw_data.tails,
            skin=skin,
            no_skin=raw_data.no_skin if hasattr(raw_data, 'no_skin') else None,
            parents=raw_data.parents,
            names=raw_data.names,
            matrix_local=raw_data.matrix_local if hasattr(raw_data, 'matrix_local') else None,
        )
        
        return skinned_data

    def _export_fbx(self, raw_data: 'RawData', output_path: Path) -> Path:
        """Export RawData to FBX file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        raw_data.export_fbx(path=str(output_path), use_tail=True)
        return output_path

    def _convert_to_glb(self, fbx_file: Path) -> Path:
        """Convert FBX to GLB format using Blender."""
        import bpy
        from src.data.extract import clean_bpy
        
        clean_bpy()
        bpy.ops.import_scene.fbx(filepath=str(fbx_file), ignore_leaf_bones=False)
        
        glb_file = fbx_file.with_suffix(".glb")
        bpy.ops.export_scene.gltf(filepath=str(glb_file))
        
        return glb_file

    def _merge_with_original(
        self,
        skinned_data: 'RawData',
        original_file: Path,
        output_path: Path,
    ) -> Path:
        """Merge skeleton/skin with original mesh to preserve textures."""
        os.chdir(self.repo_dir)
        from src.inference.merge import merge
        
        merge(
            path=str(original_file),
            output_path=str(output_path),
            vertices=skinned_data.vertices,
            joints=skinned_data.joints,
            skin=skinned_data.skin,
            parents=skinned_data.parents,
            names=skinned_data.names,
            tails=skinned_data.tails,
        )
        
        return output_path

    @fal.endpoint("/")
    def generate(
        self,
        input: UniRigInput,
        request: Request,
        response: Response,
    ) -> UniRigOutput:
        """
        Generate a fully rigged 3D model with skeleton and skinning weights.
        """
        seed = get_seed(input.seed)
        work_dir = Path(tempfile.mkdtemp(prefix="unirig_"))
        
        try:
            os.chdir(self.repo_dir)
            
            # Download input file
            input_file = self._download_input_file(input.mesh_file, work_dir)
            
            # Extract mesh
            raw_data = self._extract_mesh(input_file, input.faces_target_count)
            
            # Predict skeleton
            skeleton_data = self._predict_skeleton(raw_data, seed)
            
            # Export skeleton
            skeleton_output = work_dir / "skeleton.fbx"
            self._export_fbx(skeleton_data, skeleton_output)
            
            # Predict skin
            skinned_data = self._predict_skin(skeleton_data)
            
            # Merge with original mesh
            rigged_output = work_dir / "rigged.fbx"
            self._merge_with_original(skinned_data, input_file, rigged_output)
            
            # Convert format if needed
            if input.output_format == "glb":
                rigged_output = self._convert_to_glb(rigged_output)
                if skeleton_output.exists():
                    skeleton_output = self._convert_to_glb(skeleton_output)
            
            response.headers["x-fal-billable-units"] = "1"
            
            return UniRigOutput(
                rigged_file=File.from_path(str(rigged_output), request=request),
                skeleton_file=File.from_path(str(skeleton_output), request=request) if skeleton_output.exists() else None,
                seed=seed,
            )
            
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)

    @fal.endpoint("/skeleton")
    def generate_skeleton(
        self,
        input: SkeletonOnlyInput,
        request: Request,
        response: Response,
    ) -> SkeletonOutput:
        """Generate skeleton only (without skinning weights)."""
        seed = get_seed(input.seed)
        work_dir = Path(tempfile.mkdtemp(prefix="unirig_skeleton_"))
        
        try:
            os.chdir(self.repo_dir)
            
            input_file = self._download_input_file(input.mesh_file, work_dir)
            raw_data = self._extract_mesh(input_file, input.faces_target_count)
            skeleton_data = self._predict_skeleton(raw_data, seed)
            
            skeleton_output = work_dir / "skeleton.fbx"
            self._export_fbx(skeleton_data, skeleton_output)
            
            if input.output_format == "glb":
                skeleton_output = self._convert_to_glb(skeleton_output)
            
            response.headers["x-fal-billable-units"] = "1"
            
            return SkeletonOutput(
                skeleton_file=File.from_path(str(skeleton_output), request=request),
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
        """Generate skinning weights for a mesh with existing skeleton."""
        work_dir = Path(tempfile.mkdtemp(prefix="unirig_skin_"))
        
        try:
            os.chdir(self.repo_dir)
            
            input_file = self._download_input_file(input.mesh_file, work_dir)
            
            if input.original_mesh_file:
                original_dir = work_dir / "original"
                original_dir.mkdir(parents=True, exist_ok=True)
                original_file = self._download_input_file(input.original_mesh_file, original_dir)
            else:
                original_file = input_file
            
            raw_data = self._extract_mesh(input_file, input.faces_target_count)
            
            if raw_data.joints is None:
                raise FieldException("mesh_file", "Input file must contain a skeleton for skin prediction")
            
            skinned_data = self._predict_skin(raw_data)
            
            skinned_output = work_dir / "skinned.fbx"
            if input.original_mesh_file:
                self._merge_with_original(skinned_data, original_file, skinned_output)
            else:
                self._export_fbx(skinned_data, skinned_output)
            
            if input.output_format == "glb":
                skinned_output = self._convert_to_glb(skinned_output)
            
            response.headers["x-fal-billable-units"] = "1"
            
            return SkinOutput(
                skinned_file=File.from_path(str(skinned_output), request=request),
            )
            
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)

    @fal.endpoint("/health")
    def health_check(
        self,
        request: Request,
        response: Response,
    ) -> HealthOutput:
        """Health check endpoint."""
        import torch
        
        return HealthOutput(
            status="healthy",
            version="UniRig v1.0 (Articulation-XL)",
            gpu_available=torch.cuda.is_available(),
        )


if __name__ == "__main__":
    app = fal.wrap_app(UniRig)
    app()
