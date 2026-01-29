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
from typing import Literal

import fal
from fal.exceptions import FieldException
from fal.toolkit import FAL_MODEL_WEIGHTS_DIR, File, clone_repository
from fastapi import Request, Response
from pydantic import BaseModel, Field


OUTPUT_FORMAT_LITERAL = Literal["fbx", "glb"]
DEFAULT_OUTPUT_FORMAT: OUTPUT_FORMAT_LITERAL = "fbx"
DEFAULT_SEED: int = 12345
DEFAULT_FACES_TARGET_COUNT: int = 50000
SUPPORTED_FORMATS = ["obj", "fbx", "glb", "gltf", "vrm", "dae"]


def get_seed(seed: int | None) -> int:
    if seed is None:
        import random
        return random.randint(0, 2**32 - 1)
    return seed


class DictToAttr(dict):
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


class UniRigOutput(BaseModel):
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
        "torch==2.5.1+cu124",
        "torchvision==0.20.1+cu124",
        "transformers==4.51.3",
        "huggingface_hub>=0.20.0",
        "lightning>=2.0.0",
        "pytorch_lightning>=2.0.0",
        "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl",
        "trimesh>=4.0.0",
        "open3d>=0.18.0",
        "fast-simplification>=0.1.0",
        "bpy==4.2",
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
        "https://data.pyg.org/whl/torch-2.5.0%2Bcu124/torch_cluster-1.6.3%2Bpt25cu124-cp311-cp311-linux_x86_64.whl",
        "https://data.pyg.org/whl/torch-2.5.0%2Bcu124/torch_scatter-2.1.2%2Bpt25cu124-cp311-cp311-linux_x86_64.whl",
        "spconv-cu124>=2.3.0",
        "--extra-index-url",
        "https://download.pytorch.org/whl/cu124",
    ]

    GITHUB_REPO = "VAST-AI-Research/UniRig"
    GITHUB_COMMIT = "main"
    HF_REPO_ID = "VAST-AI/UniRig"
    
    SKELETON_CKPT = "skeleton/articulation-xl_quantization_256/model.ckpt"
    SKIN_CKPT = "skin/articulation-xl/model.ckpt"

    async def setup(self) -> None:
        import torch
        
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
        
        self.repo_dir = str(
            clone_repository(
                f"https://github.com/{self.GITHUB_REPO}.git",
                commit_hash=self.GITHUB_COMMIT,
                include_to_path=True,
                repo_name="unirig",
            )
        )
        
        if self.repo_dir not in sys.path:
            sys.path.insert(0, self.repo_dir)
        
        os.chdir(self.repo_dir)
        self._patch_unirig_code()
        
        weights_dir = Path(FAL_MODEL_WEIGHTS_DIR) / "unirig"
        weights_dir.mkdir(parents=True, exist_ok=True)
        
        self.skeleton_ckpt_path = safe_hf_download(
            repo_id=self.HF_REPO_ID,
            filename=self.SKELETON_CKPT,
            local_dir=str(weights_dir),
        )
        
        self.skin_ckpt_path = safe_hf_download(
            repo_id=self.HF_REPO_ID,
            filename=self.SKIN_CKPT,
            local_dir=str(weights_dir),
        )
        
        torch.set_grad_enabled(False)
        torch.set_float32_matmul_precision('high')
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self._load_configs()
        self._load_skeleton_model()
        self._load_skin_model()

    def _patch_unirig_code(self) -> None:
        skin_model_path = os.path.join(self.repo_dir, "src/model/unirig_skin.py")
        with open(skin_model_path, 'r') as f:
            content = f.read()
        
        old_line = "'offset': torch.tensor(batch['offset']),"
        new_line = "'offset': torch.tensor(batch['offset']).to(vertices.device),"
        
        if old_line in content:
            content = content.replace(old_line, new_line)
            with open(skin_model_path, 'w') as f:
                f.write(content)

    def _load_yaml(self, path: str) -> dict:
        import yaml
        full_path = os.path.join(self.repo_dir, path)
        with open(full_path, 'r') as f:
            return yaml.safe_load(f)

    def _load_configs(self) -> None:
        from src.tokenizer.spec import TokenizerConfig
        from src.data.transform import TransformConfig
        
        self.skeleton_tokenizer_config = TokenizerConfig.parse(
            dict_to_attr(self._load_yaml("configs/tokenizer/tokenizer_parts_articulationxl_256.yaml"))
        )
        ar_transform_yaml = self._load_yaml("configs/transform/inference_ar_transform.yaml")
        self.skeleton_transform_config = TransformConfig.parse(
            dict_to_attr(ar_transform_yaml.get('predict_transform_config', {}))
        )
        self.skeleton_model_config = self._load_yaml("configs/model/unirig_ar_350m_1024_81920_float32.yaml")
        self.skeleton_system_config = self._load_yaml("configs/system/ar_inference_articulationxl.yaml")
        
        skin_transform_yaml = self._load_yaml("configs/transform/inference_skin_transform.yaml")
        predict_config = skin_transform_yaml.get('predict_transform_config', {})
        vertex_group_config = predict_config.get('vertex_group_config', {})
        voxel_skin_kwargs = vertex_group_config.get('kwargs', {}).get('voxel_skin', {})
        if voxel_skin_kwargs.get('backend') == 'pyrender':
            voxel_skin_kwargs['backend'] = 'open3d'
        
        self.skin_transform_config = TransformConfig.parse(
            dict_to_attr(predict_config)
        )
        self.skin_model_config = self._load_yaml("configs/model/unirig_skin.yaml")

    def _load_skeleton_model(self) -> None:
        import torch
        from src.tokenizer.parse import get_tokenizer
        from src.model.parse import get_model
        from src.data.order import get_order
        
        self.skeleton_tokenizer = get_tokenizer(config=self.skeleton_tokenizer_config)
        
        model_config = dict_to_attr(self.skeleton_model_config)
        model_kwargs = dict(model_config)
        model_kwargs['tokenizer'] = self.skeleton_tokenizer
        self.skeleton_model = get_model(**model_kwargs)
        
        checkpoint = torch.load(self.skeleton_ckpt_path, map_location=self.device)
        
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
        else:
            state_dict = checkpoint
        
        self.skeleton_model.load_state_dict(state_dict, strict=False)
        self.skeleton_model.to(self.device)
        self.skeleton_model.eval()
        
        raw_generate_kwargs = dict(self.skeleton_system_config.get('generate_kwargs', {}))
        keys_to_remove = ['no_cls', 'assign_cls', 'use_dir_cls']
        self.skeleton_generate_kwargs = {k: v for k, v in raw_generate_kwargs.items() if k not in keys_to_remove}
        
        if self.skeleton_transform_config.order_config is not None:
            self.skeleton_order = get_order(config=self.skeleton_transform_config.order_config)
        else:
            self.skeleton_order = None

    def _load_skin_model(self) -> None:
        import torch
        from src.model.parse import get_model
        
        model_config = dict_to_attr(self.skin_model_config)
        model_kwargs = dict(model_config)
        self.skin_model = get_model(**model_kwargs)
        
        checkpoint = torch.load(self.skin_ckpt_path, map_location=self.device)
        
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
        else:
            state_dict = checkpoint
        
        self.skin_model.load_state_dict(state_dict, strict=False)
        self.skin_model.to(self.device)
        self.skin_model.eval()

    def _download_input_file(self, url: str, work_dir: Path) -> Path:
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
        
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces - 1)
        vertices = np.array(mesh.vertices, dtype=np.float32)
        faces_simplified = np.array(mesh.faces, dtype=np.int64)
        
        if faces_simplified.shape[0] > faces_target_count:
            vertices, faces_simplified = fast_simplification.simplify(
                vertices, faces_simplified, target_count=faces_target_count
            )
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces_simplified)
        
        raw_data = RawData(
            vertices=np.array(mesh.vertices, dtype=np.float32),
            vertex_normals=np.array(mesh.vertex_normals, dtype=np.float32),
            faces=np.array(mesh.faces, dtype=np.int64),
            face_normals=np.array(mesh.face_normals, dtype=np.float32),
            joints=np.array(joints, dtype=np.float32) if joints is not None else None,
            tails=np.array(tails, dtype=np.float32) if tails is not None else None,
            skin=np.array(skin, dtype=np.float32) if skin is not None else None,
            no_skin=None,
            parents=parents,
            names=names,
            matrix_local=np.array(matrix_local, dtype=np.float32) if matrix_local is not None else None,
        )
        
        return raw_data

    def _predict_skeleton(
        self,
        raw_data: 'RawData',
        seed: int,
    ) -> 'RawData':
        import torch
        import numpy as np
        import random
        from src.data.asset import Asset
        from src.data.transform import transform_asset
        from src.data.raw_data import RawData, RawSkeleton
        
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        asset = Asset.from_raw_data(
            raw_data=raw_data,
            cls="unknown",
            path="inference",
            data_name="raw_data.npz",
        )
        
        transform_asset(asset=asset, transform_config=self.skeleton_transform_config)
        
        vertices = torch.from_numpy(asset.sampled_vertices).float().to(self.device)
        normals = torch.from_numpy(asset.sampled_normals).float().to(self.device)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
            result = self.skeleton_model.generate(
                vertices=vertices,
                normals=normals,
                cls=None,
                **self.skeleton_generate_kwargs,
            )
        
        skeleton = RawSkeleton.from_detokenize_output(res=result, order=self.skeleton_order)
        
        skeleton_data = RawData(
            vertices=raw_data.vertices,
            vertex_normals=raw_data.vertex_normals,
            faces=raw_data.faces,
            face_normals=raw_data.face_normals,
            joints=np.array(skeleton.joints, dtype=np.float32),
            tails=np.array(skeleton.tails, dtype=np.float32),
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
        import torch
        import numpy as np
        from src.data.asset import Asset
        from src.data.transform import transform_asset
        from src.data.raw_data import RawData
        
        asset = Asset.from_raw_data(
            raw_data=raw_data,
            cls=raw_data.cls if hasattr(raw_data, 'cls') and raw_data.cls else "unknown",
            path="inference",
            data_name="raw_data.npz",
        )
        
        transform_asset(asset=asset, transform_config=self.skin_transform_config)
        
        parents_list = []
        for p in raw_data.parents:
            parents_list.append(p if p is not None else -1)
        parents_tensor = torch.tensor(parents_list, dtype=torch.long).unsqueeze(0).to(self.device)
        
        num_vertices = asset.sampled_vertices.shape[0]
        
        batch = {
            'vertices': torch.from_numpy(asset.sampled_vertices).float().unsqueeze(0).to(self.device),
            'normals': torch.from_numpy(asset.sampled_normals).float().unsqueeze(0).to(self.device),
            'joints': torch.from_numpy(raw_data.joints).float().unsqueeze(0).to(self.device),
            'tails': torch.from_numpy(raw_data.tails).float().unsqueeze(0).to(self.device),
            'parents': parents_tensor,
            'num_bones': torch.tensor([len(raw_data.joints)]).to(self.device),
            'offset': [num_vertices],
            'path': ['inference'],
            'cls': [raw_data.cls if hasattr(raw_data, 'cls') and raw_data.cls else 'unknown'],
        }
        
        if hasattr(asset, 'sampled_vertex_groups') and asset.sampled_vertex_groups:
            if 'voxel_skin' in asset.sampled_vertex_groups:
                voxel_skin = asset.sampled_vertex_groups['voxel_skin']
                batch['voxel_skin'] = torch.from_numpy(voxel_skin).float().unsqueeze(0).to(self.device)
        
        if 'voxel_skin' not in batch:
            num_joints = len(raw_data.joints)
            voxel_skin = np.zeros((num_vertices, num_joints), dtype=np.float32)
            for j in range(num_joints):
                dist = np.linalg.norm(asset.sampled_vertices - raw_data.joints[j], axis=1)
                voxel_skin[:, j] = 1.0 / (dist + 1e-6)
            voxel_skin = voxel_skin / voxel_skin.sum(axis=1, keepdims=True)
            batch['voxel_skin'] = torch.from_numpy(voxel_skin).float().unsqueeze(0).to(self.device)
        
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
            results = self.skin_model.predict_step(batch)
        
        if isinstance(results, list) and len(results) > 0:
            skin = results[0]
            if hasattr(skin, 'cpu'):
                skin = skin.cpu().numpy()
        elif isinstance(results, torch.Tensor):
            skin = results[0].cpu().numpy()
        else:
            skin = np.array(results[0])
        
        if skin.ndim == 1:
            skin = skin.reshape(-1, len(raw_data.joints))
        
        skinned_data = RawData(
            vertices=raw_data.vertices,
            vertex_normals=raw_data.vertex_normals,
            faces=raw_data.faces,
            face_normals=raw_data.face_normals,
            joints=raw_data.joints,
            tails=raw_data.tails,
            skin=skin.astype(np.float32),
            no_skin=raw_data.no_skin if hasattr(raw_data, 'no_skin') else None,
            parents=raw_data.parents,
            names=raw_data.names,
            matrix_local=raw_data.matrix_local if hasattr(raw_data, 'matrix_local') else None,
        )
        
        return skinned_data

    def _export_fbx(self, raw_data: 'RawData', output_path: Path) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        raw_data.export_fbx(path=str(output_path), use_tail=True)
        return output_path

    def _convert_to_glb(self, fbx_file: Path) -> Path:
        import bpy
        from src.data.extract import clean_bpy
        
        clean_bpy()
        bpy.ops.import_scene.fbx(filepath=str(fbx_file), ignore_leaf_bones=False)
        
        glb_file = fbx_file.with_suffix(".glb")
        bpy.ops.export_scene.gltf(
            filepath=str(glb_file),
            export_format='GLB',
        )
        
        return glb_file

    def _merge_with_original(
        self,
        skinned_data: 'RawData',
        original_file: Path,
        output_path: Path,
    ) -> Path:
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
        seed = get_seed(input.seed)
        work_dir = Path(tempfile.mkdtemp(prefix="unirig_"))
        
        try:
            os.chdir(self.repo_dir)
            
            input_file = self._download_input_file(input.mesh_file, work_dir)
            raw_data = self._extract_mesh(input_file, input.faces_target_count)
            skeleton_data = self._predict_skeleton(raw_data, seed)
            
            skeleton_output = work_dir / "skeleton.fbx"
            self._export_fbx(skeleton_data, skeleton_output)
            
            skinned_data = self._predict_skin(skeleton_data)
            
            rigged_output = work_dir / "rigged.fbx"
            self._merge_with_original(skinned_data, input_file, rigged_output)
            
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

if __name__ == "__main__":
    app = fal.wrap_app(UniRig)
    app()
