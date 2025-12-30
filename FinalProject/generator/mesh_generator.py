import torch
from shap_e.diffusion.sample import sample_latents
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.models.download import load_model, load_config
from shap_e.util.notebooks import decode_latent_mesh
import os


class ShapeEMeshGenerator:
    def __init__(
        self,
        device=None,
        guidance_scale=15.0,
        karras_steps=64
    ):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.guidance_scale = guidance_scale
        self.karras_steps = karras_steps

        # modeller
        self.transmitter = load_model("transmitter", device=self.device)
        self.text_model = load_model("text300M", device=self.device)
        self.diffusion = diffusion_from_config(load_config("diffusion"))

    def generate_meshes(
        self,
        prompt: str,
        batch_size: int = 1,
        output_prefix: str = "mesh"
    ):
        """
        prompt: text description (örn: 'a green couch')
        batch_size: kaç adet mesh üretilecek
        output_prefix: dosya isimleri için prefix
        """
        output_dir = os.path.dirname(output_prefix)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)


        latents = sample_latents(
            batch_size=batch_size,
            model=self.text_model,
            diffusion=self.diffusion,
            guidance_scale=self.guidance_scale,
            model_kwargs=dict(texts=[prompt] * batch_size),
            progress=True,
            clip_denoised=True,
            use_fp16=True,
            use_karras=True,
            karras_steps=self.karras_steps,
            sigma_min=1e-3,
            sigma_max=160,
            s_churn=0,
        )

        output_files = []

        for i, latent in enumerate(latents):
            mesh = decode_latent_mesh(self.transmitter, latent).tri_mesh()

            ply_path = f"{output_prefix}_{i}.ply"
            obj_path = f"{output_prefix}_{i}.obj"

            with open(ply_path, "wb") as f:
                mesh.write_ply(f)

            with open(obj_path, "w") as f:
                mesh.write_obj(f)

            output_files.append({
                "ply": ply_path,
                "obj": obj_path
            })

        return output_files
