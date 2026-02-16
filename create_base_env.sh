
# cd /deck/groups/MotionRL/MotionDiffusionBase

# conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
# conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# conda create -n motion_diffusion_base python=3.10 -y
# conda activate motion_diffusion_base

# # # PyTorch nightly
# pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128 --extra-index-url https://pypi.nvidia.com
# pip install --pre torchvision --index-url https://download.pytorch.org/whl/nightly/cu128 --extra-index-url https://pypi.nvidia.com --no-deps
# pip install pillow numpy

# # Requirements base
pip install -r requirements_clean.txt

# pip install scikit-learn matplotlib opencv-python-headless numexpr
# pip install moviepy==1.0.3
# pip install h5py
# pip install chumpy --no-build-isolation
pip install git+https://github.com/openai/CLIP.git
# pip install spacy
# python -m spacy download en_core_web_sm
# pip install "git+https://github.com/facebookresearch/pytorch3d.git"

# # Fix pyrender
# sed -i 's/np\.infty/np.inf/g' $(python -c "import pyrender; print(pyrender.__file__.replace('__init__.py', 'mesh.py'))")

# echo "Setup completato!"

