import subprocess
# if want to set a specific GPU to train, run "export CUDA_VISIBLE_DEVICES=2" in terminal first and then batch_txt2img.py
# python3 scripts/txt2img.py --ckpt  checkpoints/v2-1_768-ema-pruned.ckpt --W 768 --H 768 --prompt  "A patient reclining in a dental chair, the overhead light illuminating their mouth as the dentist examines their teeth and discusses their oral health."

# "checkpoints/v2-1_768-ema-pruned.ckpt" or "checkpoints/v2-1_512-ema-pruned.ckpt"
ckpt = "checkpoints/v2-1_512-ema-pruned.ckpt"

if "512" in ckpt:
    W,H = 512,512
    n_samples = 3
elif "768" in ckpt:
    W,H = 768,768
    n_samples = 3

print("load ckpt:", ckpt)
print("W: ", W)
print("H: ", H)
def run_main_py(prompt_list):
    print("Size: ",len(prompt_list))
    for i,prompt in enumerate(prompt_list):
        print("Running: ", i)
        result = subprocess.run(["python3", "scripts/txt2img.py", "--ckpt", ckpt, "--prompt", prompt, "--W", str(W), "--H", str(H), "--n_samples", str(n_samples)], capture_output=True, text=True)
        print("Output:")
        print(result.stdout)
        print("Errors:")
        print(result.stderr)
        print("-" * 30)
    
if __name__ == "__main__":
    prompt_list = [
  
   ]
    run_main_py(prompt_list)
