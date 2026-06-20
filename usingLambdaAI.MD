# Running Python Scripts on Lambda Labs GPU Cloud

A step-by-step guide for running Python scripts with LLMs on Lambda Labs.

---

## Prerequisites

- A Lambda Labs account (https://cloud.lambda.ai)
- An SSH key added to your Lambda account
- Your Python script and input data ready locally

---

## Step 1: Launch an Instance

1. Go to https://cloud.lambda.ai/instances
2. Click **Launch instance**
3. Choose an **instance type** based on your needs:
   - Small models (< 3B parameters): Any GPU works (A10, RTX A6000)
   - Medium models (7B-8B): A10 (24GB) or A100
   - Large models (70B+): A100 (80GB) or multiple GPUs
4. Select a **region** with availability
5. For **filesystem**: Select "Don't attach a filesystem" (unless you need persistent storage)
6. Select your **SSH key**
7. Click **Launch**

Wait 3-5 minutes for the instance to boot. You'll see an IP address appear in the dashboard.

---

## Step 2: Connect to Your Instance

Open a terminal on your local machine and run:

```bash
# Connect to the instance via SSH
# Replace <INSTANCE-IP> with your instance's IP address from the dashboard
ssh -i ~/.ssh/lambda_key.pem ubuntu@<INSTANCE-IP>
```

If prompted about authenticity, type `yes` to continue.

You should see a prompt like `ubuntu@129-153-77-37:~$`

---

## Step 3: Upload Your Files

Open a **new local terminal window** (keep the SSH session open).

```bash
# Upload your Python script
# Replace the paths as needed
scp -i ~/.ssh/lambda_key.pem ~/Desktop/your_script.py ubuntu@<INSTANCE-IP>:~/

# Upload your input data
scp -i ~/.ssh/lambda_key.pem ~/Desktop/input.csv ubuntu@<INSTANCE-IP>:~/
```

Go back to your SSH terminal and verify the files arrived:

```bash
# List files in home directory
ls -la
```

---

## Step 4: Set Up a Clean Python Environment

Lambda instances have preinstalled packages that can conflict with pip installs. Always create a fresh virtual environment:

```bash
# Create a clean virtual environment (no system packages)
python3 -m venv ~/cleanenv

# Activate the virtual environment
source ~/cleanenv/bin/activate
```

Your prompt should now show `(cleanenv)` at the beginning.

---

## Step 5: Install Required Packages

```bash
# Install PyTorch (for GPU support) and transformers (for LLMs)
pip install torch transformers

# Install pandas (for CSV handling)
pip install pandas

# Install any other packages your script needs
# pip install <package_name>
```


---

## Step 6: Run Your Script

```bash
# Run your Python script
python your_script.py
```

Monitor the output. For long-running jobs, you can use `tmux` or `screen` to keep the process running if you disconnect.

---

## Step 7: Download Your Results

Once the script completes, open your **local terminal** (not SSH) and download the output:

```bash
# Download results to your Desktop
# Replace output.csv with your actual output filename
scp -i ~/.ssh/lambda_key.pem ubuntu@<INSTANCE-IP>:~/output.csv ~/Desktop/
```

---

## Step 8: Terminate Your Instance (IMPORTANT!)

**You are billed as long as the instance is running, even if your script is finished.**

1. Go to https://cloud.lambda.ai/instances
2. Find your instance
3. Click **Terminate**
4. Confirm termination

---

## Quick Reference: Common Commands

### On your local machine:

```bash
# Connect to instance
ssh -i ~/.ssh/lambda_key.pem ubuntu@<INSTANCE-IP>

# Upload a file
scp -i ~/.ssh/lambda_key.pem /local/path/file.txt ubuntu@<INSTANCE-IP>:~/

# Download a file
scp -i ~/.ssh/lambda_key.pem ubuntu@<INSTANCE-IP>:~/file.txt /local/path/
```

### On the Lambda instance:

```bash
# Create virtual environment
python3 -m venv ~/cleanenv

# Activate virtual environment
source ~/cleanenv/bin/activate

# Deactivate virtual environment
deactivate

# Check GPU is available
python -c "import torch; print(torch.cuda.is_available())"

# Check available GPU memory
nvidia-smi
```

---

## Troubleshooting

### "Permission denied (publickey)" when connecting
Your SSH key isn't being found. Specify it explicitly:
```bash
ssh -i ~/.ssh/lambda_key.pem ubuntu@<INSTANCE-IP>
```

### "ModuleNotFoundError" when running script
Make sure you've activated your virtual environment:
```bash
source ~/cleanenv/bin/activate
```
Then install the missing package:
```bash
pip install <package_name>
```

### Model class not found (e.g., "Could not find GPTNeoForCausalLM")
This usually means a corrupted or conflicting transformers install. Create a fresh environment:
```bash
deactivate
rm -rf ~/cleanenv
python3 -m venv ~/cleanenv
source ~/cleanenv/bin/activate
pip install torch transformers
```

### Script runs out of GPU memory
- Try a smaller model
- Reduce batch size in your script
- Choose an instance with more GPU memory

## Cost Tips

- **Terminate instances immediately** after your job completes
- Use smaller/cheaper GPUs for testing, then scale up for full runs
- Check instance pricing at https://lambda.ai/service/gpu-cloud