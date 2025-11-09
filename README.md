# pre-phd-genomics
repo to do phd prep and analysis


# Setup Instructions

1. After cloning the repository, navigate to the project directory and execute the installation script:

   ```bash
   git clone <repository_url>
   cd pre-phd-genomics
   ./install.sh
   ```
## Git Workflow (Mayo GCP VM → Laptop → GitHub)

Mayo's GCP VM blocks git push to GitHub. Use laptop as intermediary:

### One-time setup (on laptop):
```bash
# Clone from GitHub
git clone https://github.com/meehljd/pre-phd-genomics.git
cd pre-phd-genomics

# Add VM as remote
git remote add mayo ext_meehl_joshua_mayo_edu@a3-3tb-disk:~/pre-phd-genomics
```

### Regular workflow:
```bash
# 1. Work on Mayo VM, commit locally
# (on VM)
git add .
git commit -m "your changes"

# 2. Pull from VM to laptop
# (on laptop)
git pull mayo master --no-rebase

# 3. Push to GitHub
git push origin master
```

### Notes:
- VM can pull from GitHub: `git pull origin master`
- VM cannot push to GitHub (network policy)
- All pushes must go through laptop or Cloud Shell
