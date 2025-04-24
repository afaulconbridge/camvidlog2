## prepare machine

```bash
sudo apt update

sudo apt install -y pipx
pipx ensurepath

pipx install uv

# for NFS mounting
sudo apt install -y cifs-utils
```

## prepare environment

```bash
# secrets
if [ -f .secrets.env ]; then
    set -a
    . .secrets.env
    set +a
fi

sudo mkdir -p /mnt/shared
sudo mount -t cifs -o rw,vers=3.0,user=adam,pass=${CIFS_PASS},dir_mode=0775,file_mode=0775,uid=1000,gid=9999 //192.168.1.102/shared-mirror /mnt/shared
```

## run

```bash
uv run python run.py --help
```
